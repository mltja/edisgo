import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from datetime import datetime
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import (
    check_mapping,
    optimize,
    prepare_time_invariant_parameters,
    setup_model,
    update_model,
)
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative
from edisgo.tools.tools import convert_impedances_to_mv, extract_feeders_nx

optimize_storage = False
optimize_ev = True
objective = "residual_load"


def run_optimized_charging_feeder_parallel(
    grid_feeder_tuple, mapping_dir, working_dir, redirect_output
):
    feeder_dir = os.path.join(working_dir, "feeder")
    opt_feeder_dir = os.path.join(working_dir, "opt_feeder")
    grid_id = grid_feeder_tuple[0]
    feeder_id = grid_feeder_tuple[1]

    if redirect_output:
        old_output = sys.stdout
        file = os.path.join(working_dir, "feeder_{}.out".format(feeder_id))
        sys.stdout = open(file, "a")

    timesteps_per_iteration = 24 * 4
    iterations_per_era = 7
    overlap_interations = 24
    solver = "gurobi"
    solver = "gurobi_direct"

    config = pd.Series(
        {
            "objective": objective,
            "solver": solver,
            "timesteps_per_iteration": timesteps_per_iteration,
            "iterations_per_era": iterations_per_era,
        }
    )

    feeder_dir = os.path.join(feeder_dir, feeder_id)
    opt_feeder_dir = os.path.join(opt_feeder_dir, feeder_id)

    os.makedirs(opt_feeder_dir, exist_ok=True)

    if len(os.listdir(opt_feeder_dir)) == 225:
        print("Feeder {} of grid {} already solved.".format(feeder_id, grid_id))
        return
    else:
        charging_start = None
        energy_level_start = None
        start_iter = 0
    config.to_csv(opt_feeder_dir + "/config.csv")

    try:
        edisgo_orig = import_edisgo_from_files(feeder_dir, import_timeseries=True)

        print("eDisGo object imported.")

        edisgo_obj = convert_impedances_to_mv(edisgo_orig)

        print("Converted impedances to mv.")

        downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(
            edisgo_obj.topology
        )

        downstream_nodes_matrix = downstream_nodes_matrix.loc[
            edisgo_obj.topology.buses_df.index, edisgo_obj.topology.buses_df.index
        ]
        print("Downstream node matrix imported.")

        flexibility_bands = pd.DataFrame()
        for use_case in ["home", "work"]:
            flexibility_bands_tmp = pd.read_csv(
                os.path.join(
                    mapping_dir,
                    "ev_flexibility_bands_{}_{}.csv".format(grid_id, use_case),
                ),
                index_col=0,
                dtype=np.float32,
            )
            rename_dict = {
                col: col + "_{}".format(use_case)
                for col in flexibility_bands_tmp.columns
            }
            flexibility_bands_tmp.rename(columns=rename_dict, inplace=True)
            flexibility_bands = pd.concat(
                [flexibility_bands, flexibility_bands_tmp], axis=1
            )
        # remove numeric problems
        flexibility_bands.loc[
            :,
            flexibility_bands.columns[flexibility_bands.columns.str.contains("power")],
        ] = (
            flexibility_bands[
                flexibility_bands.columns[
                    flexibility_bands.columns.str.contains("power")
                ]
            ]
            + 1e-6
        ).values
        print("Flexibility bands imported.")
        mapping_home = pd.read_csv(
            mapping_dir + "/cp_data_home_within_grid_{}.csv".format(grid_id)
        ).set_index("edisgo_id")
        mapping_work = pd.read_csv(
            mapping_dir + "/cp_data_work_within_grid_{}.csv".format(grid_id)
        ).set_index("edisgo_id")
        mapping_home["use_case"] = "home"
        mapping_work["use_case"] = "work"
        mapping = pd.concat(
            [mapping_work, mapping_home], sort=False
        )  # , mapping_hpc, mapping_public
        print("Mapping imported.")

        # extract data for feeder
        mapping = mapping.loc[
            mapping.index.isin(edisgo_obj.topology.charging_points_df.index)
        ]
        cp_identifier = [
            "_".join(
                [
                    str(mapping.loc[cp, "ags"]),
                    str(mapping.loc[cp, "cp_idx"]),
                    mapping.loc[cp, "use_case"],
                ]
            )
            for cp in mapping.index
        ]
        flex_band_identifier = []
        for cp in cp_identifier:
            flex_band_identifier.append("lower_" + cp)
            flex_band_identifier.append("upper_" + cp)
            flex_band_identifier.append("power_" + cp)
        flexibility_bands = flexibility_bands[flex_band_identifier]

        check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
        print("Data checked. Please pay attention to warnings.")

        # Create dict with time invariant parameters
        parameters = prepare_time_invariant_parameters(
            edisgo_obj,
            downstream_nodes_matrix,
            pu=False,
            optimize_storage=False,
            optimize_ev_charging=True,
            cp_mapping=mapping,
        )
        print("Time-invariant parameters extracted.")

        energy_level = {}
        charging_ev = {}

        for iteration in range(
            int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)
        ):

            print("Starting optimisation for week {}.".format(iteration))
            start_time = (iteration * timesteps_per_iteration) % 672
            if iteration % iterations_per_era != iterations_per_era - 1:
                timesteps = edisgo_obj.timeseries.timeindex[
                    iteration
                    * timesteps_per_iteration : (iteration + 1)
                    * timesteps_per_iteration
                    + overlap_interations
                ]
                flexibility_bands_week = flexibility_bands.iloc[
                    start_time : start_time
                    + timesteps_per_iteration
                    + overlap_interations
                ].set_index(timesteps)
                energy_level_end = None
            else:
                timesteps = edisgo_obj.timeseries.timeindex[
                    iteration
                    * timesteps_per_iteration : (iteration + 1)
                    * timesteps_per_iteration
                ]
                flexibility_bands_week = flexibility_bands.iloc[
                    start_time : start_time + timesteps_per_iteration
                ].set_index(timesteps)
                energy_level_end = True
            # Check if problem will be feasible
            if charging_start is not None:
                low_power_cp = []
                violation_lower_bound_cp = []
                violation_upper_bound_cp = []
                for cp_tmp in energy_level_start.index:
                    flex_id_tmp = "_".join(
                        [
                            str(mapping.loc[cp_tmp, "ags"]),
                            str(mapping.loc[cp_tmp, "cp_idx"]),
                            mapping.loc[cp_tmp, "use_case"],
                        ]
                    )
                    if (
                        energy_level_start[cp_tmp]
                        > flexibility_bands_week.iloc[0]["upper_" + flex_id_tmp]
                    ):
                        if (
                            energy_level_start[cp_tmp]
                            - flexibility_bands_week.iloc[0]["upper_" + flex_id_tmp]
                            > 1e-4
                        ):
                            raise ValueError(
                                "Optimisation should not return values higher than upper bound. "
                                "Problem for {}. Initial energy level is {}, but upper bound {}.".format(
                                    cp_tmp,
                                    energy_level_start[cp_tmp],
                                    flexibility_bands_week.iloc[0][
                                        "upper_" + flex_id_tmp
                                    ],
                                )
                            )
                        else:
                            energy_level_start[cp_tmp] = (
                                flexibility_bands_week.iloc[0]["upper_" + flex_id_tmp]
                                - 1e-6
                            )
                            violation_upper_bound_cp.append(cp_tmp)
                    if (
                        energy_level_start[cp_tmp]
                        < flexibility_bands_week.iloc[0]["lower_" + flex_id_tmp]
                    ):
                        if (
                            -energy_level_start[cp_tmp]
                            + flexibility_bands_week.iloc[0]["lower_" + flex_id_tmp]
                            > 1e-4
                        ):
                            raise ValueError(
                                "Optimisation should not return values lower than lower bound. "
                                "Problem for {}. Initial energy level is {}, but lower bound {}.".format(
                                    cp_tmp,
                                    energy_level_start[cp_tmp],
                                    flexibility_bands_week.iloc[0][
                                        "lower_" + flex_id_tmp
                                    ],
                                )
                            )
                        else:
                            energy_level_start[cp_tmp] = (
                                flexibility_bands_week.iloc[0]["lower_" + flex_id_tmp]
                                + 1e-6
                            )
                            violation_lower_bound_cp.append(cp_tmp)
                    if charging_start[cp_tmp] < 1e-5:
                        low_power_cp.append(cp_tmp)
                        charging_start[cp_tmp] = 0
                print("Very small charging power: {}, set to 0.".format(low_power_cp))
                print(
                    "Charging points {} violates lower bound.".format(
                        violation_lower_bound_cp
                    )
                )
                print(
                    "Charging points {} violates upper bound.".format(
                        violation_upper_bound_cp
                    )
                )
            try:
                model = update_model(
                    model,
                    timesteps,
                    parameters,
                    optimize_storage=optimize_storage,
                    optimize_ev=optimize_ev,
                    energy_band_charging_points=flexibility_bands_week,
                    charging_start=charging_start,
                    energy_level_start=energy_level_start,
                    energy_level_end=energy_level_end,
                )
            except:
                print("Exception: {}".format(sys.exc_info()))
                traceback.print_exc(file=sys.stdout)
                model = setup_model(
                    parameters,
                    timesteps,
                    objective=objective,
                    optimize_storage=False,
                    optimize_ev_charging=True,
                    energy_band_charging_points=flexibility_bands_week,
                    charging_start=charging_start,
                    energy_level_start=energy_level_start,
                    energy_level_end=energy_level_end,
                    overlap_interations=overlap_interations,
                )

            print("Set up model for week {}.".format(iteration))

            result_dict = optimize(model, solver)
            charging_ev[iteration] = result_dict["x_charge_ev"]
            energy_level[iteration] = result_dict["energy_level_cp"]

            if iteration % iterations_per_era != iterations_per_era - 1:
                charging_start = charging_ev[iteration].iloc[-overlap_interations]
                energy_level_start = energy_level[iteration].iloc[-overlap_interations]
            else:
                charging_start = None
                energy_level_start = None

            print("Finished optimisation for week {}.".format(iteration))
            for res_name, res in result_dict.items():
                res.astype(np.float16).to_csv(
                    opt_feeder_dir
                    + "/{}_{}_{}_{}.csv".format(res_name, grid_id, feeder_id, iteration)
                )
            print("Saved results for week {}.".format(iteration))

    except:
        print("Exception: {}".format(sys.exc_info()))
        traceback.print_exc(file=sys.stdout)
        print(
            "Something went wrong with feeder {} of grid {}".format(feeder_id, grid_id)
        )

        if "iteration" in locals():
            if iteration >= 1:
                charging_start = charging_ev[iteration - 1].iloc[-overlap_interations]
                charging_start.to_csv(
                    "results/tests/charging_start_{}_{}_{}.csv".format(
                        grid_id, feeder_id, iteration
                    )
                )
                energy_level_start = energy_level[iteration - 1].iloc[
                    -overlap_interations
                ]
                energy_level_start.to_csv(
                    "results/tests/energy_level_start_{}_{}_{}.csv".format(
                        grid_id, feeder_id, iteration
                    )
                )

    if redirect_output:
        sys.stdout = old_output

    return grid_feeder_tuple


def combine_results_for_grid(feeders, grid_id, res_dir, res_name):
    res_grid = pd.DataFrame()
    for feeder_id in feeders:
        res_feeder = pd.DataFrame()
        for i in range(14):
            try:
                res_feeder_tmp = pd.read_csv(
                    res_dir
                    + "/{}/{}_{}_{}_{}.csv".format(
                        feeder_id, res_name, grid_id, feeder_id, i
                    ),
                    index_col=0,
                    parse_dates=True,
                )
                res_feeder = pd.concat([res_feeder, res_feeder_tmp], sort=False)
            except:
                print("Exception: {}".format(sys.exc_info()))
                traceback.print_exc(file=sys.stdout)
                print(
                    "Results for feeder {} in grid {} could not be loaded.".format(
                        feeder_id, grid_id
                    )
                )
        try:
            res_grid = pd.concat([res_grid, res_feeder], axis=1, sort=False)
        except:
            print("Exception: {}".format(sys.exc_info()))
            traceback.print_exc(file=sys.stdout)
            print("Feeder {} not added".format(feeder_id))
    res_grid = res_grid.loc[~res_grid.index.duplicated(keep="last")]
    return res_grid


def extract_feeders(edisgo_dir, redirect_output=True):
    working_dir = os.path.join(edisgo_dir, "working_dir")
    feeder_dir = os.path.join(working_dir, "feeder")

    edisgo_obj = import_edisgo_from_files(
        edisgo_dir, import_topology=True, import_timeseries=True
    )
    os.makedirs(working_dir, exist_ok=True)

    if not os.path.isdir(feeder_dir):
        os.makedirs(feeder_dir, exist_ok=True)

        if redirect_output:
            old_output = sys.stdout
            file = open(os.path.join(working_dir, "extract_feeders_nx.out"), "a")
            sys.stdout = file

        extract_feeders_nx(edisgo_obj, working_dir)

        if redirect_output:
            sys.stdout = old_output
    else:
        print("Feeders already extracted")


def optimize_feeders(
    edisgo_dir,
    mapping_dir,
    multiprocessing=False,
    single_feeder=False,
    redirect_output=True,
):
    working_dir = os.path.join(edisgo_dir, "working_dir")
    feeder_dir = os.path.join(working_dir, "feeder")
    grid_id = edisgo_dir.split("/")[-1].split("_")[0]

    if redirect_output:
        old_output = sys.stdout
        file = open(os.path.join(working_dir, "optimize_feeders.out"), "a", buffering=1)
        sys.stdout = file

    grid_id_feeder_tuples = []
    for feeder in os.listdir(feeder_dir):
        if os.path.isdir(os.path.join(feeder_dir, feeder)):
            grid_id_feeder_tuples.append(
                (
                    (grid_id, feeder),
                    mapping_dir,
                    working_dir,
                    redirect_output,
                )
            )
    grid_id_feeder_tuples.sort(key=lambda tuple: int(tuple[0][1]))

    if single_feeder:
        grid_id_feeder_tuples = [grid_id_feeder_tuples[int(single_feeder)]]

    print("Feeders to optimize:")
    for feeder in grid_id_feeder_tuples:
        print(feeder)

    if multiprocessing:

        def f_callback(x):
            print(
                "Callback:",
                x,
                file=open(os.path.join(working_dir, "callback.out"), "a"),
            )
            return x

        def f_error_callback(x):
            print(
                "Error_callback:",
                x,
                file=open(os.path.join(working_dir, "callback.out"), "a"),
            )
            return x

        with mp.Pool(processes=4, maxtasksperchild=1) as pool:
            multiple_results = []
            for arguments in grid_id_feeder_tuples:
                multiple_results.append(
                    pool.apply_async(
                        run_optimized_charging_feeder_parallel,
                        arguments,
                        callback=f_callback,
                        error_callback=f_error_callback,
                    )
                )

            should_we_continue = False
            while not should_we_continue:
                status_dict = {}
                mem_total = 0
                for process in mp.active_children():
                    try:
                        pid = process.pid
                        proc = psutil.Process(pid)
                        status_dict[pid] = {}
                        status_dict[pid]["status"] = proc.status()
                        status_dict[pid]["cpu"] = proc.cpu_percent(interval=0.1)
                        status_dict[pid]["mem"] = proc.memory_percent()
                        mem_total = mem_total + status_dict[pid]["mem"]
                    except:
                        status_dict = {'status': {'status': 'exception'}}
                        print("Exception: {}".format(sys.exc_info()))
                        traceback.print_exc(file=sys.stdout)
                print(datetime.now().isoformat(sep="T", timespec="seconds"))
                for status in status_dict.items():
                    print(status)
                print("Mem total: {}".format(mem_total))
                print(
                    ["sleeping" == status["status"] for status in status_dict.values()]
                )
                should_we_continue = all(
                    ["sleeping" == status["status"] for status in status_dict.values()]
                )
                print(should_we_continue)
                time.sleep(10)
    else:
        for arguments in grid_id_feeder_tuples:
            print("Optimize feeder: {}".format(arguments[0][1]))
            run_optimized_charging_feeder_parallel(*arguments)

    if redirect_output:
        sys.stdout = old_output


def extract_edisgo_object(edisgo_dir, save=False):
    working_dir = os.path.join(edisgo_dir, "working_dir")
    opt_feeder_dir = os.path.join(working_dir, "opt_feeder")
    opt_dir = os.path.join(working_dir, "opt")

    grid_id = edisgo_dir.split("/")[-1].split("_")[0]

    feeders = []
    for feeder in os.listdir(opt_feeder_dir):
        feeders.append(feeder)

    x_charge_ev_grid = combine_results_for_grid(
        feeders, grid_id, opt_feeder_dir, "x_charge_ev"
    )

    edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    # update timeseries of original object
    tmp = edisgo.timeseries.charging_points_active_power
    tmp.update(x_charge_ev_grid)
    edisgo.timeseries.charging_points_active_power = tmp

    if save:
        print("Save optimised edisgo object to file")
        edisgo.save(opt_dir)

    return edisgo
