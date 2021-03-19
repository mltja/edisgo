import os
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import results_helper_functions

from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools import pypsa_io
from edisgo.tools.tools import assign_feeder, get_path_length_to_station
from pathlib import Path
from time import perf_counter

import logging
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# # possible options for scenario are "dumb_charging" and "smart_charging"
# scenario = "test"
#
# #results_base_path = "/home/local/RL-INSTITUT/birgit.schachler/rli-daten_02/open_BEA_Berechnungen"
# results_base_path = "/home/birgit/Schreibtisch/Elia_Ergebnisse"
# if scenario == "dumb":
#     # results_path = os.path.join(
#     #     results_base_path, "2020-12-11_22-13_simbev_scenario_dumb_charging")
#     results_path = os.path.join(
#         results_base_path, "2020-10-19_19-57_elia_scenario_dumb_charging")
# elif scenario == "reduced":
#     results_path = os.path.join(
#         results_base_path, "2020-12-11_22-14_simbev_scenario_reduced_charging")
# elif scenario == "grouped":
#     results_path = os.path.join(
#         results_base_path, "2020-12-13_18-51_simbev_scenario_grouped_charging")
# elif scenario == "residual_load":
#     results_path = os.path.join(
#         results_base_path, "2021-01-06_17-23_simbev_scenario_residual_load_charging")
# elif scenario == "test":
#     results_path = os.path.join(
#         results_base_path, "Test_LV")
# else:
#     raise ValueError
#
# mv_grid_ids = [176]#[176, 177, 1056, 1423, 1574, 1690, 1811, 1839,
#                #2079, 2095, 2534, 3008, 3280] # 566, 3267

# num_threads = 1
# curtailment_step = 0.1 # 0.2
max_iterations = 1000


def _overwrite_edisgo_timeseries(edisgo, pypsa_network):
    """
    Overwrites generator and load time series in edisgo after curtailment.

    pypsa_network contains the curtailed time series that are written to
    edisgo object.

    """

    # overwrite time series in edisgo
    time_steps = pypsa_network.generators_t.p_set.index

    # generators: overwrite time series for all except slack
    gens = pypsa_network.generators[
        pypsa_network.generators.control != "Slack"].index
    edisgo.timeseries._generators_active_power.loc[
        time_steps, gens] = pypsa_network.generators_t.p_set.loc[
        time_steps, gens]
    edisgo.timeseries._generators_reactive_power.loc[
        time_steps, gens] = pypsa_network.generators_t.q_set.loc[
        time_steps, gens]

    # loads: distinguish between charging points and conventional loads
    loads = pypsa_network.loads[
        pypsa_network.loads.index.isin(edisgo.topology.loads_df.index)
    ].index

    if not len(loads) == 0:
        edisgo.timeseries._loads_active_power.loc[
            time_steps, loads] = pypsa_network.loads_t.p_set.loc[
            time_steps, loads]
        edisgo.timeseries._loads_reactive_power.loc[
            time_steps, loads] = pypsa_network.loads_t.q_set.loc[
            time_steps, loads]

    if not edisgo.topology.charging_points_df.empty:
        charging_points = pypsa_network.loads[
            pypsa_network.loads.index.isin(edisgo.topology.charging_points_df.index)
        ].index

        if not len(charging_points) == 0:
            edisgo.timeseries._charging_points_active_power.loc[
                time_steps, charging_points] = pypsa_network.loads_t.p_set.loc[
                time_steps, charging_points]
            edisgo.timeseries._charging_points_reactive_power.loc[
                time_steps, charging_points] = pypsa_network.loads_t.q_set.loc[
                time_steps, charging_points]


def _save_results_when_curtailment_failed(edisgo_obj, results_dir, mode):

    edisgo_obj.save(
        os.path.join(
            results_dir,
            "edisgo_curtailment_{}".format(mode)
        ),
        parameters="powerflow_results")

    rel_load = results_helper_functions.relative_load(edisgo_obj)
    rel_load.to_csv(
        os.path.join(
            results_dir,
            "edisgo_curtailment_{}".format(mode),
            "relative_load.csv"
        )
    )
    voltage_dev = results_helper_functions.voltage_diff(edisgo_obj)
    voltage_dev.to_csv(
        os.path.join(
            results_dir,
            "edisgo_curtailment_{}".format(mode),
            "voltage_deviation.csv"
        )
    )


def _curtail(pypsa_network, gens, loads, time_steps, curtailment_step=0.05): # TODO

    # get time series for loads and generators
    gens_ts = pypsa_network.generators_t.p_set.loc[
        time_steps, gens]
    loads_ts = pypsa_network.loads_t.p_set.loc[
        time_steps, loads]

    # evaluate whether it is a load or feed-in case
    # calculate residual load
    residual_load = gens_ts.sum(axis=1) - loads_ts.sum(axis=1)

    # get time steps where to curtail generators and where to
    # curtail loads
    ts_curtail_gens = residual_load[residual_load > 0].index
    ts_curtail_loads = residual_load[residual_load < 0].index

    if not ts_curtail_gens.empty:
        # curtail generators by specified curtailment factor
        # active power
        pypsa_network.generators_t.p_set.loc[ts_curtail_gens, gens] = (
            gens_ts.loc[ts_curtail_gens, :] -
            curtailment_step *
            gens_ts.loc[ts_curtail_gens, :])
        # reactive power
        tmp = pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens]
        pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens] = (
            tmp - curtailment_step * tmp)

    if not ts_curtail_loads.empty:
        # curtail loads by specified curtailment factor
        # active power
        pypsa_network.loads_t.p_set.loc[ts_curtail_loads, loads] = (
            loads_ts.loc[ts_curtail_loads, :] -
            curtailment_step *
            loads_ts.loc[ts_curtail_loads, :])
        # reactive power
        tmp = pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads]
        pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads] = (
            tmp - curtailment_step * tmp)

    if ts_curtail_loads.empty and ts_curtail_gens.empty:
        print("Nothing to curtail. I'm stuck in a while loop.")

    return pypsa_network


def _calculate_curtailed_energy(pypsa_network_orig, pypsa_network):
    gens = pypsa_network_orig.generators[
        pypsa_network_orig.generators.control != "Slack"].index
    curtailed_feedin_ts = (
            pypsa_network_orig.generators_t.p_set.loc[:, gens] -
            pypsa_network.generators_t.p_set.loc[:, gens]
    )
    curtailed_load_ts = (
            pypsa_network_orig.loads_t.p_set -
            pypsa_network.loads_t.p_set
    )
    return curtailed_feedin_ts, curtailed_load_ts


def my_pf(pypsa, timesteps, mode="lpf", x_tol=1e-5):
    if mode == "lpf":
        pypsa.lpf(timesteps)

        pf_results = pypsa.pf(timesteps, use_seed=True, x_tol=x_tol)

    elif mode == "iteratively":
        gen_p_set_orig = pypsa.generators_t["p_set"].copy()
        gen_q_set_orig = pypsa.generators_t["q_set"].copy()
        load_p_set_orig = pypsa.loads_t["p_set"].copy()
        load_q_set_orig = pypsa.loads_t["q_set"].copy()

        pypsa.generators_t["p_set"] = gen_p_set_orig.multiply(0.1)
        pypsa.generators_t["q_set"] = gen_q_set_orig.multiply(0.1)
        pypsa.loads_t["p_set"] = load_p_set_orig.multiply(0.1)
        pypsa.loads_t["q_set"] = load_q_set_orig.multiply(0.1)

        pypsa.lpf(timesteps)

        pf_results = pypsa.pf(timesteps, use_seed=True, x_tol=x_tol)

        for i in np.arange(0.2, 1.1, 0.1):
            pypsa.generators_t["p_set"] = gen_p_set_orig.multiply(i)
            pypsa.generators_t["q_set"] = gen_q_set_orig.multiply(i)
            pypsa.loads_t["p_set"] = load_p_set_orig.multiply(i)
            pypsa.loads_t["q_set"] = load_q_set_orig.multiply(i)

            pf_results = pypsa.pf(timesteps, use_seed=True, x_tol=x_tol)

    elif mode == "use_seed":
        pf_results = pypsa.pf(timesteps, use_seed=True, x_tol=x_tol)

    else:
        pf_results = pypsa.pf(timesteps, x_tol=x_tol)

    return pf_results, pypsa

def curtailment_lv_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day,
        pypsa_network=None, lv_grid=False):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    # get voltage issues in LV
    lv_buses = edisgo.topology.buses_df.lv_feeder.dropna().index
    if not voltage_dev.empty:
        voltage_dev_lv = voltage_dev.loc[:, lv_buses]
    else:
        voltage_dev_lv = voltage_dev.copy()
    voltage_issues = voltage_dev_lv[
        voltage_dev_lv != 0].dropna(how="all").dropna(
            axis=1, how="all")
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        if pypsa_network is None:
            pypsa_network = edisgo.to_pypsa(
                timesteps=time_steps_issues,
            )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].unique()

            elia_logger.debug(
                "Number of LV feeders with voltage issues: {}".format(
                    len(feeders)))
            elia_logger.debug(
                "Number of time steps with voltage issues in LV: {}".format(
                    len(time_steps_issues)))

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.lv_feeder == feeder].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses)].index)

                # get time steps with voltage issues in feeder
                ts_issues = voltage_issues.loc[
                            :, buses].dropna(how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with voltage issues
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError(
                    "Power flow analysis did not converge for the "
                    "following time steps: {}.".format(
                        time_steps_issues[
                            ~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # get voltage issues in LV
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            voltage_dev_lv = voltage_dev.loc[:, lv_buses]
            voltage_issues = voltage_dev_lv[
                voltage_dev_lv != 0].dropna(how="all").dropna(
                axis=1, how="all")
            buses_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if not lv_grid:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(time_steps_issues) > 0:

                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir, "{}_{}_{}_lv_voltage".format(scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve LV voltage "
                                 "issues.")
        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)

        # calculate curtailment
        # ToDo: Why does .at not work?
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV voltage issues to solve.")
        pass

    if lv_grid:
        return pypsa_network, curtailment
    else:
        return curtailment


def curtailment_mvlv_stations_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day, mode="mvlv", mv_grid_is_agg=False):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    # get stations with voltage issues
    mvlv_stations = edisgo.topology.transformers_df.bus1.values
    voltage_dev_mvlv_stations = voltage_dev.loc[:, mvlv_stations]
    voltage_issues = voltage_dev_mvlv_stations[
        voltage_dev_mvlv_stations != 0].dropna(how="all").dropna(
            axis=1, how="all")
    stations_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        # create pypsa network with aggregated loads and generators at
        # station's secondary side
        # ToDo Aggregating the LV leads to slightly different voltage results
        #  wherefore checking voltage after running power flow with
        #  non-aggregated LV might show some remaining voltage issues. The
        #  following might therefore need to be changed.
        pypsa_network = edisgo.to_pypsa(
            mode=mode,
            timesteps=time_steps_issues,
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            elia_logger.debug(
                "Number of MV/LV stations with voltage issues: {}".format(
                    len(stations_issues)))
            elia_logger.debug(
                "Number of time steps with voltage issues at "
                "MV/LV stations: {}".format(
                    len(time_steps_issues)))

            # for each station calculate curtailment
            for station in stations_issues:
                # get loads and gens in grid
                gens_grid = pypsa_network.generators[
                    pypsa_network.generators.bus == station].index
                loads_grid = pypsa_network.loads[
                    pypsa_network.loads.bus == station].index

                # get time steps with issues at that station
                ts_issues = voltage_issues.loc[:, station].dropna(
                    how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_grid, loads_grid, ts_issues)

            # run power flow analysis on limited number of time steps
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError("Power flow analysis did not converge for the"
                                 "following time steps: {}.".format(
                    time_steps_issues[~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # get stations with voltage issues
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            voltage_dev_mvlv_stations = voltage_dev.loc[:, mvlv_stations]
            voltage_issues = voltage_dev_mvlv_stations[
                voltage_dev_mvlv_stations != 0].dropna(how="all").dropna(
                axis=1, how="all")
            stations_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        if not mv_grid_is_agg:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(stations_issues) > 0:

                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir, "{}_{}_{}_mvlv_stations_voltage".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve voltage "
                                 "issues at MV/LV stations.")
        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV/LV stations with voltage issues.")
        pass
    return curtailment


def curtailment_mv_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir,
        scenario, strategy, day, mode="mvlv", mv_grid_is_agg=False):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    # get voltage issues in MV
    mv_buses = edisgo.topology.mv_grid.buses_df.index
    voltage_dev_mv = voltage_dev.loc[:, mv_buses]
    voltage_issues = voltage_dev_mv[voltage_dev_mv != 0].dropna(
        how="all").dropna(axis=1, how="all")
    # voltage_issues.to_csv( # TODO
    #     os.path.join(grid_results_dir, "voltage_{}_voltage_issues.csv".format(grid_results_dir.parts[-4]))
    # )
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode=mode, timesteps=time_steps_issues,
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()

            # if iteration_count == 0: # TODO
            #     buses_df_issues.to_csv(
            #         os.path.join(grid_results_dir, "voltage_{}_buses_df_issues.csv".format(grid_results_dir.parts[-4]))
            #     )
            #     pd.DataFrame(feeders).to_csv(
            #         os.path.join(grid_results_dir, "voltage_{}_feeders.csv".format(grid_results_dir.parts[-4]))
            #     )

            elia_logger.debug(
                "Number of MV feeders with voltage issues: {}".format(
                    len(feeders)))
            elia_logger.debug(
                "Number of time steps with voltage issues in MV: {}".format(
                    len(time_steps_issues)))

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.mv_feeder == feeder].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses)].index)

                # if iteration_count == 0: # TODO
                #     gens_feeder_df = edisgo.topology.generators_df[
                #         edisgo.topology.generators_df.bus.isin(buses)]
                #     gens_feeder_df.to_csv(
                #         os.path.join(grid_results_dir, "voltage_{}_gens_feeder_{}_df.csv".format(
                #             grid_results_dir.parts[-4], feeder
                #         ))
                #     )
                #
                #     lines_df = edisgo.topology.lines_df.copy()
                #
                #     lines_df = lines_df[
                #         (lines_df.bus0.isin(buses)) &
                #         (lines_df.bus1.isin(buses)) &
                #         (lines_df.bus0 == feeder) &
                #         (lines_df.bus1 == feeder)
                #     ]
                #
                #     lines_df.to_csv(
                #         os.path.join(grid_results_dir, "voltage_{}_lines_df_{}_df.csv".format(
                #             grid_results_dir.parts[-4], feeder
                #         ))
                #     )

                # get time steps with voltage issues in feeder
                ts_issues = voltage_issues.loc[
                            :, buses].dropna(how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with MV issues
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError(
                    "Power flow analysis did not converge for the "
                    "following time steps: {}.".format(
                        time_steps_issues[
                            ~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # get voltage issues in MV
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            voltage_dev_mv = voltage_dev.loc[:, mv_buses]
            voltage_issues = voltage_dev_mv[
                voltage_dev_mv != 0].dropna(
                how="all").dropna(axis=1, how="all")
            buses_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        if not mv_grid_is_agg:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(time_steps_issues) > 0:
                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir,
                    "{}_{}_{}_mv_voltage".format(scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve MV voltage "
                                 "issues.")

        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)


        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV voltage issues to solve.")
        pass
    return curtailment


def curtailment_lv_lines_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day,
        pypsa_network=None, lv_grid=False, bar=0.98):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    # get overloading issues in LV
    lv_lines = edisgo.topology.lines_df.lv_feeder.dropna().index
    if not rel_load.empty:
        rel_load_lv = rel_load.loc[:, lv_lines]
    else:
        rel_load_lv = rel_load.copy()
    overloading_issues = rel_load_lv[rel_load_lv > bar].dropna(
        how="all").dropna(axis=1, how="all")
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        if pypsa_network is None:
            pypsa_network = edisgo.to_pypsa(
                timesteps=time_steps_issues,
            )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        buses_df = edisgo.topology.buses_df

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with overloading issues
            # get all buses with issues
            buses_issues = edisgo.topology.lines_df.loc[
                           lines_issues, ["bus0", "bus1"]].stack().unique()
            buses_df_issues = buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].dropna().unique()

            elia_logger.debug(
                "Number of LV feeders with overloading issues: {}".format(
                    len(feeders)))
            elia_logger.debug(
                "Number of time steps with overloading issues "
                "in LV: {}".format(
                    len(time_steps_issues)))

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[
                    buses_df_issues.lv_feeder == feeder]
                b = buses_in_feeder.loc[
                         :, "path_length_to_station"].sort_values(
                    ascending=False).index[0]

                # get all generators and loads downstream
                buses_downstream = buses_df[
                    (buses_df.lv_feeder == feeder) &
                    (buses_df.path_length_to_station >=
                     buses_in_feeder.at[b, "path_length_to_station"])].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(
                        buses_downstream)].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(
                        buses_downstream)].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses_downstream)].index)

                # get time steps with overloading issues at that line
                connected_lines = edisgo.topology.get_connected_lines_from_bus(
                    b).index
                rel_load_connected_lines = rel_load.loc[:, connected_lines]
                ts_issues = rel_load_connected_lines[
                    rel_load_connected_lines > bar].dropna(
                    how="all").dropna(axis=1, how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, curtailment_step=0.1) # TODO

            # run power flow analysis on all time steps with MV issues
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError(
                    "Power flow analysis did not converge for the "
                    "following time steps: {}.".format(
                        time_steps_issues[
                            ~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading issues in LV
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_lv = rel_load.loc[:, lv_lines]
            overloading_issues = rel_load_lv[rel_load_lv > bar].dropna(
                how="all").dropna(axis=1, how="all")
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if not lv_grid:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(time_steps_issues) > 0:

                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir, "{}_{}_{}_lv_overloading".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve overloading "
                                 "issues in LV.")
        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")
            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc[
            "lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV overloading issues to solve.")
        pass
    if lv_grid:
        return pypsa_network, curtailment
    else:
        return curtailment


def curtailment_mvlv_stations_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mv_grid_id, mode="mvlv",
        mv_grid_is_agg=False):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    # get overloading issues at MV/LV stations
    mvlv_stations = [_ for _ in rel_load.columns if "mvlv_station" in _]
    rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
    overloading_issues = rel_load_mvlv_stations[
        rel_load_mvlv_stations > 0.99].dropna(how="all").dropna(
        axis=1, how="all")
    stations_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    stations_secondary_sides = {
        _: "BusBar_mvgd_{}_lvgd_{}_LV".format(mv_grid_id, _.split("_")[-1])
        for _ in mvlv_stations}

    if len(time_steps_issues) > 0:
        # create pypsa network with aggregated loads and generators at
        # station's secondary side
        pypsa_network = edisgo.to_pypsa(
            mode=mode,
            timesteps=time_steps_issues,
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            elia_logger.debug(
                "Number of MV/LV stations with overloading issues: {}".format(
                    len(stations_issues)))
            elia_logger.debug(
                "Number of time steps with overloading issues at "
                "MV/LV stations: {}".format(
                    len(time_steps_issues)))

            # for each station calculate curtailment
            for station in stations_issues:
                # get loads and gens in grid
                gens_grid = pypsa_network.generators[
                    pypsa_network.generators.bus ==
                    stations_secondary_sides[station]].index
                loads_grid = pypsa_network.loads[
                    pypsa_network.loads.bus ==
                    stations_secondary_sides[station]].index

                # get time steps with issues at that station
                ts_issues = overloading_issues.loc[:, station].dropna(
                    how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_grid, loads_grid, ts_issues)

            # run power flow analysis on limited number of time steps
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError("Power flow analysis did not converge for the"
                                 "following time steps: {}.".format(
                    time_steps_issues[~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading and voltage issues at stations
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
            overloading_issues = rel_load_mvlv_stations[
                rel_load_mvlv_stations > 0.99].dropna(how="all").dropna(
                axis=1, how="all")
            stations_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if not mv_grid_is_agg:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(time_steps_issues) > 0:

                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir, "{}_{}_{}_mvlv_stations_overloading".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve overloading "
                                 "issues at MV/LV stations.")
        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")
            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV/LV stations with overloading issues.")
        pass
    return curtailment


def curtailment_mv_lines_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mode="mvlv", mv_grid_is_agg=False):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    mv_lines = edisgo.topology.mv_grid.lines_df.index
    rel_load_mv = rel_load.loc[:, mv_lines]
    overloading_issues = rel_load_mv[rel_load_mv > 0.99].dropna(
        how="all").dropna(axis=1, how="all")
    # overloading_issues.to_csv(  # TODO
    #     os.path.join(grid_results_dir, "overloading_{}_issues.csv".format(grid_results_dir.parts[-4]))
    # )
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode=mode, timesteps=time_steps_issues,
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        buses_df = edisgo.topology.buses_df

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with overloading issues
            # get all buses with issues
            buses_issues = edisgo.topology.lines_df.loc[
                lines_issues, ["bus0", "bus1"]].stack().unique()
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].dropna().unique()

            # if iteration_count == 0: # TODO
            #     buses_df_issues.to_csv(
            #         os.path.join(grid_results_dir, "overloading_{}_buses_df_issues.csv".format(grid_results_dir.parts[-4]))
            #     )
            #     pd.DataFrame(feeders).to_csv(
            #         os.path.join(grid_results_dir, "overloading_{}_feeders.csv".format(grid_results_dir.parts[-4]))
            #     )

            elia_logger.debug(
                "Number of MV feeders with overloading issues: {}".format(
                    len(feeders)))
            elia_logger.debug(
                "Number of time steps with overloading issues "
                "in LV: {}".format(
                    len(time_steps_issues)))

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[
                    buses_df_issues.mv_feeder == feeder]
                b = buses_in_feeder.loc[
                         :, "path_length_to_station"].sort_values(
                    ascending=False).index[0]

                # get all generators and loads downstream
                buses_downstream = buses_df[
                    (buses_df.mv_feeder == feeder) &
                    (buses_df.path_length_to_station >=
                     buses_in_feeder.at[b, "path_length_to_station"])].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(
                        buses_downstream)].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses_downstream)].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses_downstream)].index)

                # get time steps with overloading issues at that line
                connected_lines = edisgo.topology.get_connected_lines_from_bus(
                    b).index
                rel_load_connected_lines = rel_load.loc[:, connected_lines]
                ts_issues = rel_load_connected_lines[
                    rel_load_connected_lines > 0.99].dropna(
                    how="all").dropna(axis=1, how="all").index

                # if iteration_count == 0: # TODO
                #     gens_feeder_df = edisgo.topology.generators_df[
                #         edisgo.topology.generators_df.bus.isin(
                #             buses_downstream
                #         )
                #     ]
                #     gens_feeder_df.to_csv(
                #         os.path.join(grid_results_dir, "overloading_{}_gens_feeder_{}_df.csv".format(
                #             grid_results_dir.parts[-4], feeder
                #         ))
                #     )
                #
                #     connected_lines = edisgo.topology.get_connected_lines_from_bus(b)
                #
                #     connected_lines.to_csv(
                #         os.path.join(grid_results_dir, "overloading_{}_lines_df_{}_df.csv".format(
                #             grid_results_dir.parts[-4], feeder
                #         ))
                #     )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with MV issues
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError(
                    "Power flow analysis did not converge for the "
                    "following time steps: {}.".format(
                        time_steps_issues[
                            ~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading issues in LV
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_mv = rel_load.loc[:, mv_lines]
            overloading_issues = rel_load_mv[rel_load_mv > 0.99].dropna(
                how="all").dropna(axis=1, how="all")
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        if not mv_grid_is_agg:
            # rerun power flow to update power flow results
            edisgo.analyze()

            if len(time_steps_issues) > 0:

                _save_results_when_curtailment_failed(
                    edisgo, grid_results_dir, "{}_{}_{}_mv_overloading".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")))

                raise ValueError("Curtailment not sufficient to solve grid "
                                 "issues in MV.")
        else:
            pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")
            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV overloading issues to solve.")
        pass
    return curtailment


def curtailment_hvmv_station_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day):

    elia_logger = logging.getLogger(
        'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))

    hvmv_station = "hvmv_station_{}".format(edisgo.topology.mv_grid)
    rel_load_hvmv_station = rel_load.loc[:, hvmv_station]
    overloading_issues = rel_load_hvmv_station[
        rel_load_hvmv_station > 0.99].dropna(how="all")
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            gens = edisgo.topology.generators_df.index
            loads = edisgo.topology.loads_df.index

            # reduce active and reactive power of loads or generators
            # (depending on whether it is a load or feed-in case)
            pypsa_network = _curtail(
                pypsa_network, gens, loads, time_steps_issues)

            # run power flow analysis on all time steps with overloading issues
            if iteration_count == 0:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues)
            else:
                pf_results, pypsa_network = my_pf(pypsa_network, time_steps_issues, mode="use_seed")

            if all(pf_results["converged"]["0"].tolist()):
                pypsa_io.process_pfa_results(edisgo, pypsa_network,
                                             time_steps_issues)
            else:
                raise ValueError(
                    "Power flow analysis did not converge for the "
                    "following time steps: {}.".format(
                        time_steps_issues[
                            ~pf_results["converged"]["0"]].tolist())
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading issues
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_hvmv_station = rel_load.loc[:, hvmv_station]
            overloading_issues = rel_load_hvmv_station[
                rel_load_hvmv_station > 0.99].dropna(how="all")
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "hvmv_station_overloading")

            raise ValueError("Curtailment not sufficient to solve grid "
                             "issues at HV/MV station.")

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.at[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No overloading issues at HV/MV station to solve.")
    return curtailment


def curtail_lv_grids(
        edisgo,
        grid_results_dir,
        day,
        scenario,
        strategy,
        curtailment,
):
    try:
        elia_logger = logging.getLogger(
            'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))
        elia_logger.setLevel(logging.DEBUG)

        lv_grids = list(edisgo.topology._grids.keys())

        lv_grids = [
            lv_grid for lv_grid in lv_grids if "LVGrid" in lv_grid
        ]

        pypsa_network = edisgo.to_pypsa()

        pypsa_network.lpf(edisgo.timeseries.timeindex)

        arr_bus0 = pypsa_network.buses_t.v_ang.loc[:, pypsa_network.lines.bus0].values
        arr_bus1 = pypsa_network.buses_t.v_ang.loc[:, pypsa_network.lines.bus1].values

        arr_angle_diff = np.absolute(arr_bus0 - arr_bus1).max(axis=0)

        s_angle_diff = pd.Series(arr_angle_diff, index=pypsa_network.lines.index) * 180 / np.pi

        bar = 3 # TODO

        s_angle_diff = s_angle_diff[s_angle_diff.ge(bar)].filter(like="_lvgd_")

        lv_grid_list_overloading = [
            bus.split("_lvgd_")[-1].split("_")[0] for bus in s_angle_diff.index.tolist()
        ]

        lv_grid_list_overloading = list(set(lv_grid_list_overloading))

        lv_grids = [
            lv_grid for lv_grid in lv_grids if any(sub in lv_grid for sub in lv_grid_list_overloading)
        ]

        for count_grids, lv_grid in enumerate(lv_grids):

            t1 = perf_counter()

            pypsa_lv = edisgo.to_pypsa(
                mode="lv",
                lv_grid_name=lv_grid,
            )

            pypsa_lv_orig = pypsa_lv.copy()

            i = 0

            converged = False

            while i < max_iterations and not converged:
                try:
                    pf_results, pypsa_lv = my_pf(pypsa_lv, edisgo.timeseries.timeindex)

                    converged = True

                except:
                    if i == 0:
                        print(
                            "First PF didn't converge for day {} in {}.".format(
                                day, lv_grid,
                            )
                        )

                    timeindex = edisgo.timeseries.residual_load.nsmallest(
                        int(len(edisgo.timeseries.residual_load) / 20),
                        keep="all",
                    ).index.tolist()

                    _curtail(
                        pypsa_lv, pypsa_lv.generators.index, pypsa_lv.loads.index, timeindex,
                        curtailment_step=0.2,
                    )

                    _overwrite_edisgo_timeseries(edisgo, pypsa_lv)

                    i += 1

            print("It took {} seconds for the initial power flow analysis on day {} in {}.".format(
                round(perf_counter() - t1, 0), day, lv_grid
            ))

            i = 0

            t1 = perf_counter()

            while i < max_iterations and all(pf_results["converged"]["0"].tolist()) is False:
                elia_logger.debug(
                    "Number of time steps with convergence issues: {}".format(
                        len(edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist())
                    )
                )

                _curtail(
                    pypsa_lv, pypsa_lv.generators.index, pypsa_lv.loads.index,
                    edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
                )

                curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                    pypsa_lv_orig, pypsa_lv)
                elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

                j = 0

                converged = False

                while j < max_iterations and not converged:
                    try:
                        pf_results, pypsa_lv = my_pf(pypsa_lv, edisgo.timeseries.timeindex)

                        converged = True

                    except:
                        if i == 0 and j == 0:
                            print(
                                "PF Nr. {} didn't converge for day {} in {}.".format(
                                    i + 2, day, lv_grid,
                                )
                            )

                        _curtail(
                            pypsa_lv, pypsa_lv.generators.index, pypsa_lv.loads.index,
                            edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
                        )

                        _overwrite_edisgo_timeseries(edisgo, pypsa_lv)

                        j += 1

                i += 1

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(pypsa_lv_orig, pypsa_lv)

            curtailment.loc[
                "lv_convergence_problems", "feed-in"] += curtailed_feedin.sum().sum()
            curtailment.loc[
                "lv_convergence_problems", "load"] += curtailed_load.sum().sum()

            print("It took {} seconds to overcome the initial convergence problems in {}.".format(
                round(perf_counter() - t1, 0), lv_grid
            ))

            pypsa_io.process_pfa_results(edisgo, pypsa_lv, edisgo.timeseries.timeindex)

            _overwrite_edisgo_timeseries(edisgo, pypsa_lv)

            voltage_dev = results_helper_functions.voltage_diff(edisgo)

            pypsa_lv, curtailment = curtailment_lv_voltage(
                edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day,
                pypsa_network=pypsa_lv, lv_grid=True
            )

            # ToDo Only recalculate voltage deviation if curtailment was conducted
            #  (will be done when voltage deviation is attribute in results object)
            _ = results_helper_functions.voltage_diff(edisgo)

            # curtailment due to overloading issues
            # recalculate relative line loading
            rel_load = results_helper_functions.relative_load(edisgo)

            pypsa_lv, curtailment = curtailment_lv_lines_overloading(
                edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day,
                pypsa_network=pypsa_lv, lv_grid=True
            )

            bar = 1 + 0.1 # 1e-3 # TODO

            lv_grid_matching = lv_grid.lower()

            lv_grid_matching = lv_grid_matching[:2] + "_" + lv_grid_matching[2:]

            mvlv_transformer_rating = edisgo.topology.transformers_df[
                edisgo.topology.transformers_df.index.str.contains(lv_grid_matching)
            ].s_nom.sum()

            transformer_loading_mw = pypsa_lv.loads_t["p_set"].sum(axis=1) - pypsa_lv.generators_t["p_set"].sum(axis=1)

            transformer_loading_mvar = pypsa_lv.loads_t["q_set"].sum(axis=1) - pypsa_lv.generators_t["q_set"].sum(
                axis=1)

            transformer_loading_mva = np.sqrt(transformer_loading_mw**2 + transformer_loading_mvar**2)

            transformer_overloading = transformer_loading_mva[
                transformer_loading_mva.ge(mvlv_transformer_rating * bar)
            ]

            i = 0

            while not transformer_overloading.empty and i < max_iterations:
                elia_logger.debug(
                    "Number of time steps with overloading issues: {}".format(
                        len(transformer_overloading)
                    )
                )

                _curtail(
                    pypsa_lv, pypsa_lv.generators.index, pypsa_lv.loads.index,
                    transformer_overloading.index.tolist(),
                )

                curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                    pypsa_lv_orig, pypsa_lv)
                elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

                transformer_loading_mw = pypsa_lv.loads_t["p_set"].sum(axis=1) - pypsa_lv.generators_t["p_set"].sum(
                    axis=1)

                transformer_loading_mvar = pypsa_lv.loads_t["q_set"].sum(axis=1) - pypsa_lv.generators_t["q_set"].sum(
                    axis=1)

                transformer_loading_mva = np.sqrt(transformer_loading_mw ** 2 + transformer_loading_mvar ** 2)

                transformer_overloading = transformer_loading_mva[
                    transformer_loading_mva.ge(mvlv_transformer_rating * bar)]

                i += 1

            if i == 0:
                elia_logger.debug("No MVLV overloading issues to solve.")

            _overwrite_edisgo_timeseries(edisgo, pypsa_lv)

            print(count_grids+1, r"/", len(lv_grids))

        return edisgo, curtailment

    except:
        traceback.print_exc()


def curtail_mv_grid(
        edisgo,
        grid_results_dir,
        day,
        scenario,
        strategy,
        curtailment,
):
    try:
        elia_logger = logging.getLogger(
            'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))
        elia_logger.setLevel(logging.DEBUG)

        t1 = perf_counter()

        pypsa_mv = edisgo.to_pypsa(
            mode="mv",
        )

        pypsa_mv_orig = pypsa_mv.copy()

        i = 0

        converged = False

        while i < max_iterations and not converged:
            try:
                pf_results, pypsa_mv = my_pf(pypsa_mv, edisgo.timeseries.timeindex)

                converged = True

            except:
                if i == 0:
                    print(
                        "First PF didn't converge for day {} in MV Grid.".format(
                            day,
                        )
                    )

                timeindex = edisgo.timeseries.residual_load.nsmallest(
                    int(len(edisgo.timeseries.residual_load) / 20),
                    keep="all",
                ).index.tolist()

                _curtail(
                    pypsa_mv, pypsa_mv.generators.index, pypsa_mv.loads.index, timeindex,
                    curtailment_step=0.2,
                )

                _overwrite_edisgo_timeseries(edisgo, pypsa_mv)

                i += 1

        print("It took {} seconds for the initial power flow analysis on day {} in MV Grid.".format(
            round(perf_counter() - t1, 0), day,
        ))

        i = 0

        t1 = perf_counter()

        while i < max_iterations and all(pf_results["converged"]["0"].tolist()) is False:
            elia_logger.debug(
                "Number of time steps with convergence issues: {}".format(
                    len(edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist())
                )
            )

            _curtail(
                pypsa_mv, pypsa_mv.generators.index, pypsa_mv.loads.index,
                edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
            )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_mv_orig, pypsa_mv)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            j = 0

            converged = False

            while j < max_iterations and not converged:
                try:
                    pf_results, pypsa_mv = my_pf(pypsa_mv, edisgo.timeseries.timeindex)

                    converged = True

                except:
                    if i == 0 and j == 0:
                        print(
                            "PF Nr. {} didn't converge for day {} in MV Grid.".format(
                                i + 2, day,
                            )
                        )

                    _curtail(
                        pypsa_mv, pypsa_mv.generators.index, pypsa_mv.loads.index,
                        edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
                    )

                    _overwrite_edisgo_timeseries(edisgo, pypsa_mv)

                    j += 1

            i += 1

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(pypsa_mv_orig, pypsa_mv)

        curtailment.loc[
            "mv_convergence_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc[
            "mv_convergence_problems", "load"] += curtailed_load.sum().sum()

        print("It took {} seconds to overcome the initial convergence problems in MV Grid.".format(
            round(perf_counter() - t1, 0)
        ))

        pypsa_io.process_pfa_results(edisgo, pypsa_mv, edisgo.timeseries.timeindex)

        _overwrite_edisgo_timeseries(edisgo, pypsa_mv)

        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day, mode="mv", mv_grid_is_agg=True)

        # ToDo Only recalculate voltage deviation if curtailment was conducted
        #  (will be done when voltage deviation is attribute in results object)
        _ = results_helper_functions.voltage_diff(edisgo)

        # curtailment due to overloading issues

        # recalculate relative line loading
        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mode="mv", mv_grid_is_agg=True)

        return edisgo, curtailment

    except:
        traceback.print_exc()


def curtail_mvlv_grid(
        edisgo,
        grid_results_dir,
        day,
        scenario,
        strategy,
        curtailment,
        mv_grid_id,
):
    try:
        elia_logger = logging.getLogger(
            'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))
        elia_logger.setLevel(logging.DEBUG)

        t1 = perf_counter()

        pypsa_mvlv = edisgo.to_pypsa(
            mode="mvlv",
        )

        pypsa_mvlv_orig = pypsa_mvlv.copy()

        i = 0

        converged = False

        while i < max_iterations and not converged:
            try:
                pf_results, pypsa_mvlv = my_pf(pypsa_mvlv, edisgo.timeseries.timeindex)

                converged = True

            except:
                if i == 0:
                    print(
                        "First PF didn't converge for day {} in MVLV Grid.".format(
                            day,
                        )
                    )

                timeindex = edisgo.timeseries.residual_load.nsmallest(
                    int(len(edisgo.timeseries.residual_load) / 20),
                    keep="all",
                ).index.tolist()

                _curtail(
                    pypsa_mvlv, pypsa_mvlv.generators.index, pypsa_mvlv.loads.index, timeindex,
                    curtailment_step=0.2,
                )

                _overwrite_edisgo_timeseries(edisgo, pypsa_mvlv)

                i += 1

        print("It took {} seconds for the initial power flow analysis on day {} in MVLV Grid.".format(
            round(perf_counter() - t1, 0), day,
        ))

        i = 0

        t1 = perf_counter()

        while i < max_iterations and all(pf_results["converged"]["0"].tolist()) is False:
            elia_logger.debug(
                "Number of time steps with convergence issues: {}".format(
                    len(edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist())
                )
            )

            _curtail(
                pypsa_mvlv, pypsa_mvlv.generators.index, pypsa_mvlv.loads.index,
                edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
            )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_mvlv_orig, pypsa_mvlv)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            j = 0

            converged = False

            while j < max_iterations and not converged:
                try:
                    pf_results, pypsa_mvlv = my_pf(pypsa_mvlv, edisgo.timeseries.timeindex)

                    converged = True

                except:
                    if i == 0 and j == 0:
                        print(
                            "PF Nr. {} didn't converge for day {} in MVLV Grid.".format(
                                i + 2, day,
                            )
                        )

                    _curtail(
                        pypsa_mvlv, pypsa_mvlv.generators.index, pypsa_mvlv.loads.index,
                        edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
                    )

                    _overwrite_edisgo_timeseries(edisgo, pypsa_mvlv)

                    j += 1

            i += 1

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(pypsa_mvlv_orig, pypsa_mvlv)

        curtailment.loc[
            "mv_convergence_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc[
            "mv_convergence_problems", "load"] += curtailed_load.sum().sum()

        print("It took {} seconds to overcome the initial convergence problems in MVLV Grid.".format(
            round(perf_counter() - t1, 0)
        ))

        pypsa_io.process_pfa_results(edisgo, pypsa_mvlv, edisgo.timeseries.timeindex)

        _overwrite_edisgo_timeseries(edisgo, pypsa_mvlv)

        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mvlv_stations_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day, mv_grid_is_agg=True)

        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day, mv_grid_is_agg=True)

        # ToDo Only recalculate voltage deviation if curtailment was conducted
        #  (will be done when voltage deviation is attribute in results object)
        _ = results_helper_functions.voltage_diff(edisgo)

        # curtailment due to overloading issues

        # recalculate relative line loading
        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mvlv_stations_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mv_grid_id, mv_grid_is_agg=True)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mv_grid_is_agg=True)

        return edisgo, curtailment

    except:
        traceback.print_exc()


def calculate_curtailment(
        grid_dir,
        edisgo,
        strategy,
        day,
):
    try:
        if day is None:
            day = "full"

        mv_grid_id = int(grid_dir.parts[-2])
        scenario = grid_dir.parts[-3]

        elia_logger = logging.getLogger(
            'MA: {} {} {}'.format(scenario, edisgo.topology.id, strategy))
        elia_logger.setLevel(logging.DEBUG)

        grid_results_dir = Path(os.path.join( # TODO
            grid_dir,
            "weekly_curtailment_v2",
        ))

        os.makedirs(
            grid_results_dir,
            exist_ok=True,
        )

        edisgo.timeseries.residual_load.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_residual_load_before_curtailment.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )

        # save original time series before curtailment
        feedin_ts = edisgo.timeseries.generators_active_power.copy()
        load_ts = edisgo.timeseries.loads_active_power.copy()
        charging_ts = edisgo.timeseries.charging_points_active_power.copy()

        # assign feeders and path length to station
        assign_feeder(edisgo, mode="mv_feeder")
        assign_feeder(edisgo, mode="lv_feeder")
        get_path_length_to_station(edisgo)

        curtailment = pd.DataFrame(
            data=0,
            columns=["feed-in", "load"],
            index=[
                "lv_problems",
                "mv_problems",
                "lv_convergence_problems",
                "mv_convergence_problems",
            ]
        )

        # print("CPs:", edisgo.timeseries.charging_points_active_power.sum().sum())
        # print("Loads:", edisgo.timeseries.loads_active_power.sum().sum())
        #
        # for col in edisgo.timeseries._charging_points_active_power.columns:
        #     edisgo.timeseries._charging_points_active_power[col].values[:] = 0
        # for col in edisgo.timeseries.charging_points_active_power.columns:
        #     edisgo.timeseries.charging_points_active_power[col].values[:] = 0

        print("CPs:", edisgo.timeseries.charging_points_active_power.sum().sum())
        print("Loads:", edisgo.timeseries.loads_active_power.sum().sum())
        print("Gens:", edisgo.timeseries.generators_active_power.sum().sum())

        t0 = perf_counter()

        edisgo, curtailment = curtail_lv_grids(
            edisgo,
            grid_results_dir,
            day,
            scenario,
            strategy,
            curtailment,
        )

        print(
            "It took {} seconds to calculate all lv grids.".format(perf_counter()-t0)
        )

        # t0 = perf_counter()
        #
        # edisgo, curtailment = curtail_mv_grid(
        #     edisgo,
        #     grid_results_dir,
        #     day,
        #     scenario,
        #     strategy,
        #     curtailment,
        # )
        #
        # print(
        #     "It took {} seconds to calculate the mv grid.".format(perf_counter() - t0)
        # )
        #
        # t0 = perf_counter()
        #
        # edisgo, curtailment = curtail_mvlv_grid(
        #     edisgo,
        #     grid_results_dir,
        #     day,
        #     scenario,
        #     strategy,
        #     curtailment,
        #     mv_grid_id,
        # )
        #
        # print(
        #     "It took {} seconds to calculate the mvlv grid.".format(perf_counter() - t0)
        # )

        t1 = perf_counter()

        pypsa_network = edisgo.to_pypsa()

        pypsa_network_orig = pypsa_network.copy()

        i = 0

        converged = False

        while i < max_iterations and not converged:
            try:
                pf_results, pypsa_network = my_pf(pypsa_network, edisgo.timeseries.timeindex)

                converged = True

            except:
                if i == 0:
                    elia_logger.debug(
                        "First PF didn't converge for day {} with strategy {} in grid {} and scenario {}".format(
                            day, strategy, mv_grid_id, scenario
                        ),
                        file=open(
                            "convergence_failed.txt",
                            "a",
                        )
                    )

                if edisgo.timeseries.residual_load.max() > abs(edisgo.timeseries.residual_load.min()):
                    timeindex = edisgo.timeseries.residual_load.nlargest(
                        int(len(edisgo.timeseries.residual_load) / 10),
                        keep="all",
                    ).index.tolist()
                else:
                    timeindex = edisgo.timeseries.residual_load.nsmallest(
                        int(len(edisgo.timeseries.residual_load) / 10),
                        keep="all",
                    ).index.tolist()

                _curtail(
                    pypsa_network, pypsa_network.generators.index, pypsa_network.loads.index, timeindex,
                    curtailment_step=0.05,
                )

                curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                    pypsa_network_orig, pypsa_network)
                elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

                _overwrite_edisgo_timeseries(edisgo, pypsa_network)

                i += 1

        print("It took {} seconds for the initial power flow analysis on day {}.".format(
            round(perf_counter() - t1, 0), day
        ))

        i = 0

        t1 = perf_counter()

        while i < max_iterations and all(pf_results["converged"]["0"].tolist()) is False:
            elia_logger.debug(
                "Number of time steps with convergence issues: {}".format(
                    len(edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist())
                )
            )

            _curtail(
                pypsa_network, pypsa_network.generators.index, pypsa_network.loads.index,
                edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
                curtailment_step=0.05,
            )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            j = 0

            converged = False

            while j < max_iterations and not converged:
                try:
                    pf_results, pypsa_network = my_pf(pypsa_network, edisgo.timeseries.timeindex)

                    converged = True

                except:
                    if i == 0 and j == 0:
                        elia_logger.debug(
                            "PF Nr. {} didn't converge for day {} with strategy {}".format(
                                i+2, day, strategy
                            )
                        )

                    if edisgo.timeseries.residual_load.max() > abs(edisgo.timeseries.residual_load.min()):
                        timeindex = edisgo.timeseries.residual_load.nlargest(
                            int(len(edisgo.timeseries.residual_load) / 10),
                            keep="all",
                        ).index.tolist()
                    else:
                        timeindex = edisgo.timeseries.residual_load.nsmallest(
                            int(len(edisgo.timeseries.residual_load) / 10),
                            keep="all",
                        ).index.tolist()

                    _curtail(
                        pypsa_network, pypsa_network.generators.index, pypsa_network.loads.index, timeindex,
                        # curtailment_step=0.05,
                    )

                    _overwrite_edisgo_timeseries(edisgo, pypsa_network)

                    j += 1

            i += 1

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(pypsa_network_orig, pypsa_network)

        curtailment.loc[
            "mv_convergence_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc[
            "mv_convergence_problems", "load"] += curtailed_load.sum().sum()

        print("It took {} seconds to overcome the initial convergence problems.".format(
            round(perf_counter() - t1, 0)
        ))

        pypsa_io.process_pfa_results(edisgo, pypsa_network, edisgo.timeseries.timeindex)

        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        # curtailment due to voltage issues

        # import voltage deviation
        # path = os.path.join(reload_dir, 'voltage_deviation.csv')
        # voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_lv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day)

        # ToDo Only recalculate voltage deviation if curtailment was conducted
        #  (will be done when voltage deviation is attribute in results object)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mvlv_stations_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day)

        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, day)

        # curtailment due to overloading issues

        # recalculate relative line loading
        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_lv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mvlv_stations_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day, mv_grid_id)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment_hvmv_station_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, day)

        # check if everything was solved
        voltage_dev = results_helper_functions.voltage_diff(edisgo)
        issues = voltage_dev[
            abs(voltage_dev) > 1e-2].dropna( # TODO
            how="all").dropna(axis=1, how="all")
        if not issues.empty:
            print("Not all voltage issues solved on day {} of Grid {} with strategy {}.".format(
                day, mv_grid_id, strategy
            ))
            issues.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_{}_voltage_issues.csv".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")
                    ),
                )
            )
        else:
            # print("Success. All voltage issues solved on day {} of Grid {} with strategy {}.".format(
            #     day, mv_grid_id, strategy
            # ))
            pass
        rel_load = results_helper_functions.relative_load(edisgo)
        issues = rel_load[
            rel_load > 1+1e-2].dropna( # TODO
            how="all").dropna(axis=1, how="all")
        if not issues.empty:
            print("Not all overloading issues solved on day {} of Grid {} with strategy {}.".format(
                day, mv_grid_id, strategy
            ))
            issues.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_{}_overloading_issues.csv".format(
                        scenario, strategy, day.strftime("%Y-%m-%d")
                    ),
                )
            )
        else:
            # print("Success. All overloading issues solved on day {} of Grid {} with strategy {}.".format(
            #     day, mv_grid_id, strategy
            # ))
            pass

        # save curtailment sums
        curtailment.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d"))))

        # save time series
        curtailed_feedin = feedin_ts - edisgo.timeseries.generators_active_power
        curtailed_load = pd.concat(
            [(load_ts - edisgo.timeseries.loads_active_power),
             (charging_ts - edisgo.timeseries.charging_points_active_power)],
            axis=1)
        curtailed_feedin.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_per_gen.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )
        curtailed_load.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_per_load.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )
        curtailed_feedin.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_feedin.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )
        curtailed_load.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_demand.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )

        edisgo.timeseries.residual_load.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_residual_load_after_curtailment.csv".format(
                scenario, strategy, day.strftime("%Y-%m-%d")))
        )

    except Exception as e:
        mv_grid_id = int(grid_dir.parts[-2])
        scenario = grid_dir.parts[-3]
        print("Error in {} on day {} MV grid {}.".format(scenario, day, mv_grid_id))
        traceback.print_exc()


# if __name__ == "__main__":
#     if num_threads == 1:
#         for mv_grid_id in mv_grid_ids:
#             calculate_curtailment(mv_grid_id)
#     else:
#         with multiprocessing.Pool(num_threads) as pool:
#             pool.map(calculate_curtailment, mv_grid_ids)
