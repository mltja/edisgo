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
curtailment_step = 0.1 # 0.2 # TODO
max_iterations = 50


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
    loads = edisgo.topology.loads_df.index
    edisgo.timeseries._loads_active_power.loc[
        time_steps, loads] = pypsa_network.loads_t.p_set.loc[
        time_steps, loads]
    edisgo.timeseries._loads_reactive_power.loc[
        time_steps, loads] = pypsa_network.loads_t.q_set.loc[
        time_steps, loads]

    if not edisgo.topology.charging_points_df.empty:
        charging_points = edisgo.topology.charging_points_df.index
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


def _curtail(pypsa_network, gens, loads, time_steps):

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

    # curtail loads or generators by specified curtailment factor
    # active power
    pypsa_network.generators_t.p_set.loc[ts_curtail_gens, gens] = (
        gens_ts.loc[ts_curtail_gens, :] -
        curtailment_step *
        gens_ts.loc[ts_curtail_gens, :])
    pypsa_network.loads_t.p_set.loc[ts_curtail_loads, loads] = (
        loads_ts.loc[ts_curtail_loads, :] -
        curtailment_step *
        loads_ts.loc[ts_curtail_loads, :])
    # reactive power
    tmp = pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens]
    pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens] = (
        tmp - curtailment_step * tmp)
    tmp = pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads]
    pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads] = (
        tmp - curtailment_step * tmp)

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


def curtailment_lv_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # get voltage issues in LV
    lv_buses = edisgo.topology.buses_df.lv_feeder.dropna().index
    voltage_dev_lv = voltage_dev.loc[:, lv_buses]
    voltage_issues = voltage_dev_lv[
        voltage_dev_lv != 0].dropna(how="all").dropna(
            axis=1, how="all")
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].unique()

            # elia_logger.debug(
            #     "Number of LV feeders with voltage issues: {}".format(
            #         len(feeders)))
            # elia_logger.debug(
            #     "Number of time steps with voltage issues in LV: {}".format(
            #         len(time_steps_issues)))

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
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

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
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_lv_voltage".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve LV voltage "
                             "issues.")

        # calculate curtailment
        # ToDo: Why does .at not work?
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No LV voltage issues to solve.")
        pass
    return curtailment


def curtailment_mvlv_stations_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

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
            mode="mvlv",
            timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            # elia_logger.debug(
            #     "Number of MV/LV stations with voltage issues: {}".format(
            #         len(stations_issues)))
            # elia_logger.debug(
            #     "Number of time steps with voltage issues at "
            #     "MV/LV stations: {}".format(
            #         len(time_steps_issues)))

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
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

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
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(stations_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_mvlv_stations_voltage".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve voltage "
                             "issues at MV/LV stations.")

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No MV/LV stations with voltage issues.")
        pass
    return curtailment


def curtailment_mv_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # get voltage issues in MV
    mv_buses = edisgo.topology.mv_grid.buses_df.index
    voltage_dev_mv = voltage_dev.loc[:, mv_buses]
    voltage_issues = voltage_dev_mv[voltage_dev_mv != 0].dropna(
        how="all").dropna(axis=1, how="all")
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()

            # elia_logger.debug(
            #     "Number of MV feeders with voltage issues: {}".format(
            #         len(feeders)))
            # elia_logger.debug(
            #     "Number of time steps with voltage issues in MV: {}".format(
            #         len(time_steps_issues)))

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

                # get time steps with voltage issues in feeder
                ts_issues = voltage_issues.loc[
                            :, buses].dropna(how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

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
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_mv_voltage".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve MV voltage "
                             "issues.")

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No MV voltage issues to solve.")
        pass
    return curtailment


def curtailment_lv_lines_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # get overloading issues in LV
    lv_lines = edisgo.topology.lines_df.lv_feeder.dropna().index
    rel_load_lv = rel_load.loc[:, lv_lines]
    overloading_issues = rel_load_lv[rel_load_lv > 1].dropna(
        how="all").dropna(axis=1, how="all")
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(timesteps=time_steps_issues)

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

            # elia_logger.debug(
            #     "Number of LV feeders with overloading issues: {}".format(
            #         len(feeders)))
            # elia_logger.debug(
            #     "Number of time steps with overloading issues "
            #     "in LV: {}".format(
            #         len(time_steps_issues)))

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
                    rel_load_connected_lines > 1].dropna(
                    how="all").dropna(axis=1, how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading issues in LV
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_lv = rel_load.loc[:, lv_lines]
            overloading_issues = rel_load_lv[rel_load_lv > 1].dropna(
                how="all").dropna(axis=1, how="all")
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_lv_overloading".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve overloading "
                             "issues in LV.")

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No LV overloading issues to solve.")
        pass
    return curtailment


def curtailment_mvlv_stations_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk, mv_grid_id):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # get overloading issues at MV/LV stations
    mvlv_stations = [_ for _ in rel_load.columns if "mvlv_station" in _]
    rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
    overloading_issues = rel_load_mvlv_stations[
        rel_load_mvlv_stations > 1].dropna(how="all").dropna(
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
            mode="mvlv",
            timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            # elia_logger.debug(
            #     "Number of MV/LV stations with overloading issues: {}".format(
            #         len(stations_issues)))
            # elia_logger.debug(
            #     "Number of time steps with overloading issues at "
            #     "MV/LV stations: {}".format(
            #         len(time_steps_issues)))

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
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading and voltage issues at stations
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
            overloading_issues = rel_load_mvlv_stations[
                rel_load_mvlv_stations > 1].dropna(how="all").dropna(
                axis=1, how="all")
            stations_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_mvlv_stations_overloading".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve overloading "
                             "issues at MV/LV stations.")

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No MV/LV stations with overloading issues.")
        pass
    return curtailment


def curtailment_mv_lines_overloading(
        edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk):

    elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    mv_lines = edisgo.topology.mv_grid.lines_df.index
    rel_load_mv = rel_load.loc[:, mv_lines]
    overloading_issues = rel_load_mv[rel_load_mv > 1].dropna(
        how="all").dropna(axis=1, how="all")
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues)

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

            # elia_logger.debug(
            #     "Number of MV feeders with overloading issues: {}".format(
            #         len(feeders)))
            # elia_logger.debug(
            #     "Number of time steps with overloading issues "
            #     "in LV: {}".format(
            #         len(time_steps_issues)))

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
                    rel_load_connected_lines > 1].dropna(
                    how="all").dropna(axis=1, how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues)

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues)
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
            # elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            #     curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # recheck for overloading issues in LV
            rel_load = results_helper_functions.relative_load(edisgo)
            rel_load_mv = rel_load.loc[:, mv_lines]
            overloading_issues = rel_load_mv[rel_load_mv > 1].dropna(
                how="all").dropna(axis=1, how="all")
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)
        # rerun power flow to update power flow results
        edisgo.analyze()

        if len(time_steps_issues) > 0:

            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "{}_{}_{}_mv_overloading".format(scenario, strategy, chunk))

            raise ValueError("Curtailment not sufficient to solve grid "
                             "issues in MV.")

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.loc[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        # elia_logger.debug("No MV overloading issues to solve.")
        pass
    return curtailment


def calculate_curtailment(
        grid_dir,
        edisgo,
        strategy,
        chunk,
):
    try:
        mv_grid_id = int(grid_dir.parts[-2])
        scenario = grid_dir.parts[-3]

        elia_logger = logging.getLogger('elia_project: {}'.format(mv_grid_id))
        elia_logger.setLevel(logging.DEBUG)

        grid_results_dir = os.path.join(
            grid_dir,
            "curtailment",
        )

        os.makedirs(
            grid_results_dir,
            exist_ok=True,
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
            index=["lv_problems", "mv_problems", "convergence_problems"])

        t1 = perf_counter()

        pypsa_network = edisgo.to_pypsa()

        pypsa_network_orig = pypsa_network.copy()

        pf_results = pypsa_network.pf(edisgo.timeseries.timeindex)

        # i = 0
        #
        # converged = False
        #
        # while i < max_iterations and not converged:
        #
        #     try:
        #         pf_results = pypsa_network.pf(edisgo.timeseries.timeindex)
        #
        #         converged = True
        #
        #     except:
        #         timeindex = edisgo.timeseries.residual_load.nsmallest(
        #             int(len(edisgo.timeseries.residual_load) / 20), "0").index
        #
        #         _curtail(
        #             pypsa_network, pypsa_network.generators.index, pypsa_network.loads.index,
        #             edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
        #         )
        #
        #         i += 1

        # print("It took {} seconds for the initial power flow analysis.".format(round(perf_counter() - t1, 0)))

        i = 0

        t1 = perf_counter()

        while i < max_iterations and all(pf_results["converged"]["0"].tolist()) is False:
            _curtail(
                pypsa_network, pypsa_network.generators.index, pypsa_network.loads.index,
                edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist(),
            )

            pf_results = pypsa_network.pf(edisgo.timeseries.timeindex)

            i += 1

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(pypsa_network_orig, pypsa_network)

        curtailment.loc[
            "convergence_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.loc[
            "convergence_problems", "load"] += curtailed_load.sum().sum()

        # print("It took {} seconds to overcome the initial convergence problems.".format(
        #     round(perf_counter() - t1, 0)
        # ))

        pypsa_io.process_pfa_results(edisgo, pypsa_network, edisgo.timeseries.timeindex)

        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        # curtailment due to voltage issues

        # import voltage deviation
        # path = os.path.join(reload_dir, 'voltage_deviation.csv')
        # voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_lv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk)

        # ToDo Only recalculate voltage deviation if curtailment was conducted
        #  (will be done when voltage deviation is attribute in results object)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mvlv_stations_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk)

        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        curtailment = curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, chunk)

        # curtailment due to overloading issues

        # recalculate relative line loading
        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_lv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mvlv_stations_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk, mv_grid_id)

        rel_load = results_helper_functions.relative_load(edisgo)

        curtailment = curtailment_mv_lines_overloading(
            edisgo, curtailment, rel_load, grid_results_dir, scenario, strategy, chunk)

        # check if everything was solved
        voltage_dev = results_helper_functions.voltage_diff(edisgo)
        issues = voltage_dev[
            abs(voltage_dev) > 2e-3].dropna(
            how="all").dropna(axis=1, how="all")
        if not issues.empty:
            print("Not all voltage issues solved in chunk {} of Grid {} with strategy {}.".format(
                chunk, mv_grid_id, strategy
            ))
        else:
            # print("Success. All voltage issues solved in chunk {} of Grid {} with strategy {}.".format(
            #     chunk, mv_grid_id, strategy
            # ))
            pass
        rel_load = results_helper_functions.relative_load(edisgo)
        issues = rel_load[
            rel_load > 1+2e-3].dropna(
            how="all").dropna(axis=1, how="all")
        if not issues.empty:
            print("Not all overloading issues solved in chunk {} of Grid {} with strategy {}.".format(
                chunk, mv_grid_id, strategy
            ))
        else:
            # print("Success. All overloading issues solved in chunk {} of Grid {} with strategy {}.".format(
            #     chunk, mv_grid_id, strategy
            # ))
            pass

        # save curtailment sums
        curtailment.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment.csv".format(scenario, strategy, chunk)))

        # save time series
        curtailed_feedin = feedin_ts - edisgo.timeseries.generators_active_power
        curtailed_load = pd.concat(
            [(load_ts - edisgo.timeseries.loads_active_power),
             (charging_ts - edisgo.timeseries.charging_points_active_power)],
            axis=1)
        curtailed_feedin.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_per_gen.csv".format(scenario, strategy, chunk))
        )
        curtailed_load.to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_per_load.csv".format(scenario, strategy, chunk))
        )
        curtailed_feedin.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_feedin.csv".format(scenario, strategy, chunk))
        )
        curtailed_load.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "{}_{}_{}_curtailment_ts_demand.csv".format(scenario, strategy, chunk))
        )

        # edisgo.timeseries.residual_load.to_csv(
        #     os.path.join(grid_results_dir, "{}_{}_{}_residual_load.csv".format(scenario, strategy, chunk))
        # )

    except Exception as e:
        mv_grid_id = int(grid_dir.parts[-1])
        scenario = grid_dir.parts[-3][:-11]
        print("Error in {} in chunk {} MV grid {}.".format(scenario, chunk, mv_grid_id))
        traceback.print_exc()


# if __name__ == "__main__":
#     if num_threads == 1:
#         for mv_grid_id in mv_grid_ids:
#             calculate_curtailment(mv_grid_id)
#     else:
#         with multiprocessing.Pool(num_threads) as pool:
#             pool.map(calculate_curtailment, mv_grid_ids)
