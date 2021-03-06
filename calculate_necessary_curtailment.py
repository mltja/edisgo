import os
import logging
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing
import traceback
import results_helper_functions
import timeseries_import

from datetime import datetime, timedelta
from edisgo import EDisGo
from pathlib import Path
from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.results import opf_expand_network
from edisgo.tools import pypsa_io
from edisgo.tools.tools import assign_feeder, get_path_length_to_station
from edisgo.network.timeseries import get_component_timeseries

logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")


# # possible options for scenario are "dumb_charging" and "smart_charging"
# scenario = "dumb"
#
# results_base_path = "/home/local/RL-INSTITUT/birgit.schachler/rli-daten_02/open_BEA_Berechnungen"
# #results_base_path = "/home/birgit/Schreibtisch/Elia_Ergebnisse"
# if scenario == "dumb":
#     results_path = os.path.join(
#         results_base_path, "2020-12-11_22-13_simbev_scenario_dumb_charging")
# elif scenario == "reduced":
#     results_path = os.path.join(
#         results_base_path, "2020-12-11_22-14_simbev_scenario_reduced_charging")
# elif scenario == "grouped":
#     results_path = os.path.join(
#         results_base_path, "2020-12-13_18-51_simbev_scenario_grouped_charging")
# elif scenario == "residual_load":
#     results_path = os.path.join(
#         results_base_path, "2021-01-06_17-23_simbev_scenario_residual_load_charging")
# else:
#     raise ValueError
#
# mv_grid_ids = [176, 177, 1056, 1423, 1574, 1690, 1811, 1839,
#                2079, 2095, 2534, 3008, 3280] # 566, 3267
#
# num_threads = 1
# curtailment_step = 0.2  # 0.05 TODO
# max_iterations = 50


def get_stations_and_timesteps_with_issues(voltage_dev, rel_load, mv_grid_id):
    tmp = voltage_dev[voltage_dev != 0].dropna(how="all").dropna(axis=1,
                                                                 how="all")
    stations_voltage_issues = [_ for _ in tmp.columns if
                               _.split("_")[-1] == "LV"]
    time_steps_voltage_issues = tmp.loc[:, stations_voltage_issues].dropna(
        how="all").index

    tmp = rel_load[rel_load > 1].dropna(how="all").dropna(axis=1, how="all")
    stations_load_issues = [_ for _ in tmp.columns if "mvlv_station" in _]
    time_steps_load_issues = tmp.loc[:, stations_load_issues].dropna(
        how="all").index

    stations_load_issues_bus = [
        "BusBar_mvgd_{}_lvgd_{}_LV".format(mv_grid_id, _.split("_")[-1])
        for _ in stations_load_issues]
    station_issues = list(
        set(stations_load_issues_bus + stations_voltage_issues))
    time_steps_issues = time_steps_voltage_issues.append(
        time_steps_load_issues).unique()

    return station_issues, time_steps_issues


def get_mv_lines_and_timesteps_with_issues(voltage_dev, rel_load, mv_grid_id,
                                           tolerance=2e-3):
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(mv_grid_id))

    tmp = voltage_dev[abs(voltage_dev) > tolerance].dropna(how="all").dropna(
        axis=1, how="all")
    buses_issues = [_ for _ in tmp.columns if _.split("_")[-1] != "LV"]
    tmp = tmp.loc[:, buses_issues].dropna(how="all")
    time_steps_voltage_issues = tmp.index

    if not tmp.empty:
        elia_logger.debug("Max. voltage dev.: {}".format(
            tmp.abs().max().max()))

    tmp = rel_load[rel_load > 1 + tolerance].dropna(how="all").dropna(axis=1,
                                                                      how="all")
    lines_issues = [_ for _ in tmp.columns if not "mvlv_station" in _]
    tmp = tmp.loc[:, lines_issues].dropna(how="all")
    time_steps_load_issues = tmp.index

    if not tmp.empty:
        elia_logger.debug("Max. overload: {}".format(tmp.max().max()))

    time_steps_issues = time_steps_voltage_issues.append(
        time_steps_load_issues).unique()

    return lines_issues, buses_issues, time_steps_issues


def get_time_steps_issues_station(voltage_dev, rel_load, station):
    tmp = voltage_dev.loc[:, station]
    ts_voltage_issues = tmp[tmp != 0].index

    tmp = rel_load.loc[:,
          "mvlv_station_LVGrid_{}".format(station.split("_")[-2])]
    ts_load_issues = tmp[tmp > 1].index

    return ts_voltage_issues.append(ts_load_issues).unique()


def get_time_steps_issues_bus(voltage_dev, rel_load, b, edisgo):
    tmp = voltage_dev.loc[:, b]
    ts_voltage_issues = tmp[tmp != 0].dropna(how="all").index

    lines_df = edisgo.topology.lines_df
    lines = lines_df.loc[lines_df.bus0.isin([b]) | lines_df.bus1.isin([b])]
    tmp = rel_load.loc[:, lines.index]
    ts_load_issues = tmp[tmp > 1].dropna(how="all").dropna(axis=1, how="all").index

    return ts_voltage_issues.append(ts_load_issues).unique()


def curtail(edisgo, pypsa_network, voltage_dev, rel_load):
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # get stations and time steps with overloading and voltage issues
    stations_issues, time_steps_issues = get_stations_and_timesteps_with_issues(
        voltage_dev, rel_load, edisgo.topology.id)

    # for each station calculate curtailment
    for station in stations_issues:
        # get loads and gens in grid
        gens_grid = pypsa_network.generators[
            pypsa_network.generators.bus == station].index
        loads_grid = pypsa_network.loads[
            pypsa_network.loads.bus == station].index
        # get time series for loads and gens in grid
        gens_grid_ts = pypsa_network.generators_t.p_set.loc[:, gens_grid]
        loads_grid_ts = pypsa_network.loads_t.p_set.loc[:, loads_grid]

        # evaluate whether it is a load or feed-in case (using residual load)
        # calculate residual load
        residual_load_grid = gens_grid_ts.sum(axis=1) - loads_grid_ts.sum(
            axis=1)
        # get time steps with issues at that station
        ts_issues = get_time_steps_issues_station(voltage_dev, rel_load,
                                                  station)
        # get time steps where to curtail generators and where to curtail loads
        residual_load_grid = residual_load_grid.loc[ts_issues]
        ts_curtail_gens = residual_load_grid[residual_load_grid > 0].index
        ts_curtail_loads = residual_load_grid[residual_load_grid < 0].index

        # curtail loads or generators by specified curtailment factor
        # active power
        pypsa_network.generators_t.p_set.loc[ts_curtail_gens, gens_grid] = (
                gens_grid_ts.loc[ts_curtail_gens, :] -
                curtailment_step * gens_grid_ts.loc[ts_curtail_gens, :])
        pypsa_network.loads_t.p_set.loc[ts_curtail_loads, loads_grid] = (
                loads_grid_ts.loc[ts_curtail_loads, :] -
                curtailment_step * loads_grid_ts.loc[ts_curtail_loads, :])
        # reactive power
        tmp = pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens_grid]
        pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens_grid] = (
                tmp - curtailment_step * tmp)
        tmp = pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads_grid]
        pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads_grid] = (
                tmp - curtailment_step * tmp)

    # run power flow analysis on limited number of time steps
    elia_logger.debug(
        "Number of time steps: {}".format(len(time_steps_issues)))
    pf_results = pypsa_network.pf(time_steps_issues)
    if all(pf_results["converged"]["0"].tolist()):
        pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)
    else:
        raise ValueError("Power flow analysis did not converge for the"
                         "following time steps: {}.".format(
            time_steps_issues[~pf_results["converged"]["0"]].tolist())
        )


def calculate_curtailed_energy(pypsa_network_orig, pypsa_network):
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


def calculate_curtailment_mvlv_stations(
        edisgo, voltage_dev, rel_load, curtailment, max_iterations,
        grid_results_dir, scenario, strategy):
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(edisgo.topology.id))

    # create pypsa network with aggregated loads and generators at station's
    # secondary side
    stations_issues, time_steps_issues = get_stations_and_timesteps_with_issues(
        voltage_dev, rel_load, edisgo.topology.id)
    pypsa_network = edisgo.to_pypsa(mode="mvlv", timesteps=time_steps_issues)

    # save original pypsa network to determine curtailed energy
    pypsa_network_orig = pypsa_network.copy()

    iteration_count = 0
    while len(stations_issues) > 0 and iteration_count < max_iterations:
        curtail(edisgo, pypsa_network, voltage_dev, rel_load)
        curtailed_feedin, curtailed_load = calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
            curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

        # recheck for overloading and voltage issues at stations
        rel_load = results_helper_functions.relative_load(edisgo)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        stations_issues, time_steps_issues = get_stations_and_timesteps_with_issues(
            voltage_dev, rel_load, edisgo.topology.id)
        elia_logger.debug(
            "Remaining number of stations with issues: {}".format(
                len(stations_issues)))

        iteration_count += 1

    if len(stations_issues) > 0:
        edisgo.save(
            os.path.join(
                grid_results_dir,
                "{}_{}_edisgo_curtailment_lv".format(scenario, strategy),
            ),
            parameters="powerflow_results",
        )

        rel_load.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_edisgo_curtailment_lv".format(scenario, strategy),
                "relative_load.csv"),
        )

        voltage_dev.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_edisgo_curtailment_lv".format(scenario, strategy),
                "voltage_deviation.csv")
        )

        raise ValueError("Curtailment not sufficient to solve grid "
                         "issues in LV.")

    curtailed_feedin, curtailed_load = calculate_curtailed_energy(
        pypsa_network_orig, pypsa_network)
    curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
    curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    # overwrite time series in edisgo
    time_steps = pypsa_network.generators_t.p_set.index

    # generators: overwrite time series for all except slack
    gens = pypsa_network_orig.generators[
        pypsa_network_orig.generators.control != "Slack"].index
    edisgo.timeseries._generators_active_power.loc[
        time_steps, gens] = pypsa_network.generators_t.p_set.loc[
        time_steps, gens]
    edisgo.timeseries._generators_reactive_power.loc[
        time_steps, gens] = pypsa_network.generators_t.q_set.loc[
        time_steps, gens]
    # loads: distinguish between charging points and conventional loads
    charging_points = edisgo.topology.charging_points_df.index
    loads = edisgo.topology.loads_df.index
    edisgo.timeseries._loads_active_power.loc[
        time_steps, loads] = pypsa_network.loads_t.p_set.loc[time_steps, loads]
    edisgo.timeseries._charging_points_active_power.loc[
        time_steps, charging_points] = pypsa_network.loads_t.p_set.loc[
        time_steps, charging_points]
    edisgo.timeseries._loads_reactive_power.loc[
        time_steps, loads] = pypsa_network.loads_t.q_set.loc[time_steps, loads]
    edisgo.timeseries._charging_points_reactive_power.loc[
        time_steps, charging_points] = pypsa_network.loads_t.q_set.loc[
        time_steps, charging_points]

    # # save results, timeseries and topology
    # edisgo.save(os.path.join(grid_results_dir, "edisgo_curtailment_lv"),
    #             parameters="powerflow_results")

    # save curtailment time series (per load/generator and sum over all
    # loads/generators
    # curtailed_feedin.to_csv(
    #     os.path.join(grid_results_dir, "curtailment_ts_per_gen_lv.csv")
    # )
    # curtailed_load.to_csv(
    #     os.path.join(grid_results_dir, "curtailment_ts_per_load_lv.csv")
    # )
    # curtailed_feedin.sum(axis=1).to_csv(
    #     os.path.join(grid_results_dir, "curtailment_ts_feedin_lv.csv")
    # )
    # curtailed_load.sum(axis=1).to_csv(
    #     os.path.join(grid_results_dir, "curtailment_ts_demand_lv.csv")
    # )

    return curtailment


def calculate_curtailment_mv_lines(
        edisgo, voltage_dev, rel_load, curtailment, grid_results_dir,
        scenario, strategy):
    mv_grid_id = edisgo.topology.id
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(mv_grid_id))

    lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
        voltage_dev, rel_load, mv_grid_id)

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        # assign main MV feeder and path length to station
        assign_feeder(edisgo)
        get_path_length_to_station(edisgo)

        buses_df = edisgo.topology.buses_df
        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with grid issues
            # get all buses with issues
            buses_lines = edisgo.topology.lines_df.loc[lines_issues, :].loc[:,
                          ["bus0", "bus1"]].stack().unique()
            buses_issues.extend(list(buses_lines))
            buses_issues = set(buses_issues)
            mv_station = edisgo.topology.mv_grid.station.index[0]
            if mv_station in buses_issues:
                buses_issues.remove(mv_station)
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()
            elia_logger.debug(
                "Number of feeders with MV issues: {}".format(len(feeders)))
            elia_logger.debug(
                "Number of buses with MV issues: {}".format(
                    len(buses_df_issues)))

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[
                    buses_df_issues.mv_feeder == feeder]
                b = buses_in_feeder.loc[
                    :, "path_length_to_station"].sort_values(
                    ascending=False).index[0]

                # get all generators and loads downstream
                buses = buses_df[(buses_df.mv_feeder == feeder) &
                                 (buses_df.path_length_to_station >=
                                  buses_in_feeder.at[
                                      b, "path_length_to_station"])]

                gens_grid = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses.index)].index
                loads_grid = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses.index)].index
                loads_grid = loads_grid.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses.index)].index)

                # get time steps with issues at that line
                ts_issues = get_time_steps_issues_bus(
                    voltage_dev,
                    rel_load,
                    b, edisgo)

                # get time series for loads and gens in grid
                gens_grid_ts = pypsa_network.generators_t.p_set.loc[
                               :, gens_grid]
                loads_grid_ts = pypsa_network.loads_t.p_set.loc[
                                :, loads_grid]

                # evaluate whether it is a load or feed-in case (using
                # residual load)
                # calculate residual load
                residual_load_grid = (gens_grid_ts.sum(axis=1) -
                                      loads_grid_ts.sum(axis=1))

                # get time steps where to curtail generators and where to
                # curtail loads
                residual_load_grid = residual_load_grid.loc[ts_issues]
                ts_curtail_gens = residual_load_grid[
                    residual_load_grid > 0].index
                ts_curtail_loads = residual_load_grid[
                    residual_load_grid < 0].index

                # curtail loads or generators by specified curtailment
                # factor active power
                pypsa_network.generators_t.p_set.loc[
                    ts_curtail_gens, gens_grid] = (
                        gens_grid_ts.loc[ts_curtail_gens, :] -
                        curtailment_step *
                        gens_grid_ts.loc[ts_curtail_gens, :])
                pypsa_network.loads_t.p_set.loc[
                    ts_curtail_loads, loads_grid] = (
                        loads_grid_ts.loc[ts_curtail_loads, :] -
                        curtailment_step *
                        loads_grid_ts.loc[ts_curtail_loads, :])
                # reactive power
                tmp = pypsa_network.generators_t.q_set.loc[
                    ts_curtail_gens, gens_grid]
                pypsa_network.generators_t.q_set.loc[
                    ts_curtail_gens, gens_grid] = (
                        tmp - curtailment_step * tmp)
                tmp = pypsa_network.loads_t.q_set.loc[
                    ts_curtail_loads, loads_grid]
                pypsa_network.loads_t.q_set.loc[
                    ts_curtail_loads, loads_grid] = (
                        tmp - curtailment_step * tmp)

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

            # recheck for overloading and voltage issues
            rel_load = results_helper_functions.relative_load(edisgo)
            voltage_dev = results_helper_functions.voltage_diff(edisgo)

            curtailed_feedin, curtailed_load = calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
                voltage_dev, rel_load, mv_grid_id)
            elia_logger.debug(
                "Remaining number of time steps with issues: {}".format(
                    len(time_steps_issues)))
            iteration_count += 1

        # calculate curtailment
        if len(time_steps_issues) > 0:
            edisgo.save(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv".format(scenario, strategy),
                ),
                parameters="powerflow_results",
            )
            rel_load.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv".format(scenario, strategy),
                    "relative_load.csv",
                )
            )
            voltage_dev.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv".format(scenario, strategy),
                    "voltage_deviation.csv")
            )
            raise ValueError("Curtailment not sufficient to solve grid "
                             "issues in MV.")

        curtailed_feedin, curtailed_load = calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.at[
            "mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

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
        charging_points = edisgo.topology.charging_points_df.index
        loads = edisgo.topology.loads_df.index
        edisgo.timeseries._loads_active_power.loc[
            time_steps, loads] = pypsa_network.loads_t.p_set.loc[
            time_steps, loads]
        edisgo.timeseries._charging_points_active_power.loc[
            time_steps, charging_points] = pypsa_network.loads_t.p_set.loc[
            time_steps, charging_points]
        edisgo.timeseries._loads_reactive_power.loc[
            time_steps, loads] = pypsa_network.loads_t.q_set.loc[
            time_steps, loads]
        edisgo.timeseries._charging_points_reactive_power.loc[
            time_steps, charging_points] = pypsa_network.loads_t.q_set.loc[
            time_steps, charging_points]

        # # save results, timeseries and topology
        # edisgo.save(os.path.join(grid_results_dir, "edisgo_curtailment_mv"),
        #             parameters="powerflow_results")
        # curtailed_feedin.to_csv(
        #     os.path.join(grid_results_dir, "curtailment_ts_per_gen_mv.csv")
        # )
        # curtailed_load.to_csv(
        #     os.path.join(grid_results_dir, "curtailment_ts_per_load_mv.csv")
        # )
        # curtailed_feedin.sum(axis=1).to_csv(
        #     os.path.join(grid_results_dir, "curtailment_ts_feedin_mv.csv")
        # )
        # curtailed_load.sum(axis=1).to_csv(
        #     os.path.join(grid_results_dir, "curtailment_ts_demand_mv.csv")
        # )
    else:
        elia_logger.debug("No MV issues to solve.")
    return curtailment


def calculate_curtailment_mv_voltage(
        edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy, tolerance=2e-3):
    mv_grid_id = edisgo.topology.id
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(mv_grid_id))

    # assign main MV feeder
    assign_feeder(edisgo)

    # find voltage issues in mv
    tmp = voltage_dev[abs(voltage_dev) > tolerance].dropna(how="all").dropna(
        axis=1, how="all")
    buses_issues = [_ for _ in tmp.columns if _.split("_")[-1] != "LV"]
    voltage_issues = tmp.loc[:, buses_issues].dropna(how="all")
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders where voltage deviation is larger than 10%
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()

            elia_logger.debug(
                "Number of feeders with 10% issues: {}".format(len(feeders)))
            elia_logger.debug(
                "Number of time steps with 10% issues: {}".format(
                    len(time_steps_issues)))

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.mv_feeder == feeder].index

                gens_grid = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)].index
                loads_grid = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)].index
                loads_grid = loads_grid.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(
                            buses)].index)

                # get time steps with issues at that line
                ts_issues = voltage_issues.loc[:, buses].dropna(how="all").index

                # get time series for loads and gens in grid
                gens_grid_ts = pypsa_network.generators_t.p_set.loc[
                               :, gens_grid]
                loads_grid_ts = pypsa_network.loads_t.p_set.loc[
                                :, loads_grid]

                # evaluate whether it is a load or feed-in case (using
                # residual load)
                # calculate residual load
                residual_load_grid = (gens_grid_ts.sum(axis=1) -
                                      loads_grid_ts.sum(axis=1))

                # get time steps where to curtail generators and where to
                # curtail loads
                residual_load_grid = residual_load_grid.loc[ts_issues]
                ts_curtail_gens = residual_load_grid[
                    residual_load_grid > 0].index
                ts_curtail_loads = residual_load_grid[
                    residual_load_grid < 0].index

                # curtail loads or generators by specified curtailment
                # factor active power
                pypsa_network.generators_t.p_set.loc[
                    ts_curtail_gens, gens_grid] = (
                        gens_grid_ts.loc[ts_curtail_gens, :] -
                        curtailment_step *
                        gens_grid_ts.loc[ts_curtail_gens, :])
                pypsa_network.loads_t.p_set.loc[
                    ts_curtail_loads, loads_grid] = (
                        loads_grid_ts.loc[ts_curtail_loads, :] -
                        curtailment_step *
                        loads_grid_ts.loc[ts_curtail_loads, :])
                # reactive power
                tmp = pypsa_network.generators_t.q_set.loc[
                    ts_curtail_gens, gens_grid]
                pypsa_network.generators_t.q_set.loc[
                    ts_curtail_gens, gens_grid] = (
                        tmp - curtailment_step * tmp)
                tmp = pypsa_network.loads_t.q_set.loc[
                    ts_curtail_loads, loads_grid]
                pypsa_network.loads_t.q_set.loc[
                    ts_curtail_loads, loads_grid] = (
                        tmp - curtailment_step * tmp)

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

            curtailed_feedin, curtailed_load = calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network)
            elia_logger.debug("Curtailed energy (feed-in/load): {}, {}".format(
                curtailed_feedin.sum().sum(), curtailed_load.sum().sum()))

            # find MV voltage issues
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            tmp = voltage_dev[abs(voltage_dev) > tolerance].dropna(
                how="all").dropna(
                axis=1, how="all")
            buses_issues = [_ for _ in tmp.columns if _.split("_")[-1] != "LV"]
            voltage_issues = tmp.loc[:, buses_issues].dropna(how="all")
            time_steps_issues = voltage_issues.index

            elia_logger.debug(
                "Remaining number of time steps with issues: {}".format(
                    len(time_steps_issues)))
            iteration_count += 1

        rel_load = results_helper_functions.relative_load(edisgo)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        # calculate curtailment
        if len(time_steps_issues) > 0:
            edisgo.save(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv_voltage".format(scenario, strategy),
                ),
                parameters="powerflow_results",
            )
            rel_load.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv_voltage".format(scenario, strategy),
                    "relative_load.csv"
                )
            )

            voltage_dev.to_csv(
                os.path.join(
                    grid_results_dir,
                    "{}_{}_edisgo_curtailment_mv_voltage".format(scenario, strategy),
                    "voltage_deviation.csv"
                )
            )

            raise ValueError("Curtailment not sufficient to solve MV voltage "
                             "issues.")

        curtailed_feedin, curtailed_load = calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network)
        curtailment.at["mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

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
        charging_points = edisgo.topology.charging_points_df.index
        loads = edisgo.topology.loads_df.index
        edisgo.timeseries._loads_active_power.loc[
            time_steps, loads] = pypsa_network.loads_t.p_set.loc[
            time_steps, loads]
        edisgo.timeseries._charging_points_active_power.loc[
            time_steps, charging_points] = pypsa_network.loads_t.p_set.loc[
            time_steps, charging_points]
        edisgo.timeseries._loads_reactive_power.loc[
            time_steps, loads] = pypsa_network.loads_t.q_set.loc[
            time_steps, loads]
        edisgo.timeseries._charging_points_reactive_power.loc[
            time_steps, charging_points] = pypsa_network.loads_t.q_set.loc[
            time_steps, charging_points]

    else:
        elia_logger.debug("No MV voltage issues to solve.")
    return curtailment


def calculate_curtailment(
        grid_dir,
        edisgo,
        strategy,
):
    try:
        mv_grid_id = int(grid_dir.parts[-1])
        scenario = grid_dir.parts[-3][:-11]

        elia_logger = logging.getLogger('elia_project: {}'.format(mv_grid_id))
        elia_logger.setLevel(logging.DEBUG)

        results_path = Path(
            os.path.join(
                grid_dir.parent.parent.parent,
                "eDisGo_curtailment_results",
            )
        )

        grid_results_dir = os.path.join(
            results_path, str(mv_grid_id))
        # reload_dir = os.path.join(
        #     results_path, str(mv_grid_id))

        os.makedirs(
            grid_results_dir,
            exist_ok=True,
        )

        # # reimport edisgo object
        # edisgo = import_edisgo_from_files(
        #     reload_dir,
        #     import_timeseries=True,
        #     import_results=True,
        #     parameters="powerflow_results",
        #     import_residual_load=False
        # )

        feedin_ts = edisgo.timeseries.generators_active_power.copy()
        load_ts = edisgo.timeseries.loads_active_power.copy()
        charging_ts = edisgo.timeseries.charging_points_active_power.copy()

        # import relative load and voltage deviation for all 8760 time steps
        edisgo.analyze()

        rel_load = results_helper_functions.relative_load(edisgo)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)

        # path = os.path.join(reload_dir, 'relative_load.csv')
        # rel_load = pd.read_csv(path, index_col=0, parse_dates=True)
        # path = os.path.join(reload_dir, 'voltage_deviation.csv')
        # voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)

        curtailment = pd.DataFrame(
            data=0,
            columns=["feed-in", "load"],
            index=["lv_problems", "mv_problems"])

        curtailment = calculate_curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir, scenario, strategy)

        curtailment = calculate_curtailment_mvlv_stations(
            edisgo, voltage_dev, rel_load, curtailment, max_iterations,
            grid_results_dir, scenario, strategy)

        curtailment.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_lv_problems.csv".format(scenario, strategy),
            )
        )

        curtailment = calculate_curtailment_mv_lines(
            edisgo, voltage_dev, rel_load, curtailment, grid_results_dir, scenario, strategy)

        curtailment.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment.csv".format(scenario, strategy)
            )
        )

        # save time series
        curtailed_feedin = feedin_ts - edisgo.timeseries.generators_active_power
        curtailed_load = pd.concat(
            [(load_ts - edisgo.timeseries.loads_active_power),
             (charging_ts - edisgo.timeseries.charging_points_active_power)],
            axis=1)
        curtailed_feedin.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_ts_per_gen.csv".format(scenario, strategy)
            )
        )
        curtailed_load.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_ts_per_load.csv".format(scenario, strategy)
            )
        )
        curtailed_feedin.sum(axis=1).to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_ts_feedin.csv".format(scenario, strategy)
            )
        )
        curtailed_load.sum(axis=1).to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_ts_demand.csv".format(scenario, strategy)
            )
        )

        # calculate curtailment of generators in the LV
        curtailment_lv = pd.DataFrame(
            columns=["feed-in", "load"],
            index=["lv", "mv"])
        curtailment_lv.at["lv", "load"] = curtailment.loc[:, "load"].sum()
        curtailment_lv.at["mv", "load"] = 0
        if curtailment.at["mv_problems", "feed-in"] > 0:
            cols = curtailed_feedin.columns
            gens_lv = [_ for _ in cols if ("lvgd" in _ or "LVGrid" in _)]
            gens_mv = [_ for _ in cols if not ("lvgd" in _ or "LVGrid" in _)]
            gens_lv_curtailment = curtailed_feedin.loc[:, gens_lv].sum().sum()
            gens_mv_curtailment = curtailed_feedin.loc[:, gens_mv].sum().sum()

            curtailment_lv.at["lv", "feed-in"] = gens_lv_curtailment
            curtailment_lv.at["mv", "feed-in"] = gens_mv_curtailment
        else:
            curtailment_lv.at["lv", "feed-in"] = curtailment.loc[
                "lv_problems", "feed-in"]
            curtailment_lv.at["mv", "feed-in"] = 0

        curtailment_lv.to_csv(
            os.path.join(
                grid_results_dir,
                "{}_{}_curtailment_lv.csv".format(scenario, strategy)
            )
        )

    except Exception as e:
        mv_grid_id = int(grid_dir.parts[-1])
        print("Error in MV grid {}.".format(mv_grid_id))
        traceback.print_exc()


def reinforce_transformers_and_lines(
        edisgo,
        by=0,
        mode="mvlv",
):
    try:
        if mode == "mvlv":
            # lines
            edisgo.topology.lines_df.s_nom *= (1 + by)
            edisgo.topology.lines_df.r /= (1 + by)
            edisgo.topology.lines_df.x /= (1 + by)

        elif mode == "lv":
            edisgo.topology.lines_df[
                edisgo.topology.lines_df.type_info.str.contains("NAYY")
            ] = edisgo.topology.lines_df[
                edisgo.topology.lines_df.type_info.str.contains("NAYY")
            ].assign(
                s_nom=edisgo.topology.lines_df[
                    edisgo.topology.lines_df.type_info.str.contains("NAYY")
                ].s_nom.multiply(1 + by),
                r=edisgo.topology.lines_df[
                    edisgo.topology.lines_df.type_info.str.contains("NAYY")
                ].r.divide(1 + by),
                x=edisgo.topology.lines_df[
                    edisgo.topology.lines_df.type_info.str.contains("NAYY")
                ].x.divide(1 + by),
            )

        if "lv" in mode:
            # transformers mv-lv
            df_t_lv = edisgo.topology.transformers_df.copy()
            df_eq = edisgo.topology.equipment_data["lv_transformers"].copy()

            for bus in df_t_lv.bus0.unique():
                df_t_bus = df_t_lv.copy().loc[
                    df_t_lv.bus0 == bus
                    ]

                s_nom_bus = df_t_bus.s_nom.sum()

                count_t = len(df_t_bus)

                bus0 = df_t_bus.bus0.iat[0]
                bus1 = df_t_bus.bus1.iat[0]

                name = df_t_bus.index.tolist()[0].rsplit("_", 1)[0] + "_"

                df_eq_bus = df_eq.copy().loc[
                    df_eq.S_nom >= by * s_nom_bus
                    ]

                if not df_eq_bus.empty:
                    index = name + str(count_t + 1)
                    x_pu = df_eq_bus.x_pu.iat[0]
                    r_pu = df_eq_bus.r_pu.iat[0]
                    s_nom = df_eq_bus.S_nom.iat[0]

                    df_t_lv.loc[index] = [
                        bus0,
                        bus1,
                        x_pu,
                        r_pu,
                        s_nom,
                        "",
                    ]

                else:
                    p = 1
                    while p < by * s_nom_bus:
                        p += 1

                    names = [
                        name + str(count_t + i) for i in range(1, p + 1)
                    ]

                    x_pu = df_eq.x_pu.iat[-1]
                    r_pu = df_eq.r_pu.iat[-1]
                    s_nom = df_eq.S_nom.iat[-1]

                    for index in names:
                        df_t_lv.loc[index] = [
                            bus0,
                            bus1,
                            x_pu,
                            r_pu,
                            s_nom,
                            "",
                        ]

            df_t_lv = df_t_lv.sort_index()

        if "mv" in mode:
            # transformers hv-mv
            df_t_mv = edisgo.topology.transformers_hvmv_df.copy()
            df_eq = edisgo.topology.equipment_data["mv_transformers"].copy()

            s_nom = df_t_mv.s_nom.sum()

            count_t = len(df_t_mv)

            bus0 = df_t_mv.bus0.iat[0]
            bus1 = df_t_mv.bus1.iat[0]

            name = df_t_mv.index.tolist()[0].rsplit("_", 1)[0] + "_"

            df_eq_ok = df_eq.copy().loc[
                df_eq.S_nom >= by * s_nom
                ]

            if not df_eq_ok.empty:
                index = name + str(count_t + 1)
                s_nom = df_eq_ok.S_nom.iat[0]

                df_t_mv.loc[index] = [
                    bus0,
                    bus1,
                    s_nom,
                    np.nan,
                    np.nan,
                    "",
                    "",
                ]

            else:
                p = 63
                j = 1
                while p < by * s_nom:
                    p += 63
                    j += 1

                names = [
                    name + str(count_t + i) for i in range(1, j)
                ]

                s_nom = df_eq.S_nom.iat[-1]

                for index in names:
                    df_t_mv.loc[index] = [
                        bus0,
                        bus1,
                        s_nom,
                        np.nan,
                        np.nan,
                        "",
                        "",
                    ]

            df_t_mv = df_t_mv.sort_index()

            edisgo.topology.transformers_df = df_t_lv.copy()
            edisgo.topology.transformers_hvmv_df = df_t_mv.copy()

        return edisgo

    except:
        traceback.print_exc()


def integrate_public_charging(
        ding0_dir,
        grid_dir,
        grid_id,
        files,
        date="2011-01-01",
        generator_scenario="ego100",
        days=None,
        mode="full",
):
    try:
        len_timeindex = 8760

        timeindex = pd.date_range(
            date,
            periods=len_timeindex,
            freq='H',
        )

        # https://www.bmwi.de/Redaktion/DE/Publikationen/Energie/erneuerbare-energien-in-zahlen-2019.pdf?__blob=publicationFile&v=6#page=16

        p_bio = 9983  # MW
        e_bio = 50009  # GWh

        vls_bio = e_bio / (p_bio / 1000)

        share_bio = vls_bio / 8760

        p_hydro = 5595  # MW
        e_hydro = 20058  # GWh

        vls_hydro = e_hydro / (p_hydro / 1000)

        share_hydro = vls_hydro / 8760

        timeseries_generation_dispatchable = pd.DataFrame(
            {
                "biomass": [share_bio] * len_timeindex,
                "other": [share_hydro] * len_timeindex,
            },
            index=timeindex,
        )

        edisgo = EDisGo(
            ding0_grid=os.path.join(
                ding0_dir, str(grid_id)
            ),
            generator_scenario=generator_scenario,
            timeseries_load="demandlib",
            timeseries_generation_fluctuating="oedb",
            timeseries_generation_dispatchable=timeseries_generation_dispatchable,
            timeindex=timeindex,
        )

        timeseries_data_path = os.path.join(
            grid_dir.parent.parent.parent.parent,
            r"hp.csv",
        )

        timeseries_HP = timeseries_import.get_residential_heat_pump_timeseries(
            timeseries_data_path,
        )

        # add heat pump load to residential load
        residential_annual_consumption_ego = 131166982.09864908  # MWh

        df_topology_residential = edisgo.topology.loads_df[edisgo.topology.loads_df.sector == "residential"]

        df_topology_residential = df_topology_residential.assign(
            consumption_factor=df_topology_residential.annual_consumption.divide(
                df_topology_residential.annual_consumption.sum()
            )
        )

        grid_factor = df_topology_residential.annual_consumption.sum() / residential_annual_consumption_ego

        timeseries_HP = timeseries_HP.assign(
            HP_grid=timeseries_HP.HP * grid_factor
        )

        for col, con_factor in list(
                zip(
                    df_topology_residential.index.tolist(),
                    df_topology_residential.consumption_factor.tolist(),
                )
        ):
            edisgo.timeseries._loads_active_power[col] += timeseries_HP.HP_grid.multiply(con_factor)

        edisgo.topology.loads_df.annual_consumption = [
            edisgo.timeseries._loads_active_power[col].sum() for col in edisgo.topology.loads_df.index.tolist()
        ]

        edisgo.topology.loads_df.peak_load = [
            edisgo.timeseries._loads_active_power[col].max() for col in edisgo.topology.loads_df.index.tolist()
        ]

        if mode == "full":
            timeindex = pd.date_range(
                date,
                periods=len_timeindex * 4,
                freq="15min",
            )
        elif mode == "half-year":
            timeindex = pd.date_range(
                date,
                periods=len_timeindex * 4 / 2,
                freq="15min",
            )
        elif mode == "weeks":
            timeindeces = [0] * 2

            len_week = 96 * 7

            for count, day in enumerate(days):
                timeindeces[count] = pd.date_range(
                    day,
                    periods=len_week,
                    freq="15min",
                )

            timeindex = pd.DatetimeIndex(
                np.sort(
                    np.concatenate(
                        (
                            timeindeces[0],
                            timeindeces[1],
                        ),
                        axis=None,
                    ),
                    axis=0,
                )
            )

        edisgo.timeseries.timeindex = timeindex

        edisgo.timeseries.charging_points_active_power = pd.DataFrame(index=timeindex)
        edisgo.timeseries.charging_points_reactive_power = pd.DataFrame(index=timeindex)

        edisgo.timeseries.generators_active_power = edisgo.timeseries.generators_active_power.ffill()
        edisgo.timeseries.generators_reactive_power = edisgo.timeseries.generators_reactive_power.ffill()
        edisgo.timeseries.loads_active_power = edisgo.timeseries.loads_active_power.ffill()
        edisgo.timeseries.loads_reactive_power = edisgo.timeseries.loads_reactive_power.ffill()

        edisgo.timeseries.storage_units_active_power = pd.DataFrame(index=edisgo.timeseries.timeindex)
        edisgo.timeseries.storage_units_reactive_power = pd.DataFrame(index=edisgo.timeseries.timeindex)

        edisgo.timeseries.generation_dispatchable = edisgo.timeseries.generation_dispatchable.append(
            pd.Series(
                name=edisgo.timeseries.generation_dispatchable.index.tolist()[-1] + timedelta(hours=1)
            )
        )

        edisgo.timeseries.generation_dispatchable = edisgo.timeseries.generation_dispatchable.append(
            pd.Series(
                name=edisgo.timeseries.generation_dispatchable.index.tolist()[-1] + timedelta(hours=1)
            )
        ).resample("15min").ffill().iloc[:-1]

        edisgo.timeseries.generation_fluctuating = edisgo.timeseries.generation_fluctuating.append(
            pd.Series(
                name=edisgo.timeseries.generation_fluctuating.index.tolist()[-1] + timedelta(hours=1)
            )
        ).resample("15min").ffill().iloc[:-1]

        edisgo.timeseries.load = edisgo.timeseries.load.append(
            pd.Series(
                name=edisgo.timeseries.load.index.tolist()[-1] + timedelta(hours=1)
            )
        ).resample("15min").ffill().iloc[:-1]

        comp_type = "ChargingPoint"

        ts_reactive_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_reactive_power",
        )

        geo_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("geojson" in f and "mapped" not in f and ("public" in f or "hpc" in f))
        ]

        geo_files.sort()

        ts_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("h5" in f and ("public" in f or "hpc" in f))
        ]

        ts_files.sort()

        if len(ts_files) == 2:
            use_cases = ["fast", "public"]
        else:
            use_cases = ["public"]

        for geo_f, ts_f, use_case in list(
                zip(
                    geo_files,
                    ts_files,
                    use_cases,
                )
        ):
            gdf = gpd.read_file(
                geo_f,
            )

            df = pd.read_hdf(
                ts_f,
                key="df_load",
            )

            temp_timeindex = pd.date_range(
                "2011-01-01",
                periods=len(df),
                freq="15min",
            )

            df.index = temp_timeindex

            df = df.loc[timeindex].divide(1000)  # kW -> MW

            if "cp_idx" not in gdf.columns:
                if len(df.columns.levels[1]) == 1:
                    cp_idx = df.columns.levels[1][0]

                    gdf = gdf.assign(
                        cp_idx=cp_idx,
                    )

                else:
                    raise ValueError("Something is wrong with the cp_idx in grid {}.".format(grid_id))

            # gdf = refactor_gdf(gdf, mode="only-mv") # TODO

            # if len(gdf) > 3:
            #     gdf = gdf.iloc[:3] # TODO

            # gdf.cp_connection_rating = 350 # TODO

            if not gdf.empty:
                # TODO: choose
                # _ = [
                #     EDisGo.integrate_component(
                #         edisgo,
                #         comp_type=comp_type,
                #         geolocation=geolocation,
                #         use_case=use_case,
                #         add_ts=True,
                #         ts_active_power=df.loc[:, (ags, cp_idx)],
                #         ts_reactive_power=ts_reactive_power,
                #         p_nom=df.loc[:, (ags, cp_idx)].max(),
                #     ) for ags, cp_idx, geolocation in list(
                #         zip(
                #             gdf.ags.tolist(),
                #             gdf.cp_idx.tolist(),
                #             gdf.geometry.tolist(),
                #         )
                #     )
                # ]

                edisgo_id = [
                    EDisGo.integrate_component(
                        edisgo,
                        comp_type=comp_type,
                        geolocation=geolocation,
                        use_case=use_case,
                        add_ts=True,
                        ts_active_power=df.loc[:, (ags, cp_idx)],#.divide(divisor), # TODO
                        ts_reactive_power=ts_reactive_power,
                        p_nom=p_nom,
                    ) for ags, cp_idx, geolocation, p_nom in list(#, divisor in list( # TODO
                        zip(
                            gdf.ags.tolist(),
                            gdf.cp_idx.tolist(),
                            gdf.geometry.tolist(),
                            gdf.cp_connection_rating.divide(1000).tolist(),  # kW -> MW
                            # gdf.divisor.tolist(), # TODO
                        )
                    )
                ]

                gdf = gdf.assign(edisgo_id=edisgo_id)

                export_f = "mapped_" + geo_f.parts[-1]

                export_path = os.path.join(geo_f.parent, export_f)

                gdf.to_file(
                    export_path,
                    driver="GeoJSON",
                )

        return edisgo

    except:
        traceback.print_exc()


def integrate_private_charging(
        edisgo,
        grid_dir,
        files,
        strategy,
):
    try:
        geo_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("geojson" in f and "mapped" not in f and ("home" in f or "work" in f))
        ]

        geo_files.sort()

        ts_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("h5" in f and strategy in f and ("home" in f or "work" in f))
        ]

        ts_files.sort()

        use_cases = ["home", "work"]

        timeindex = edisgo.timeseries.timeindex

        comp_type = "ChargingPoint"

        ts_reactive_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_reactive_power",
        )

        cp_matching_dfs = [0] * len(geo_files)

        for count_files, (geo_f, ts_f, use_case) in enumerate(
            list(
                zip(
                    geo_files,
                    ts_files,
                    use_cases,
                )
            )
        ):
            gdf = gpd.read_file(
                geo_f,
            )

            df = pd.read_hdf(
                ts_f,
                key="df_load",
            )

            temp_timeindex = pd.date_range(
                "2011-01-01",
                periods=len(df),
                freq="15min",
            )

            df.index = temp_timeindex

            df = df.loc[timeindex].divide(1000) # kW -> MW

            if not "cp_idx" in gdf.columns:
                if len(df.columns.level[1]) == 1:
                    cp_idx = df.columns.level[1][0]

                    gdf = gdf.assign(
                        cp_idx=cp_idx,
                    )
                else:
                    raise ValueError("Something is wrong with the cp_idx in grid {}.".format(grid_dir.parts[-1]))

            # gdf = refactor_gdf(gdf, mode="only-mv") # TODO

            # if len(gdf) > 3:
            #     gdf = gdf.iloc[:3] # TODO

            cp_matching_dfs[count_files] = pd.DataFrame(index=[*range(len(gdf))])

            if not gdf.empty:
                # TODO: choose
                # _ = [
                #     EDisGo.integrate_component(
                #         edisgo,
                #         comp_type=comp_type,
                #         geolocation=geolocation,
                #         use_case=use_case,
                #         add_ts=True,
                #         ts_active_power=df.loc[:, (ags, cp_idx)],
                #         ts_reactive_power=ts_reactive_power,
                #         p_nom=df.loc[:, (ags, cp_idx)].max(),
                #     ) for ags, cp_idx, geolocation in list(
                #         zip(
                #             gdf.ags.tolist(),
                #             gdf.cp_idx.tolist(),
                #             gdf.geometry.tolist(),
                #         )
                #     )
                # ]

                edisgo_id = [
                    EDisGo.integrate_component(
                        edisgo,
                        comp_type=comp_type,
                        geolocation=geolocation,
                        use_case=use_case,
                        add_ts=True,
                        ts_active_power=df.loc[:, (ags, cp_idx)], #.divide(divisor), # TODO
                        ts_reactive_power=ts_reactive_power,
                        p_nom=p_nom,
                    ) for ags, cp_idx, geolocation, p_nom in list(#, divisor in list( # TODO
                        zip(
                            gdf.ags.tolist(),
                            gdf.cp_idx.tolist(),
                            gdf.geometry.tolist(),
                            gdf.cp_connection_rating.divide(1000).tolist(),  # kW -> MW
                            # gdf.divisor.tolist(), # TODO
                        )
                    )
                ]

                cp_matching_dfs[count_files]["edisgo_id"] = edisgo_id
                cp_matching_dfs[count_files]["ags"] = gdf.ags.tolist()
                cp_matching_dfs[count_files]["cp_idx"] = gdf.cp_idx.tolist()

                gdf = gdf.assign(edisgo_id=edisgo_id)

                export_f = "mapped_" + geo_f.parts[-1]

                export_path = os.path.join(geo_f.parent, export_f)

                gdf.to_file(
                    export_path,
                    driver="GeoJSON",
                )

        new_switch_line = edisgo.topology.lines_df.loc[
            (edisgo.topology.lines_df["bus0"] == edisgo.topology.switches_df.at["circuit_breaker_1", "bus_open"]) |
            (edisgo.topology.lines_df["bus1"] == edisgo.topology.switches_df.at["circuit_breaker_1", "bus_open"])
            ].index.tolist()[0]

        edisgo.topology.switches_df.at["circuit_breaker_1", "branch"] = new_switch_line

        # grid_results_dir = Path(
        #     os.path.join(  # TODO: set dir
        #         # r"\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_plot_data\generators",
        #         r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results/eDisGo_plot_data/generators",
        #         grid_dir.parts[-1],
        #     )
        # )
        #
        # os.makedirs(
        #     grid_results_dir,
        #     exist_ok=True,
        # )
        #
        # edisgo.save(
        #     grid_results_dir,
        #     save_results=False,
        #     save_topology=True,
        #     save_timeseries=False,
        # )
        #
        # print("Grid {} saved.".format(grid_dir.parts[-1]))

        return edisgo, cp_matching_dfs[0], cp_matching_dfs[1]

    except:
        traceback.print_exc()


def refactor_gdf(gdf, mode="lv", max_voltage_lv=300, max_voltage_mv=4500):
    try:
        if mode is "lv":
            gdf["divisor"] = gdf.cp_connection_rating.divide(max_voltage_lv).apply(np.ceil).astype(int)

            gdf.cp_capacity = gdf.cp_capacity.divide(gdf.divisor)
            gdf.cp_connection_rating = gdf.cp_connection_rating.divide(gdf.divisor)

            gdf_result = gdf.copy()[0:0]

            for count, divisor in enumerate(gdf.divisor.tolist()):
                gdf_result = gdf_result.append([gdf.iloc[count].to_frame().T] * divisor)

        elif mode == "mv":
            gdf_result = gdf.copy()[
                (gdf.cp_connection_rating > max_voltage_lv)
            ]

            gdf_result["divisor"] = 1

        elif mode == "only-mv":
            gdf_result = gdf.copy()[
                (gdf.cp_connection_rating > max_voltage_lv) &
                (gdf.cp_connection_rating <= max_voltage_mv)
                ]

            gdf_result["divisor"] = 1

            gdf_result = gdf_result.iloc[int(np.ceil(len(gdf_result) * 49 / 96)):]  # TODO
            # gdf_result = gdf_result.iloc[int(np.ceil(len(gdf_result)/2)):int(np.ceil(len(gdf_result)*3/4))] # TODO

        elif mode == "only-hvmv":
            gdf_result = gdf.copy()[
                (gdf.cp_connection_rating > max_voltage_mv)
            ]

            gdf_result["divisor"] = 1

        return gdf_result
    except:
        traceback.print_exc()

# if __name__ == "__main__":
#     if num_threads == 1:
#         for mv_grid_id in mv_grid_ids:
#             calculate_curtailment(mv_grid_id)
#     else:
#         with multiprocessing.Pool(num_threads) as pool:
#             pool.map(calculate_curtailment, mv_grid_ids)
