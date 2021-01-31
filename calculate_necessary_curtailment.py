import os
import logging
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing
import traceback

from edisgo import EDisGo
from pathlib import Path
from edisgo.edisgo import import_edisgo_from_files
# from elia_project import results_helper_functions # TODO: @Birgit
from edisgo.opf.results import opf_expand_network
from edisgo.tools import pypsa_io
from edisgo.network.timeseries import TimeSeries
# from edisgo.tools.tools import assign_mv_feeder_to_buses_and_lines, \ # TODO: @Birgit
#     get_path_length_to_station

logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

# possible options for scenario are "dumb_charging" and "smart_charging"
scenario = "dumb"

results_base_path = "/home/local/RL-INSTITUT/birgit.schachler/rli-daten_02/open_BEA_Berechnungen"
#results_base_path = "/home/birgit/Schreibtisch/Elia_Ergebnisse"
if scenario == "dumb":
    results_path = os.path.join(
        results_base_path, "2020-12-11_22-13_simbev_scenario_dumb_charging")
elif scenario == "reduced":
    results_path = os.path.join(
        results_base_path, "2020-12-11_22-14_simbev_scenario_reduced_charging")
elif scenario == "grouped":
    results_path = os.path.join(
        results_base_path, "2020-12-13_18-51_simbev_scenario_grouped_charging")
elif scenario == "residual_load":
    results_path = os.path.join(
        results_base_path, "2021-01-06_17-23_simbev_scenario_residual_load_charging")
else:
    raise ValueError

mv_grid_ids = [176, 177, 1056, 1423, 1574, 1690, 1811, 1839,
               2079, 2095, 2534, 3008, 3280] # 566, 3267

num_threads = 1
curtailment_step = 0.05
max_iterations = 50


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
        grid_results_dir):

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
            os.path.join(grid_results_dir, "edisgo_curtailment_lv"),
            parameters="powerflow_results")
        rel_load.to_csv(
            os.path.join(grid_results_dir, "edisgo_curtailment_lv",
                         "relative_load.csv"))
        voltage_dev.to_csv(
            os.path.join(grid_results_dir, "edisgo_curtailment_lv",
                         "voltage_deviation.csv"))
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


def calculate_curtailment_mv_lines_opf(
        edisgo, voltage_dev, rel_load, curtailment, grid_results_dir):

    mv_grid_id = edisgo.topology.id
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(mv_grid_id))
    curtailment.at["mv_problems", "feed-in"], curtailment.at[
        "mv_problems", "load"] = [0, 0]

    lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
        voltage_dev, rel_load, mv_grid_id)
    pypsa_network = edisgo.to_pypsa(mode="mvlv", timesteps=time_steps_issues)

    if len(lines_issues) > 0:
        # run power flow analysis on limited number of time steps
        pf_results = pypsa_network.pf(time_steps_issues)
        if all(pf_results["converged"]["0"].tolist()):
            pypsa_io.process_pfa_results(edisgo, pypsa_network, time_steps_issues)
        else:
            raise ValueError("Power flow analysis did not converge for the"
                             "following time steps: {}.".format(
                time_steps_issues[~pf_results["converged"]["0"]].tolist())
            )

        # recheck for overloading and voltage issues
        rel_load = results_helper_functions.relative_load(edisgo)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)
        lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
            voltage_dev, rel_load, mv_grid_id)
        elia_logger.debug(
            "Remaining number of time steps with MV issues: {}".format(
                len(time_steps_issues)))

        # differentiate between load and feed-in case
        timesteps_load_feedin_case = \
        edisgo.timeseries.timesteps_load_feedin_case.loc[time_steps_issues]
        time_steps_issues_lc = [_ for _ in timesteps_load_feedin_case.index
                                if timesteps_load_feedin_case.loc[
                                    _] == "load_case"]
        time_steps_issues_fc = [_ for _ in timesteps_load_feedin_case.index
                                if timesteps_load_feedin_case.loc[
                                    _] == "feedin_case"]

        # run MPOPF
        if len(time_steps_issues_lc) > 0:
            elia_logger.debug(
                "Running OPF (load case) for {} time steps.".format(
                    len(time_steps_issues_lc)))
            edisgo.perform_mp_opf(
                timesteps=time_steps_issues_lc,
                scenario='curtailment_lc',
                results_path=grid_results_dir,
                load_case=True,
                max_exp=1
            )
            # check NEP
            nep_factor = edisgo.opf_results.lines.nep.values.astype("float")
            if nep_factor.max().max() > (1+1e-3):
                elia_logger.error(
                    "Network expansion factor greater 1 detected.")
            # get curtailed feed-in
            curtailment_per_node = opf_expand_network.get_curtailment_per_node(
                edisgo)
            total_feedin_curtailment = curtailment_per_node.sum().sum()
            curtailment.at["mv_problems", "feed-in"] += total_feedin_curtailment
            if total_feedin_curtailment > 0:
                opf_expand_network.integrate_curtailment_as_load(
                    edisgo, curtailment_per_node)
            # get curtailed load
            load_curtailment_per_node = opf_expand_network.get_load_curtailment_per_node(
                edisgo)
            total_load_curtailment = load_curtailment_per_node.sum().sum()
            curtailment.at["mv_problems", "load"] += total_load_curtailment
            if total_load_curtailment > 0:
                opf_expand_network.integrate_curtailment_as_load(
                    edisgo, -load_curtailment_per_node)

            # check if there are any line overloadings and voltage issues left
            edisgo.analyze(mode="mvlv", timesteps=time_steps_issues_lc)

            rel_load = results_helper_functions.relative_load(edisgo)
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
                voltage_dev, rel_load, mv_grid_id)

            elia_logger.debug(
                "Remaining number of time steps with MV issues after OPF: {}".format(
                    len(time_steps_issues)))
            if len(lines_issues) > 0:
                curtailment.at["mv_problems", "feed-in"], \
                curtailment.at["mv_problems", "load"] = [np.nan, np.nan]
                return curtailment

        if len(time_steps_issues_fc) > 0:
            edisgo.perform_mp_opf(
                timesteps=time_steps_issues_fc,
                scenario='curtailment_fc',
                results_path=grid_results_dir,
                max_exp=1
            )
            # check NEP
            nep_factor = edisgo.opf_results.lines.nep.values.astype("float")
            if nep_factor.max().max() > (1 + 1e-3):
                elia_logger.error(
                    "Network expansion factor greater 1 detected.")
            # get curtailed feed-in
            curtailment_per_node = opf_expand_network.get_curtailment_per_node(
                edisgo)
            total_feedin_curtailment = curtailment_per_node.sum().sum()
            curtailment.at[
                "mv_problems", "feed-in"] += total_feedin_curtailment
            if total_feedin_curtailment > 0:
                opf_expand_network.integrate_curtailment_as_load(
                    edisgo, curtailment_per_node)
            # get curtailed load
            load_curtailment_per_node = opf_expand_network.get_load_curtailment_per_node(
                edisgo)
            total_load_curtailment = load_curtailment_per_node.sum().sum()
            curtailment.at["mv_problems", "load"] += total_load_curtailment
            if total_load_curtailment > 0:
                opf_expand_network.integrate_curtailment_as_load(
                    edisgo, -load_curtailment_per_node)

            # check if there are any line overloadings and voltage issues left
            edisgo.analyze(mode="mvlv", timesteps=time_steps_issues_fc)

            rel_load = results_helper_functions.relative_load(edisgo)
            voltage_dev = results_helper_functions.voltage_diff(edisgo)
            lines_issues, buses_issues, time_steps_issues = get_mv_lines_and_timesteps_with_issues(
                voltage_dev, rel_load, mv_grid_id)

            elia_logger.debug(
                "Remaining number of time steps with MV issues after OPF: {}".format(
                    len(time_steps_issues)))

            if len(lines_issues) > 0:
                curtailment.at["mv_problems", "feed-in"], \
                curtailment.at["mv_problems", "load"] = [np.nan, np.nan]
                return curtailment

    return curtailment


def calculate_curtailment_mv_lines(
        edisgo, voltage_dev, rel_load, curtailment, grid_results_dir):

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
        assign_mv_feeder_to_buses_and_lines(edisgo)
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
                os.path.join(grid_results_dir, "edisgo_curtailment_mv"),
                parameters="powerflow_results")
            rel_load.to_csv(
                os.path.join(grid_results_dir, "edisgo_curtailment_mv",
                             "relative_load.csv"))
            voltage_dev.to_csv(
                os.path.join(grid_results_dir, "edisgo_curtailment_mv",
                             "voltage_deviation.csv"))
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
        edisgo, curtailment, voltage_dev, grid_results_dir, tolerance=2e-3):

    mv_grid_id = edisgo.topology.id
    elia_logger = elia_logger = logging.getLogger(
        'elia_project: {}'.format(mv_grid_id))

    # assign main MV feeder
    assign_mv_feeder_to_buses_and_lines(edisgo)

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
                os.path.join(grid_results_dir, "edisgo_curtailment_mv_voltage"),
                parameters="powerflow_results")
            rel_load.to_csv(
                os.path.join(grid_results_dir, "edisgo_curtailment_mv_voltage",
                             "relative_load.csv"))
            voltage_dev.to_csv(
                os.path.join(grid_results_dir, "edisgo_curtailment_mv_voltage",
                             "voltage_deviation.csv"))
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


def calculate_curtailment(mv_grid_id):

    try:
        elia_logger = logging.getLogger('elia_project: {}'.format(mv_grid_id))
        elia_logger.setLevel(logging.DEBUG)

        grid_results_dir = os.path.join(
            results_path, str(mv_grid_id))
        reload_dir = os.path.join(
            results_path, str(mv_grid_id))

        # reimport edisgo object
        edisgo = import_edisgo_from_files(
            reload_dir,
            import_timeseries=True,
            import_results=True,
            parameters="powerflow_results",
            import_residual_load=False
        )

        feedin_ts = edisgo.timeseries.generators_active_power.copy()
        load_ts = edisgo.timeseries.loads_active_power.copy()
        charging_ts = edisgo.timeseries.charging_points_active_power.copy()

        # import relative load and voltage deviation for all 8760 time steps
        path = os.path.join(reload_dir, 'relative_load.csv')
        rel_load = pd.read_csv(path, index_col=0, parse_dates=True)
        path = os.path.join(reload_dir, 'voltage_deviation.csv')
        voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)

        curtailment = pd.DataFrame(
            data=0,
            columns=["feed-in", "load"],
            index=["lv_problems", "mv_problems"])

        curtailment = calculate_curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, grid_results_dir)

        curtailment = calculate_curtailment_mvlv_stations(
            edisgo, voltage_dev, rel_load, curtailment, max_iterations,
            grid_results_dir)

        curtailment.to_csv(
            os.path.join(grid_results_dir, "curtailment_lv_problems.csv"))

        curtailment = calculate_curtailment_mv_lines(
            edisgo, voltage_dev, rel_load, curtailment, grid_results_dir)

        curtailment.to_csv(
            os.path.join(grid_results_dir, "curtailment.csv"))

        # save time series
        curtailed_feedin = feedin_ts - edisgo.timeseries.generators_active_power
        curtailed_load = pd.concat(
            [(load_ts - edisgo.timeseries.loads_active_power),
             (charging_ts - edisgo.timeseries.charging_points_active_power)],
            axis=1)
        curtailed_feedin.to_csv(
            os.path.join(grid_results_dir, "curtailment_ts_per_gen.csv")
        )
        curtailed_load.to_csv(
            os.path.join(grid_results_dir, "curtailment_ts_per_load.csv")
        )
        curtailed_feedin.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "curtailment_ts_feedin.csv")
        )
        curtailed_load.sum(axis=1).to_csv(
            os.path.join(grid_results_dir, "curtailment_ts_demand.csv")
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
            os.path.join(grid_results_dir, "curtailment_lv.csv")
        )

    except Exception as e:
        print("Error in MV grid {}.".format(mv_grid_id))
        traceback.print_exc()


def integrate_public_charging(
        ding0_dir,
        grid_dir,
        grid_id,
        files,
        generator_scenario="ego100",
):
    try:
        len_timeindex = 24

        timeindex = pd.date_range(
            '2011-01-01',
            periods=len_timeindex,
            freq='H',
        )

        p_bio = 9983  # MW
        e_bio = 50009  # GWh

        vls_bio = e_bio / (p_bio / 1000)

        share = vls_bio / 8760

        timeseries_generation_dispatchable = pd.DataFrame(
            {
                "biomass": [share] * len_timeindex,
                "coal": [1] * len_timeindex,
                "other": [1] * len_timeindex,
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

        timeindex = pd.date_range(
            "2011-01-01",
            periods=len_timeindex*4,
            freq="15min",
        )

        edisgo.timeseries.timeindex = timeindex

        edisgo.timeseries.generators_active_power = edisgo.timeseries.generators_active_power.ffill()
        edisgo.timeseries.generators_reactive_power = edisgo.timeseries.generators_reactive_power.ffill()
        edisgo.timeseries.loads_active_power = edisgo.timeseries.loads_active_power.ffill()
        edisgo.timeseries.loads_reactive_power = edisgo.timeseries.loads_reactive_power.ffill()

        comp_type = "ChargingPoint"

        ts_reactive_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_reactive_power",
        )

        geo_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("geojson" in f and ("public" in f or "hpc" in f))
        ]

        ts_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("h5" in f and ("public" in f or "hpc" in f))
        ]

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

            df = df.iloc[:len(timeindex)]

            df.index = timeindex

            _ = [
                EDisGo.integrate_component(
                    edisgo,
                    comp_type=comp_type,
                    geolocation=geolocation,
                    use_case=use_case,
                    voltage_level=None, # TODO: @Birgit
                    mode="mv", # TODO: @Birgit
                    add_ts=True,
                    ts_active_power=df.loc[:, (ags, cp_idx)].divide(cp_connection_rating),
                    ts_reactive_power=ts_reactive_power,
                    p_nom=p_nom,
                ) for ags, cp_idx, cp_connection_rating, geolocation, p_nom in list(
                    zip(
                        gdf.ags.tolist(),
                        gdf.cp_idx.tolist(),
                        gdf.cp_connection_rating.tolist(),
                        gdf.geometry.tolist(),
                        gdf.cp_connection_rating.divide(1000).tolist(),  # kW -> MW
                    )
                )
            ]

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
            if ("geojson" in f and ("home" in f or "work" in f))
        ]

        ts_files = [
            Path(os.path.join(grid_dir, f)) for f in files
            if ("h5" in f and strategy in f and ("home" in f or "work" in f))
        ]

        use_cases = ["home", "work"]

        timeindex = edisgo.timeseries.timeindex

        comp_type = "ChargingPoint"

        ts_reactive_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_reactive_power",
        )

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

            df = df.iloc[:len(timeindex)]

            df.index = timeindex

            _ = [
                EDisGo.integrate_component(
                    edisgo,
                    comp_type=comp_type,
                    geolocation=geolocation,
                    use_case=use_case,
                    voltage_level=None,
                    add_ts=True,
                    ts_active_power=df.loc[:, (ags, cp_idx)].divide(cp_connection_rating),
                    ts_reactive_power=ts_reactive_power,
                    p_nom=p_nom,
                ) for ags, cp_idx, cp_connection_rating, geolocation, p_nom in list(
                    zip(
                        gdf.ags.tolist(),
                        gdf.cp_idx.tolist(),
                        gdf.cp_connection_rating.tolist(),
                        gdf.geometry.tolist(),
                        gdf.cp_connection_rating.divide(1000).tolist(),  # kW -> MW
                    )
                )
            ]

        return edisgo

    except:
        traceback.print_exc()


if __name__ == "__main__":
    if num_threads == 1:
        for mv_grid_id in mv_grid_ids:
            calculate_curtailment(mv_grid_id)
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(calculate_curtailment, mv_grid_ids)
