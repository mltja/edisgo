import copy
import logging
import os
import sys

from datetime import datetime
from time import time
import gc

import pandas as pd

from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.complexity_reduction import setup_logger, make_busmap, reduce_edisgo
from edisgo.tools.complexity_reduction_opt import extract_feeders, optimize_feeders, extract_edisgo_object


def telegram_bot_sendtext(bot_message):

    return

# Configuration
file_time = datetime.now().isoformat(sep="T", timespec="seconds")

edisgo_object_path = os.path.join("/storage", "complexity_reduction", "flex", "data")
mapping_dir = os.path.join("/storage", "complexity_reduction", "flex", "data", "data_cp_ev")
analyze_path = os.path.join("/storage", "complexity_reduction", "flex", "results", "run_" + file_time)

try:
    os.mkdir(analyze_path)
except FileExistsError:
    print("Analyze dir is already there")

debug_mode = False
chunk_mode = False
time_step_mode = False

if len(sys.argv) == 1:
    print("No arguments passed")
    exit(1)
else:
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-d":
            debug_mode = True
            print("Debug mode")
        elif sys.argv[i] == "-c":
            chunk_mode = True
            chunk_number = int(sys.argv[i + 1])
            # analyze_path = analyze_path + '/' + str(chunk_number)
            print("Chunk mode, selected chunk: {}".format(chunk_number))
        elif sys.argv[i] == "-t":
            time_step_mode = True
            print("Timestep mode")
        elif sys.argv[i] == "-w":
            week_mode = True
            print("Week mode")
        elif sys.argv[i] == "-g":
            chunk_mode = False
            chunk_number = 0
            grid_number_list = [int(sys.argv[i + 1])]
            print("Grid mode, selected grid: {}".format(grid_number_list[0]))

setup_logger(stream_level=None, file_level=logging.INFO, filename=os.path.join(analyze_path, "edisgo_" + str(chunk_number) + ".log"), root=False)
logger = logging.getLogger("edisgo.script")


if chunk_mode:
    if chunk_number == 1:
        grid_number_list = [177]
    elif chunk_number == 2:
        grid_number_list = [1056, 1690]
    elif chunk_number == 3:
        grid_number_list = [1811, 2534]
    else:
        grid_number_list = [176, 177, 1056, 1690, 1811, 2534]




kmd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
atmf = [1]
frwen = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
frwenwf01 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kmdpf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kmdpmf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kmdpmfwf01 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



runs = list()

for run in atmf:
    runs.append("atmf_" + str(run))

for run in frwen:
    runs.append("frwen_" + str(run))

for run in frwenwf01:
    runs.append("frwenwf0.1_" + str(run))

for run in kmdpmf:
    runs.append("kmdpmf_" + str(run))

for run in kmdpmfwf01:
    runs.append("kmdpmfwf0.1_" + str(run))

for run in kmd:
    runs.append("kmd_" + str(run))

for run in kmdpf:
    runs.append("kmdpf_" + str(run))




edisgo_object_list = list()
dirs = os.listdir(edisgo_object_path)
print(dirs)
for name in dirs:
    try:
        if int(name.split("_")[0]) in grid_number_list:
            if name.split("_")[1] == "clean":
                edisgo_object_list.append(name)
    except ValueError:
        print("{} is not an int".format(name.split("_")[0]))
edisgo_object_list.sort()

# Print configuration
logger.info("Configuration")
logger.info("Edisgo object path: {}".format(edisgo_object_path))
logger.info("Results path: {}".format(analyze_path))
logger.info("runs = {}".format(runs))
logger.info("Analyze {} edisgo objects".format(len(edisgo_object_list)))
logger.info("Edisgo object list: {}".format(edisgo_object_list))


def analyze(run, edisgo_obj):
    logger.info("Analyze run: {}".format(run))
    edisgo = copy.deepcopy(edisgo_obj)
    analysis_df = pd.read_csv(analysis_df_name, index_col="run")

    try:
        analysis_df.loc[run, "buses"] = edisgo.topology.buses_df.shape[0]
    except:
        logger.error("Run: {}, Can not read number of buses: {}".format(run, sys.exc_info()))

    try:
        start_time = time()
        edisgo.analyze(timesteps=timesteps)
        end_time = time()

        analysis_df.loc[run, "analyzable"] = True
        analysis_df.loc[run, "time_analyze"] = end_time - start_time
        edisgo.save(os.path.join(analyze_path, edisgo_object + "_" + run + "_analyze"), reduce_memory=True)
    except:
        analysis_df.loc[run, "analyzable"] = False
        logger.error("Run: {}, Not analyzeable: {}".format(run, sys.exc_info()))

    analysis_df.to_csv(analysis_df_name)
    return


def reinforce(run, edisgo_obj):
    logger.info("Reinforce run: {}".format(run))
    analysis_df = pd.read_csv(analysis_df_name, index_col="run")
    edisgo = copy.deepcopy(edisgo_obj)

    try:
        start_time = time()
        edisgo.reinforce(timesteps_pfa=timesteps)
        end_time = time()

        analysis_df.loc[run, "reinforceable"] = True
        analysis_df.loc[run, "costs"] = edisgo.results.grid_expansion_costs.loc[:, "total_costs"].sum()
        analysis_df.loc[run, "costs_mv"] = edisgo.results.grid_expansion_costs.loc[edisgo.results.grid_expansion_costs.voltage_level == "mv"].total_costs.sum()
        analysis_df.loc[run, "costs_mvlv"] = edisgo.results.grid_expansion_costs.loc[edisgo.results.grid_expansion_costs.voltage_level == "mv/lv"].total_costs.sum()
        analysis_df.loc[run, "costs_lv"] = edisgo.results.grid_expansion_costs.loc[edisgo.results.grid_expansion_costs.voltage_level == "lv"].total_costs.sum()
        analysis_df.loc[run, "time_reinforcing"] = end_time - start_time

        edisgo.save(os.path.join(analyze_path, edisgo_object + "_" + run + "_reinforce"), save_results=True, save_timeseries=False, reduce_memory=True)
    except:
        analysis_df.loc[run, "reinforceable"] = False
        logger.error("Run: {}, Not reinforceable: {}".format(run, sys.exc_info()))

    analysis_df.to_csv(analysis_df_name)
    return


def optimize(run, edisgo_obj):
    logger.info("Optimize run: {}".format(run))
    analysis_df = pd.read_csv(analysis_df_name, index_col="run")
    edisgo = copy.deepcopy(edisgo_obj)

    try:
        edisgo_dir = os.path.join(analyze_path, edisgo_object + "_" + run)
        edisgo.save(edisgo_dir)

        start_time = time()
        extract_feeders(edisgo_dir)

        if time_step_mode:
            optimize_feeders(edisgo_dir, mapping_dir, multiprocessing=False, single_feeder="0")
        else:
            optimize_feeders(edisgo_dir, mapping_dir, multiprocessing=True)

        edisgo_opt = extract_edisgo_object(edisgo_dir, save=True)
        end_time = time()

        analysis_df.loc[run, "optimizable"] = True
        analysis_df.loc[run, "time_optimize"] = end_time - start_time
    except:
        analysis_df.loc[run, "optimizable"] = False
        logger.error("Run: {}, Not optimizable: {}".format(run, sys.exc_info()))
        edisgo_opt = None

    analysis_df.to_csv(analysis_df_name)
    return edisgo_opt


def reduce(run, edisgo_obj):
    logger.info("Reduce run: {}".format(run))
    edisgo = copy.deepcopy(edisgo_obj)
    analysis_df = pd.read_csv(analysis_df_name, index_col="run")
    output_path = os.path.join(analyze_path, edisgo_object + "_" + run)
    run_suffix = ".csv"  # "_" + edisgo_object + "_" + run + ".csv"

    reduction_mode = run.split("_")[0]
    if len(run.split("_")) >= 2:
        reduction_factor = float(run.split("_")[1])
    else:
        reduction_factor = None

    analysis_df.loc[run, "reduction_mode"] = reduction_mode
    analysis_df.loc[run, "reduction_factor"] = reduction_factor

    # Make busmap
    try:
        start_time_make_busmap = time()
        busmap_df = make_busmap(edisgo_root=edisgo, mode=reduction_mode, reduction_factor=reduction_factor)
        end_time_make_busmap = time()

        os.makedirs(output_path, exist_ok=True)
        busmap_df.to_csv(os.path.join(output_path, "busmap_df" + run_suffix))

        analysis_df.loc[run, "make_busmap"] = True
        analysis_df.loc[run, "time_make_busmap"] = end_time_make_busmap - start_time_make_busmap

    except:
        analysis_df.loc[run, "make_busmap"] = False
        logger.error("Run: {}, Make busmap not successful: {}".format(run, sys.exc_info()))
        busmap_df = None

    # Reduce edisgo-object
    try:
        start_time_reduction = time()
        edisgo_reduced, linemap_df = reduce_edisgo(edisgo, busmap_df, aggregate_charging_points_mode=False)
        end_time_reduction = time()

        linemap_df.to_csv(os.path.join(output_path, "linemap_df" + run_suffix))

        analysis_df.loc[run, "buses"] = edisgo_reduced.topology.buses_df.shape[0]
        analysis_df.loc[run, "reducable"] = True
        analysis_df.loc[run, "time_reducing"] = end_time_reduction - start_time_reduction
    except:
        analysis_df.loc[run, "reducable"] = False
        logger.error("Run: {}, Not reducable: {}".format(run, sys.exc_info()))
        edisgo_reduced = None

    analysis_df.to_csv(analysis_df_name)
    return edisgo_reduced


# WORK
try:
    logger.info("Start chunk {}".format(chunk_number))
    telegram_bot_sendtext("Start chunk {}".format(chunk_number))
    for edisgo_object in edisgo_object_list:
        print("Investigate: - {}".format(edisgo_object))
        try:
            logger.info("Start grid: {}".format(edisgo_object))
            run = "root"
            print(run)

            # load edisgo_root and reset equipment changes
            edisgo_root = import_edisgo_from_files(directory=os.path.join(edisgo_object_path, edisgo_object), import_topology=True, import_timeseries=True)
            edisgo_root.results.equipment_changes = pd.DataFrame()

            # make analysis_df and write to csv
            analysis_df = pd.DataFrame()
            analysis_df.index.name = "run"
            analysis_df_name = os.path.join(analyze_path, "analysis_df_" + edisgo_object + ".csv")
            analysis_df.to_csv(analysis_df_name)

            if time_step_mode:
                timesteps = pd.DatetimeIndex(data=[])
                timesteps = timesteps.append(edisgo_root.timeseries.timeindex[0:1])
                logger.info("Timesteps: {}".format(timesteps))
            else:
                timesteps = None

            detailed_run = run + "_dumb"
            #analyze(detailed_run, edisgo_root)
            reinforce(detailed_run, edisgo_root)

            edisgo_root_opt = optimize(run + "_base", edisgo_root)

            detailed_run = run + "_opt"

            analyze(detailed_run, edisgo_root_opt)
            reinforce(detailed_run, edisgo_root_opt)

            edisgo_root_opt = None
            gc.collect()

            for run in runs:
                logger.info("Start run: {} - {}".format(edisgo_object, run))
                try:
                    print(run)

                    edisgo_reduced = reduce(run + "_base", edisgo_root)

                    detailed_run = run + "_dumb"
                    analyze(detailed_run, edisgo_reduced)
                    reinforce(detailed_run, edisgo_reduced)

                    edisgo_reduced_opt = optimize(run + "_base", edisgo_reduced)

                    detailed_run = run + "_opt"
                    analyze(detailed_run, edisgo_reduced_opt)
                    reinforce(detailed_run, edisgo_reduced_opt)

                    edisgo_reduced = None
                    edisgo_reduced_opt = None
                    gc.collect()

                    logger.info("Finish run: {} - {}".format(edisgo_object, run))
                except:
                    logger.info("Error run: {} - {} - {}".format(edisgo_object, run, sys.exc_info()))

            logger.info("Finish grid: {}".format(edisgo_object))
        except:
            logger.info("Error grid: {} - {}".format(edisgo_object, sys.exc_info()))

    logger.info("Finish chunk {}".format(chunk_number))
    telegram_bot_sendtext("Finish chunk {}".format(chunk_number))
except:
    logger.info("Abort chunk {}: {}".format(chunk_number, sys.exc_info()))
    telegram_bot_sendtext("Abort chunk {}: {}".format(chunk_number, sys.exc_info()))
