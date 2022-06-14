import copy
import logging
import os
import sys

from datetime import datetime
from time import time
import gc

import pandas as pd


from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.complexity_reduction import (setup_logger,
                                               make_busmap,
                                               reduce_edisgo,
                                               save_results_reduced_to_min_max)


def telegram_bot_sendtext(bot_message):
    return


# Configuration
file_time = datetime.now().isoformat(sep='T', timespec='seconds')


edisgo_object_path = os.path.join("/storage", "complexity_reduction", "standard", "data")
analyze_path = os.path.join("/storage", "complexity_reduction", "standard", "results", 'run_' + file_time)

try:
    os.mkdir(analyze_path)
except:
    print('Analyze dir is already there')

debug_mode = False
chunk_mode = False
time_step_mode = False

if len(sys.argv) == 1:
    print('No arguments passed')
    exit(1)
else:
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-d':
            debug_mode = True
            print('Debug mode')
        elif sys.argv[i] == '-c':
            chunk_mode = True
            chunk_number = int(sys.argv[i+1])
            print('Chunk mode, selected chunk: {}'.format(chunk_number))
        elif sys.argv[i] == '-t':
            time_step_mode = True
            print('Timestep mode')
        elif sys.argv[i] == '-w':
            week_mode = True
            print('Week mode')
        elif sys.argv[i] == '-g':
            chunk_mode = False
            chunk_number = 0
            grid_number_list = [int(sys.argv[i+1])]
            print('Grid mode, selected grid: {}'.format(grid_number_list[0]))

setup_logger(stream_level=None, file_level=logging.INFO, filename=os.path.join(analyze_path, 'edisgo_'+str(chunk_number)+'.log'))





if time_step_mode and chunk_mode:
    if chunk_number > 0:
        grid_number_list = [176, 177, 566, 1056, 1423, 1574, 1690, 1811, 1839, 2079, 2095, 2534, 3008, 3267, 3280]
        grid_number_list = [grid_number_list[chunk_number-1]]
    else:
        grid_number_list = [176, 177, 566, 1056, 1423, 1574, 1690, 1811, 1839, 2079, 2095, 2534, 3008, 3267, 3280]
elif chunk_mode:
    if chunk_number == 1:
        grid_number_list = [176]
    elif chunk_number == 2:
        grid_number_list = [3267, 177, 3280]
    elif chunk_number == 3:
        grid_number_list = [1811, 1574, 2534]
    elif chunk_number == 4:
        grid_number_list = [2079, 3008, 566,1839]
    elif chunk_number == 5:
        grid_number_list = [1690, 1056, 1423,2095]
    else:
        grid_number_list = [176, 177, 566, 1056, 1423, 1574, 1690, 1811, 1839, 2079, 2095, 2534, 3008, 3267, 3280]


km = []
kmd = []
atmf = []
frwen = []
kmpfwf01 = []
kmpf = []
kmdpf = []
kmpfwf0 = []
kmpfwf01 = []
kmdpfwf0 = []
kmdpfwf01 = []
kmpmf = []
kmdpmf = []
kmpmfwf0 = []
kmpmfwf01 = []
kmdpmfwf0 = []
kmdpmfwf01 = []



rf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


km = rf_list
kmd = rf_list
atmf = [1]
frwen = rf_list
frwenwf01 = rf_list
kmpf = rf_list
kmdpf = rf_list
kmpfwf01 = rf_list
kmdpfwf01 = rf_list
kmpmf = rf_list
kmdpmf = rf_list
kmpmfwf01 = rf_list
kmdpmfwf01 = rf_list

runs = list()
for run in km:
    runs.append('km_' + str(run))
for run in kmd:
    runs.append('kmd_' + str(run))
for run in atmf:
    runs.append('atmf_' + str(run))
for run in frwen:
    runs.append('frwen_' + str(run))
for run in frwenwf01:
    runs.append('frwenwf0.1_' + str(run))
for run in kmpmf:
    runs.append('kmpmf_' + str(run))
for run in kmdpmf:
    runs.append('kmdpmf_' + str(run))
for run in kmpmfwf0:
    runs.append('kmpmfwf0_' + str(run))
for run in kmpmfwf01:
    runs.append('kmpmfwf0.1_' + str(run))
for run in kmdpmfwf0:
    runs.append('kmdpmfwf0_' + str(run))
for run in kmdpmfwf01:
    runs.append('kmdpmfwf0.1_' + str(run))
for run in kmpf:
    runs.append('kmpf_' + str(run))
for run in kmdpf:
    runs.append('kmdpf_' + str(run))
for run in kmpfwf0:
    runs.append('kmpfwf0_' + str(run))
for run in kmpfwf01:
    runs.append('kmpfwf0.1_' + str(run))
for run in kmdpfwf0:
    runs.append('kmdpfwf0_' + str(run))
for run in kmdpfwf01:
    runs.append('kmdpfwf0.1_' + str(run))


edisgo_object_list = list()
dirs = os.listdir(edisgo_object_path)
print(dirs)
for name in dirs:
    if int(name.split('_')[0]) in grid_number_list:
        edisgo_object_list.append(name)
edisgo_object_list.sort()

# Print configuration
logging.info('Configuration')
logging.info('Edisgo object path: {}'.format(edisgo_object_path))
logging.info('Results path: {}'.format(analyze_path))
logging.info('runs = {}'.format(runs))
logging.info('Analyze {} edisgo objects'.format(len(edisgo_object_list)))
logging.info('Edisgo object list: {}'.format(edisgo_object_list))


# Analyze
try:
    telegram_bot_sendtext('START! CHUNK {}'.format(chunk_number))
    for edisgo_object in edisgo_object_list:
        print('Run - {}'.format(edisgo_object))
        analysis_df = pd.DataFrame()
        analysis_df.index.name = 'run'
        analysis_df_name = os.path.join(analyze_path, 'analysis_df_' + edisgo_object + '.csv')

        try:
            print('root')

            edisgo_root = import_edisgo_from_files(directory=os.path.join(edisgo_object_path, edisgo_object), import_topology=True, import_timeseries=True)
            edisgo_root.results.equipment_changes = pd.DataFrame()

            if time_step_mode:
                timesteps = pd.DatetimeIndex(data=['2011-01-01 00:00:00', '2011-04-01 18:00:00', '2011-07-01 12:00:00', '2011-09-01 10:00:00']).to_list()
            elif week_mode:
                timeindex = edisgo_root.timeseries.timeindex.copy()
                residual_load = edisgo_root.timeseries.residual_load.copy()
                residual_load = residual_load.resample("W", label="left", closed="left").mean()
                residual_load = residual_load.iloc[1:52]
                timesteps = [residual_load.idxmax(), residual_load.idxmin()]
                start_indexes = [list(timeindex).index(timestep) for timestep in timesteps]
                timesteps = pd.DatetimeIndex(data=[])
                for start_index in start_indexes:
                    timesteps = timesteps.append(edisgo_root.timeseries.timeindex[start_index: start_index + 7 * 24])

                timeindex = None
                residual_load = None
                gc.collect()
            else:
                timesteps = None

            run = 'root'
            analysis_df.loc[run, 'buses'] = edisgo_root.topology.buses_df.shape[0]

            # Analyze edisgo_root
            try:
                start_time_analyze = time()
                edisgo_root.analyze(timesteps=timesteps)
                end_time_analyze = time()

                analysis_df.loc[run, 'analyzable'] = True
                analysis_df.loc[run, 'time_analyze'] = end_time_analyze - start_time_analyze
                logging.info('Analyzeable')

                save_results_reduced_to_min_max(edisgo_root, os.path.join(analyze_path, edisgo_object + '_edisgo_root'))
                gc.collect()
            except:
                analysis_df.loc[run, 'analyzable'] = False
                logging.info('Not analyzeable')

                edisgo_root = None
                gc.collect()

            # Reinforce edisgo_root
            try:
                edisgo_root_reinforced = copy.deepcopy(edisgo_root)

                start_time_reinforce = time()
                edisgo_root_reinforced.reinforce(timesteps_pfa=timesteps)
                end_time_reinforce = time()

                analysis_df.loc[run, 'reinforceable'] = True
                analysis_df.loc[run, 'costs'] = edisgo_root_reinforced.results.grid_expansion_costs.loc[:, 'total_costs'].sum()
                analysis_df.loc[run, 'costs_mv'] = edisgo_root_reinforced.results.grid_expansion_costs.loc[edisgo_root_reinforced.results.grid_expansion_costs.voltage_level == 'mv'].total_costs.sum()
                analysis_df.loc[run, 'costs_mvlv'] = edisgo_root_reinforced.results.grid_expansion_costs.loc[edisgo_root_reinforced.results.grid_expansion_costs.voltage_level == 'mv/lv'].total_costs.sum()
                analysis_df.loc[run, 'costs_lv'] = edisgo_root_reinforced.results.grid_expansion_costs.loc[edisgo_root_reinforced.results.grid_expansion_costs.voltage_level == 'lv'].total_costs.sum()
                analysis_df.loc[run, 'time_reinforcing'] = end_time_reinforce - start_time_reinforce
                logging.info('Reinforcable')

                save_results_reduced_to_min_max(edisgo_root_reinforced, os.path.join(analyze_path, edisgo_object + '_edisgo_root_reinforced'))
                edisgo_root_reinforced = None
                gc.collect()
            except:
                analysis_df.loc[run, 'reinforceable'] = False
                logging.info('Not reinforceable')

                edisgo_root_reinforced = None
                gc.collect()

            analysis_df.to_csv(analysis_df_name)

            # Analyze reduction
            for run in runs:
                print(run)
                analysis_df = pd.read_csv(analysis_df_name, index_col='run')
                run_suffix = '_' + edisgo_object + '_' + run + '.csv'
                logging.info('Run: {}'.format(run))

                reduction_mode = run.split('_')[0]
                if len(run.split('_')) == 2:
                    reduction_factor = float(run.split('_')[1])
                else:
                    reduction_factor = None
                analysis_df.loc[run, 'reduction_mode'] = reduction_mode
                analysis_df.loc[run, 'reduction_factor'] = reduction_factor

                # Make busmap
                try:
                    start_time_make_busmap = time()
                    busmap_df = make_busmap(edisgo_root=edisgo_root,
                                            mode=reduction_mode,
                                            reduction_factor=reduction_factor)
                    end_time_make_busmap = time()

                    busmap_df.to_csv(os.path.join(analyze_path, 'busmap_df' + run_suffix))
                    analysis_df.loc[run, 'make_busmap'] = True
                    analysis_df.loc[run, 'time_make_busmap'] = end_time_make_busmap - start_time_make_busmap
                    logging.info('Make busmap successful')

                    gc.collect()
                except:
                    analysis_df.loc[run, 'make_busmap'] = False
                    logging.info('Make busmap not successful')

                    busmap_df = None
                    gc.collect()

                # Reduce edisgo-object
                try:
                    start_time_reduction = time()
                    edisgo_reduced, linemap_df = reduce_edisgo(edisgo_root, busmap_df)
                    end_time_reduction = time()

                    linemap_df.to_csv(os.path.join(analyze_path, 'linemap_df' + run_suffix))
                    analysis_df.loc[run, 'buses'] = edisgo_reduced.topology.buses_df.shape[0]
                    analysis_df.loc[run, 'reducable'] = True
                    analysis_df.loc[run, 'time_reducing'] = end_time_reduction - start_time_reduction
                    logging.info('Reducable')

                    linemap_df = None
                    busmap_df = None
                    gc.collect()
                except:
                    analysis_df.loc[run, 'reducable'] = False
                    logging.info('Not reducable')

                    edisgo_reduced = None
                    linemap_df = None
                    busmap_df = None
                    gc.collect()

                # Analyze reduced edisgo-object
                try:
                    start_time_analyze = time()
                    edisgo_reduced.analyze(timesteps=timesteps)
                    end_time_analyze = time()

                    analysis_df.loc[run, 'analyzable'] = True
                    analysis_df.loc[run, 'time_analyze'] = end_time_analyze - start_time_analyze
                    logging.info('Analyzable')

                    save_results_reduced_to_min_max(edisgo_reduced, os.path.join(analyze_path, edisgo_object + '_edisgo_reduced_' + run))
                    gc.collect()
                except:
                    analysis_df.loc[run, 'analyzable'] = False
                    logging.info('Not analyzable')

                    edisgo_reduced = None
                    gc.collect()

                # Reinforce reduced edisgo-object
                try:
                    edisgo_reduced_reinforced = edisgo_reduced

                    start_time_reinforce = time()
                    edisgo_reduced_reinforced.reinforce(timesteps_pfa=timesteps)
                    end_time_reinforce = time()

                    analysis_df.loc[run, 'reinforceable'] = True
                    analysis_df.loc[run, 'time_reinforcing'] = end_time_reinforce - start_time_reinforce
                    analysis_df.loc[run, 'costs'] = edisgo_reduced_reinforced.results.grid_expansion_costs.loc[:, 'total_costs'].sum()
                    analysis_df.loc[run, 'costs_mv'] = edisgo_reduced_reinforced.results.grid_expansion_costs.loc[edisgo_reduced_reinforced.results.grid_expansion_costs.voltage_level == 'mv'].total_costs.sum()
                    analysis_df.loc[run, 'costs_mvlv'] = edisgo_reduced_reinforced.results.grid_expansion_costs.loc[edisgo_reduced_reinforced.results.grid_expansion_costs.voltage_level == 'mv/lv'].total_costs.sum()
                    analysis_df.loc[run, 'costs_lv'] = edisgo_reduced_reinforced.results.grid_expansion_costs.loc[edisgo_reduced_reinforced.results.grid_expansion_costs.voltage_level == 'lv'].total_costs.sum()
                    logging.info('Reinforceable')
                    save_results_reduced_to_min_max(edisgo_reduced_reinforced, os.path.join(analyze_path, edisgo_object + '_edisgo_reduced_reinforced_' + run))

                    edisgo_reduced_reinforced = None
                    gc.collect()
                except:
                    analysis_df.loc[run, 'reinforceable'] = False
                    logging.info('Not reinforceable')

                    edisgo_reduced_reinforced = None
                    gc.collect()

                analysis_df.to_csv(analysis_df_name)
        except:
            logging.info('ERROR in {}'.format(edisgo_object))

    telegram_bot_sendtext('FINISH! CHUNK {}'.format(chunk_number))
except:
    telegram_bot_sendtext('ERROR! CHUNK {}'.format(chunk_number))
