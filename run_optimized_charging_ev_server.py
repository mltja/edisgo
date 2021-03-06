# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping
from edisgo.tools.tools import convert_impedances_to_mv
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative
import edisgo.flex_opt.charging_ev as cEV
from edisgo.io.ding0_import import convert_pypsa_to_edisgo_tmp
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from copy import deepcopy

from pathlib import Path

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 2534
root_dir = r'U:\Software'
mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)
edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(grid_id)
result_dir = 'results/{}'.format(grid_id)
os.makedirs(result_dir, exist_ok=True)
data_dir = Path(root_dir + r'\simbev_nep_2035_results\cp_standing_times_mapping')

edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

print('eDisGo object imported.')

edisgo_obj = convert_impedances_to_mv(edisgo_orig)
# ONLY FOR DEBUGGING Todo: remove
# pypsa_network = edisgo_orig.to_pypsa(mode='mvlv', #aggregate_loads='all',
#                                     aggregate_storage_units='all',
#                                     aggregate_generators='all')
# edisgo_obj = deepcopy(edisgo_orig)
# convert_pypsa_to_edisgo_tmp(edisgo_obj, pypsa_network, edisgo_dir+'/topology',
#                         convert_timeseries=True)

print('Converted impedances to mv.')

try:
    downstream_nodes_matrix = pd.read_csv(
        'grid_data/downstream_node_matrix_{}.csv'.format(grid_id),
        index_col=0)
except:
    downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj.topology)
    #downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix_{}.csv'.format(grid_id))

downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
downstream_nodes_matrix = downstream_nodes_matrix.loc[
        edisgo_obj.topology.buses_df.index,
        edisgo_obj.topology.buses_df.index]
print('Downstream node matrix imported.')

flexibility_bands = pd.DataFrame()
for use_case in ['home', 'work']:
    try:
        flexibility_bands_tmp = \
            pd.read_csv('grid_data/ev_flexibility_bands_{}_{}.csv'.format(grid_id, use_case),
                        index_col=0, dtype=np.float32)
    except:
        flexibility_bands_tmp = cEV.get_energy_bands_for_optimization(data_dir, edisgo_dir, grid_id, use_case)
    rename_dict = {col: col + '_{}'.format(use_case) for col in
                   flexibility_bands_tmp.columns}
    flexibility_bands_tmp.rename(columns=rename_dict, inplace=True)
    flexibility_bands = pd.concat([flexibility_bands, flexibility_bands_tmp],
                                  axis=1)
# remove numeric problems
flexibility_bands.loc[:,
    flexibility_bands.columns[flexibility_bands.columns.str.contains('power')]] = \
    (flexibility_bands[flexibility_bands.columns[
        flexibility_bands.columns.str.contains('power')]]+1e-6).values
print('Flexibility bands imported.')
mapping_home = \
        gpd.read_file(mapping_dir + '/cp_data_home_within_grid_{}.geojson'.
                      format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file(mapping_dir + '/cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_home['use_case'] = 'home'
mapping_work['use_case'] = 'work'
mapping = pd.concat([mapping_work, mapping_home],
                    sort=False)  # , mapping_hpc, mapping_public
print('Mapping imported.')

check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
print('Data checked. Please pay attention to warnings.')

timesteps_per_iteration = 24*4
iterations_per_era = 7
charging_start = None
energy_level_start = None
overlap_interations = 2
energy_level = {}
charging_ev = {}
for iteration in range(
        int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)):  # edisgo_obj.timeseries.timeindex.week.unique()

    print('Starting optimisation for week {}.'.format(iteration))
    # timesteps = edisgo_obj.timeseries.timeindex[
    #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
    start_time = (iteration * timesteps_per_iteration) % 672
    if iteration % iterations_per_era != iterations_per_era - 1:
        timesteps = edisgo_obj.timeseries.timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration + overlap_interations]
        flexibility_bands_week = flexibility_bands.iloc[
                                 start_time:start_time + timesteps_per_iteration + overlap_interations].set_index(
            timesteps)
    else:
        timesteps = edisgo_obj.timeseries.timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
        flexibility_bands_week = flexibility_bands.iloc[start_time:start_time + timesteps_per_iteration].set_index(
            timesteps)
    # if week == 0:
    model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective='peak_load',
                        optimize_storage=False, optimize_ev_charging=True,
                        mapping_cp=mapping,
                        energy_band_charging_points=flexibility_bands_week,
                        charging_start=charging_start,
                        energy_level_start=energy_level_start)
    print('Set up model for week {}.'.format(iteration))

    x_charge, soc, charging_ev[iteration], energy_level[iteration], curtailment_feedin, \
    curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
    v_bus, p_line, q_line = optimize(model, 'gurobi')
    if iteration % iterations_per_era != iterations_per_era - 1:
        charging_start = charging_ev[iteration].iloc[-overlap_interations]
        energy_level_start = energy_level[iteration].iloc[-overlap_interations]
    else:
        charging_start = None
        energy_level_start = None

    print('Finished optimisation for week {}.'.format(iteration))

    x_charge.astype(np.float16).to_csv(
        result_dir+'/x_charge_{}.csv'.format(iteration))
    soc.astype(np.float16).to_csv(result_dir + '/soc_{}.csv'.format(iteration))
    charging_ev[iteration].astype(np.float16).to_csv(
        result_dir + '/x_charge_ev_{}.csv'.format(iteration))
    energy_level[iteration].astype(np.float16).to_csv(
        result_dir + '/energy_band_cp_{}.csv'.format(iteration))
    curtailment_feedin.astype(np.float16).to_csv(
        result_dir + '/curtailment_feedin_{}.csv'.format(iteration))
    curtailment_load.astype(np.float16).to_csv(
        result_dir + '/curtailment_load_{}.csv'.format(iteration))
    curtailment_reactive_feedin.astype(np.float16).to_csv(
        result_dir + '/curtailment_reactive_feedin_{}.csv'.format(iteration))
    curtailment_reactive_load.astype(np.float16).to_csv(
        result_dir + '/curtailment_reactive_load_{}.csv'.format(iteration))
    v_bus.astype(np.float16).to_csv(result_dir + '/bus_voltage_{}.csv'.format(iteration))
    p_line.astype(np.float16).to_csv(result_dir + '/line_active_power_{}.csv'.format(iteration))
    q_line.astype(np.float16).to_csv(result_dir + '/line_reactive_power_{}.csv'.format(iteration))
    print('Saved results for week {}.'.format(iteration))

print('SUCCESS')
