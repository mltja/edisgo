# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping
from edisgo.tools.tools import convert_impedances_to_mv
from edisgo.io.ding0_import import convert_pypsa_to_edisgo_tmp
import pandas as pd
import geopandas as gpd
import numpy as np
from copy import deepcopy

result_dir = r'U:\Software\1056'

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 1056

edisgo_dir = r'U:\Software\{}'.format(grid_id)#Todo: change back to final in the end
edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

print('eDisGo object imported.')

edisgo_obj = convert_impedances_to_mv(edisgo_orig)

# pypsa_network = edisgo_orig.to_pypsa(mode='mvlv', aggregate_loads='all',
#                                     aggregate_storage_units='all',
#                                     aggregate_generators='all')
# edisgo_obj = deepcopy(edisgo_orig)
# convert_pypsa_to_edisgo_tmp(edisgo_obj, pypsa_network, edisgo_dir+'/topology',
#                         convert_timeseries=True)
#
# print('Converted impedances to mv.')

# downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj)
# downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix.csv', dtype=uint8)
downstream_nodes_matrix = pd.read_csv('grid_data/downstream_node_matrix.csv',
                                      index_col=0)
downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
downstream_nodes_matrix = downstream_nodes_matrix.loc[
        edisgo_obj.topology.buses_df.index,
        edisgo_obj.topology.buses_df.index]

flexibility_bands_home = \
   pd.read_csv('grid_data/ev_flexibility_bands_home.csv', index_col=0,
               dtype=np.float32)
flexibility_bands_work = \
   pd.read_csv('grid_data/ev_flexibility_bands_work.csv', index_col=0,
               dtype=np.float32)
flexibility_bands = pd.concat([flexibility_bands_work, flexibility_bands_home],
                             axis=1)
flexibility_bands = \
   flexibility_bands.groupby(flexibility_bands.columns, axis=1).sum()
# remove numeric problems
flexibility_bands[
    flexibility_bands.columns[flexibility_bands.columns.str.contains('power')]] = \
    flexibility_bands[flexibility_bands.columns[
        flexibility_bands.columns.str.contains('power')]]+1e-6
print('Flexibility bands imported.')
mapping_home = \
   gpd.read_file('grid_data/cp_data_home_within_grid_{}.geojson'.
                 format(grid_id)).set_index('edisgo_id')
mapping_work = \
   gpd.read_file('grid_data/cp_data_work_within_grid_{}.geojson'.
                 format(grid_id)).set_index('edisgo_id')

# mapping_hpc = \
#     gpd.read_file(cp_mapping_dir + '\cp_data_hpc_within_grid_{}.geojson'.
#                   format(grid_id)).set_index('edisgo_id')
# mapping_public = \
#     gpd.read_file(cp_mapping_dir + '\cp_data_public_within_grid_{}.geojson'.
#                   format(grid_id)).set_index('edisgo_id')

mapping = pd.concat([mapping_work, mapping_home], sort=False)#, mapping_hpc, mapping_public
print('Mapping imported.')

#check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
print('Data checked. Please pay attention to warnings.')

timesteps_per_week = 24*4
for week in range(int(len(edisgo_obj.timeseries.timeindex)/timesteps_per_week)):#edisgo_obj.timeseries.timeindex.week.unique()
    print('Starting optimisation for week {}.'.format(week))
    timesteps = edisgo_obj.timeseries.timeindex[week*timesteps_per_week:(week+1)*timesteps_per_week]
    # timesteps = edisgo_obj.timeseries.timeindex[
    #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
    start_time = (week*timesteps_per_week)%672
    flexibility_bands_week = flexibility_bands.iloc[start_time:start_time+timesteps_per_week].set_index(timesteps)
    #if week == 0:
    model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective='peak_load',
                            optimize_storage=False, optimize_ev_charging=True,
                            mapping_cp=mapping,
                            energy_band_charging_points=flexibility_bands_week,
                            pu=False)
    print('Set up model for week {}.'.format(week))

    x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
    curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
    v_bus, p_line, q_line = optimize(model, 'gurobi')

    print('Finished optimisation for week {}.'.format(week))
    x_charge.astype(np.float16).to_csv(
        result_dir+'/x_charge_{}.csv'.format(week))
    soc.astype(np.float16).to_csv(result_dir + '/soc_{}.csv'.format(week))
    x_charge_ev.astype(np.float16).to_csv(
        result_dir + '/x_charge_ev_{}.csv'.format(week))
    energy_level_cp.astype(np.float16).to_csv(
        result_dir + '/energy_band_cp_{}.csv'.format(week))
    curtailment_feedin.astype(np.float16).to_csv(
        result_dir + '/curtailment_feedin_{}.csv'.format(week))
    curtailment_load.astype(np.float16).to_csv(
        result_dir + '/curtailment_load_{}.csv'.format(week))
    curtailment_reactive_feedin.astype(np.float16).to_csv(
        result_dir + '/curtailment_reactive_feedin_{}.csv'.format(week))
    curtailment_reactive_load.astype(np.float16).to_csv(
        result_dir + '/curtailment_reactive_load_{}.csv'.format(week))
    v_bus.astype(np.float16).to_csv(result_dir + '/bus_voltage_{}.csv'.format(week))
    p_line.astype(np.float16).to_csv(result_dir + '/line_active_power_{}.csv'.format(week))
    q_line.astype(np.float16).to_csv(result_dir + '/line_reactive_power_{}.csv'.format(week))
    print('Saved results for week {}.'.format(week))

print('SUCCESS')
