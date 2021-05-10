# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, update_model, optimize, check_mapping
import pandas as pd
import geopandas as gpd
import numpy as np

result_dir = r'\\192.168.10.221\Daten_flexibel_02\Anyas_Daten\EV_Optimisation\1056'

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 1056

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_final\Electrification_2050\{}\reduced'.format(grid_id)#Todo: change back to final in the end
edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

print('eDisGo object imported.')

# downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj)
# downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix.csv')
downstream_nodes_matrix = pd.read_csv('grid_data/downstream_node_matrix.csv',
                                      index_col=0)
downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)

flexibility_bands_home = \
    pd.read_csv('grid_data/ev_flexibility_bands_home.csv', index_col=0,
                dtype=np.float16)
flexibility_bands_work = \
    pd.read_csv('grid_data/ev_flexibility_bands_work.csv', index_col=0,
                dtype=np.float16)
flexibility_bands = pd.concat([flexibility_bands_work, flexibility_bands_home],
                              axis=1)
print('Flexibility bands imported.')
cp_mapping_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\eDisGo_charging_time_series\{}'.format(grid_id)
mapping_home = \
    gpd.read_file(cp_mapping_dir + '\cp_data_home_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file(cp_mapping_dir + '\cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')

# mapping_hpc = \
#     gpd.read_file(cp_mapping_dir + '\cp_data_hpc_within_grid_{}.geojson'.
#                   format(grid_id)).set_index('edisgo_id')
# mapping_public = \
#     gpd.read_file(cp_mapping_dir + '\cp_data_public_within_grid_{}.geojson'.
#                   format(grid_id)).set_index('edisgo_id')

mapping = pd.concat([mapping_work, mapping_home])#, mapping_hpc, mapping_public
print('Mapping imported.')

check_mapping(mapping, edisgo_obj, flexibility_bands)
print('Data checked. Please pay attention to warnings.')

timesteps_per_week = 672
for week in range(int(len(edisgo_obj.timeseries.timeindex)/timesteps_per_week)-1):#edisgo_obj.timeseries.timeindex.week.unique()
    timesteps = edisgo_obj.timeseries.timeindex[week*timesteps_per_week:(week+1)*timesteps_per_week]
    # timesteps = edisgo_obj.timeseries.timeindex[
    #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
    flexibility_bands = flexibility_bands.iloc[:len(timesteps)].set_index(timesteps)
    if week == 0:
        model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps,
                            optimize_storage=False, mapping_cp=mapping,
                            energy_band_charging_points=flexibility_bands)
    else:
        model = update_model(model, timesteps, flexibility_bands)
    x_charge, soc, x_charge_ev, energy_band_cp = optimize(model, 'glpk')
    x_charge.astype(np.float16).to_csv(
        result_dir+'/x_charge_{}.csv'.format(week))
    soc.astype(np.float16).to_csv(result_dir + '/soc_{}.csv'.format(week))
    x_charge_ev.astype(np.float16).to_csv(
        result_dir + '/x_charge_ev_{}.csv'.format(week))
    energy_band_cp.astype(np.float16).to_csv(
        result_dir + '/energy_band_cp_{}.csv'.format(week))
