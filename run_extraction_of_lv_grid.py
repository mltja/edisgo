from edisgo import EDisGo
import numpy as np
import pandas as pd
import edisgo.flex_opt.optimization as opt
from edisgo.edisgo import import_edisgo_from_files
import geopandas as gpd
from edisgo.tools.tools import get_aggregated_bands

from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative


result_dir = r'\\192.168.10.221\Daten_flexibel_02\Anyas_Daten\EV_Optimisation\1056'

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 1056

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_final\Electrification_2050\{}\reduced'.format(grid_id)#Todo: change back to final in the end
edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True,
                                  import_results=True)

#edisgo.convert_to_pu_system()

print('eDisGo object imported.')

# downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj)
# downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix.csv', dtype=uint8)
downstream_nodes_matrix = pd.read_csv('grid_data/downstream_node_matrix.csv',
                                      index_col=0)
downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
print('Downstream node matrix imported.')
flexibility_bands_home = \
    pd.read_csv('grid_data/ev_flexibility_bands_home.csv', index_col=0,
                dtype=np.float16)
flexibility_bands_work = \
    pd.read_csv('grid_data/ev_flexibility_bands_work.csv', index_col=0,
                dtype=np.float16)
flexibility_bands = pd.concat([flexibility_bands_work, flexibility_bands_home],
                              axis=1)
flexibility_bands = \
    flexibility_bands.groupby(flexibility_bands.columns, axis=1).sum()
print('Flexibility bands imported.')
mapping_home = \
    gpd.read_file('grid_data/cp_data_home_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file('grid_data/cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')

mapping = pd.concat([mapping_work, mapping_home], sort=False)
print('Mapping imported.')

# check_mapping(mapping, edisgo_obj, flexibility_bands) TOdo: integrate
# print('Data checked. Please pay attention to warnings.')

lv_grids = [grid for grid in edisgo.topology.mv_grid.lv_grids]

lv_grid = lv_grids[0]
#timeseries = lv_grid.convert_to_pu_system()
downstream_node_matrix = downstream_nodes_matrix.loc[lv_grid.buses_df.index,
                                                     lv_grid.buses_df.index]
cp_lv_grid = lv_grid.charging_points_df.loc[
    lv_grid.charging_points_df.use_case == 'home'].index.append(
    lv_grid.charging_points_df.loc[
        lv_grid.charging_points_df.use_case == 'work'].index)
mapping_lv_grid = mapping.loc[cp_lv_grid]
cp_band_id = \
    ('upper_' + mapping_lv_grid.ags.astype(str) +
     '_'+mapping_lv_grid.cp_idx.astype(str)).append(
        'lower_' + mapping_lv_grid.ags.astype(str)+'_' +
        mapping_lv_grid.cp_idx.astype(str)).append(
        'power_' + mapping_lv_grid.ags.astype(str)+'_' +
        mapping_lv_grid.cp_idx.astype(str)).values
bands_lv_grid = flexibility_bands.loc[
                :, flexibility_bands.columns.isin(cp_band_id)]

timeindex=edisgo.timeseries.timeindex[0:96]
tmp_bands_lv_grid = flexibility_bands.iloc[:len(timeindex)].set_index(timeindex)
model = opt.setup_model(lv_grid, downstream_node_matrix, timesteps=timeindex,
                        optimize_storage=False, mapping_cp=mapping_lv_grid,
                        energy_band_charging_points=tmp_bands_lv_grid, pu=False)
x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
   curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
   v_bus, p_line, q_line = opt.optimize(model, 'glpk')

pypsa_network = edisgo.to_pypsa(mode='lv',
                                lv_grid_name=repr(lv_grid))
pypsa_network.pf()
voltage = pypsa_network.buses_t.v_mag_pu
# compare results
diff = voltage-v_bus/lv_grid.nominal_voltage
max_diff = diff.abs().max().max()
print('Maximum deviation voltage: {}%'.format(max_diff*100))
print('SUCCESS')
