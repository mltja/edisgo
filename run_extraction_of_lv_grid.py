import numpy as np
import pandas as pd
import edisgo.flex_opt.optimization as opt
from edisgo.edisgo import import_edisgo_from_files
import geopandas as gpd
import os

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 1056

edisgo_dir = r'U:\Software\{}'.format(grid_id)
edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

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
flexibility_bands = flexibility_bands.iloc[0:-1].append(
        flexibility_bands.iloc[0:-1]).set_index(edisgo.timeseries.timeindex)
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

for lv_grid in lv_grids[0:10]:
    if os.path.exists(os.path.join(edisgo_dir, repr(lv_grid))):
        print('{} already solved, skipping'.format(repr(lv_grid)))
        pass
    print('Starting evaluation for {}.'.format(repr(lv_grid)))
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
    downstream_node_matrix = downstream_nodes_matrix.loc[lv_grid.buses_df.index,
                                                         lv_grid.buses_df.index]
    model = opt.setup_model(lv_grid, downstream_node_matrix,
                            optimize_storage=False, optimize_ev_charging=True,
                            mapping_cp=mapping_lv_grid, 
                            energy_band_charging_points=tmp_bands_lv_grid, pu=False)
    try:
        x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
           curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
           v_bus, p_line, q_line = opt.optimize(model, 'gurobi')
           
        os.makedirs(os.path.join(edisgo_dir, repr(lv_grid)), exist_ok=True)
           
        v_bus.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'bus_voltages.csv'))
        p_line.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'line_active_power.csv'))
        q_line.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'line_reactive_power.csv'))
        curtailment_feedin.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'curtailment_feedin.csv'))
        curtailment_load.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'curtailment_load.csv'))
        curtailment_reactive_feedin.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'curtailment_reactive_feedin.csv'))
        curtailment_reactive_load.to_csv(os.path.join(edisgo_dir, repr(lv_grid), 'curtailment_reactive_load.csv'))
    except:
        print('ERROR: {} could not be solved'.format(repr(lv_grid)))

print('SUCCESS')
