import numpy as np
import pandas as pd
import edisgo.flex_opt.optimization as opt
from edisgo.edisgo import import_edisgo_from_files
import geopandas as gpd
from edisgo.tools.tools import get_aggregated_bands

grid_id = 1056

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_final\Electrification_2050\{}\reduced'.format(grid_id)
edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True,
                                  import_results=True)

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

mapping_home = \
    gpd.read_file('grid_data/cp_data_home_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file('grid_data/cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')

mapping = pd.concat([mapping_work, mapping_home], sort=False)

lv_grids = [grid for grid in edisgo.topology.mv_grid.lv_grids]

weeks = {0: 'summer', 1: 'winter'}

for lv_grid in lv_grids:

    downstream_node_matrix = downstream_nodes_matrix.loc[
        lv_grid.buses_df.index,
        lv_grid.buses_df.index]
    cp_lv_grid = lv_grid.charging_points_df.loc[
        lv_grid.charging_points_df.use_case == 'home'].index.append(
        lv_grid.charging_points_df.loc[
            lv_grid.charging_points_df.use_case == 'work'].index)
    mapping_lv_grid = mapping.loc[cp_lv_grid]
    cp_band_id = \
        ('upper_' + mapping_lv_grid.ags.astype(str) +
         '_' + mapping_lv_grid.cp_idx.astype(str)).append(
            'lower_' + mapping_lv_grid.ags.astype(str) + '_' +
            mapping_lv_grid.cp_idx.astype(str)).append(
            'power_' + mapping_lv_grid.ags.astype(str) + '_' +
            mapping_lv_grid.cp_idx.astype(str)).values
    bands_lv_grid = flexibility_bands.loc[
                    :, flexibility_bands.columns.isin(cp_band_id)]

    timesteps_per_week = 672
    for week in range(int(len(
            edisgo.timeseries.timeindex) / timesteps_per_week)):
        print('Starting optimisation for {} week.'.format(weeks[week]))
        timeindex = edisgo.timeseries.timeindex[
                    week * timesteps_per_week:(week + 1) * timesteps_per_week]
        tmp_bands_lv_grid = bands_lv_grid.iloc[:len(timeindex)].set_index(timeindex)

        slack_voltages = edisgo.results.v_res[lv_grid.transformers_df.bus1.iloc[0]]

        model_min = opt.setup_model(lv_grid, downstream_node_matrix,
                                    timesteps=timeindex,
                                    optimize_storage=False, mapping_cp=mapping_lv_grid,
                                    energy_band_charging_points=tmp_bands_lv_grid,
                                    pu=False, v_slack=slack_voltages,
                                    objective='minimize_energy_level')

        x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
        curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
        v_bus, p_line, q_line, p_agr_min = opt.optimize(model_min, 'glpk',
                                                        mode='energy_band')

        model_max = opt.setup_model(lv_grid, downstream_node_matrix,
                                    timesteps=timeindex,
                                    optimize_storage=False,
                                    mapping_cp=mapping_lv_grid,
                                    energy_band_charging_points=tmp_bands_lv_grid,
                                    pu=False, v_slack=slack_voltages,
                                    objective='maximize_energy_level')

        x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
        curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
        v_bus, p_line, q_line, p_agr_max = opt.optimize(model_max, 'glpk',
                                                        mode='energy_band')

        power_bands = pd.concat(
            [p_agr_min.rename(columns={repr(lv_grid): 'lower'}),
             p_agr_max.rename(columns={repr(lv_grid): 'upper'})], axis=1)
        energy_bands = get_aggregated_bands(tmp_bands_lv_grid)

        mode = "minimize"
        model_min_eb = opt.setup_model_bands(energy_bands, power_bands, mode=mode)
        energy_level_min, charging = opt.optimize_bands(model_min_eb, 'glpk', mode)

        mode = "maximize"
        model_max_eb = opt.setup_model_bands(energy_bands, power_bands, mode=mode)
        energy_level_max, charging_max = opt.optimize_bands(model_max_eb, 'glpk',
                                                            mode)

        energy_band_grid = pd.concat([energy_level_min, energy_level_max], axis=1)
        power_bands.to_csv('grid_data/aggregated_bands/{}_{}_power.csv'.format(
            repr(lv_grid), weeks[week]))
        energy_band_grid.to_csv('grid_data/aggregated_bands/{}_{}_energy.csv'.format(
            repr(lv_grid), weeks[week]))
        energy_bands.to_csv('grid_data/aggregated_bands/{}_{}_bands_unconstrained.csv'.format(
            repr(lv_grid), weeks[week]))

print('SUCCESS')