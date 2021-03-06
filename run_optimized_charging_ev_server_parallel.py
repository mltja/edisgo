# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping
from edisgo.tools.tools import convert_impedances_to_mv
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import multiprocessing as mp

from pathlib import Path

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)


def run_optimized_charging_parallel(grid_id, iteration):
    root_dir = r'U:\Software'
    mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)
    edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(grid_id)
    result_dir = 'results/{}'.format(grid_id)
    os.makedirs(result_dir, exist_ok=True)

    edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    print('eDisGo object imported.')

    edisgo_obj = convert_impedances_to_mv(edisgo_orig)

    downstream_nodes_matrix = pd.read_csv(
            'grid_data/downstream_node_matrix_{}.csv'.format(grid_id),
            index_col=0)

    downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
    print('Downstream node matrix imported.')

    flexibility_bands = pd.DataFrame()
    for use_case in ['home', 'work']:
        flexibility_bands_tmp = \
                pd.read_csv('grid_data/ev_flexibility_bands_{}_{}.csv'.format(grid_id, use_case),
                            index_col=0, dtype=np.float32)
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

    timesteps_per_week = 24*4
    print('Starting optimisation for week {}.'.format(iteration))
    start_time = (iteration * timesteps_per_week) % 672
    if iteration % 7 != 6:
        timesteps = edisgo_obj.timeseries.timeindex[iteration * timesteps_per_week:(iteration + 1) * timesteps_per_week+2]
        flexibility_bands_week = flexibility_bands.iloc[start_time:start_time + timesteps_per_week+2].set_index(timesteps)
        energy_level_end = None
    else:
        timesteps = edisgo_obj.timeseries.timeindex[iteration * timesteps_per_week:(iteration + 1) * timesteps_per_week]
        flexibility_bands_week = flexibility_bands.iloc[start_time:start_time+timesteps_per_week].set_index(timesteps)
        energy_level_end = edisgo_obj.timeseries.charging_points_active_power.loc[:, mapping.index].sum()

    model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective='residual_load',
                            optimize_storage=False, optimize_ev_charging=True,
                            mapping_cp=mapping,
                            energy_band_charging_points=flexibility_bands_week,
                            pu=False, energy_level_end=energy_level_end)
    print('Set up model for week {}.'.format(iteration))

    x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
    curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
    v_bus, p_line, q_line = optimize(model, 'gurobi')

    print('Finished optimisation for week {}.'.format(iteration))
    x_charge.astype(np.float16).to_csv(
        result_dir+'/x_charge_{}.csv'.format(iteration))
    soc.astype(np.float16).to_csv(result_dir + '/soc_{}.csv'.format(iteration))
    x_charge_ev.astype(np.float16).to_csv(
        result_dir + '/x_charge_ev_{}.csv'.format(iteration))
    energy_level_cp.astype(np.float16).to_csv(
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


if __name__ == '__main__':
    pool = mp.Pool(int(mp.cpu_count()/2))
    grid_ids = [2534, 1811, 1690, 177, 176, 1056]
    weeks = [i for i in range(14)]
    results = [pool.apply_async(run_optimized_charging_parallel, args=(grid_id, week))
               for grid_id in grid_ids for week in weeks]
    pool.close()
    print('SUCCESS')
