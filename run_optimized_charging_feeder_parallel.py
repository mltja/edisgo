import multiprocessing as mp
import os
import datetime


from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping
from edisgo.tools.tools import convert_impedances_to_mv
import pandas as pd
import geopandas as gpd
import numpy as np


def run_optimized_charging_feeder_parallel(grid_feeder_tuple):
    objective = 'residual_load'
    timesteps_per_iteration = 24 * 4
    iterations_per_era = 7
    solver = 'gurobi'

    config = pd.Series({'objective':objective, 'solver': solver,
                        'timesteps_per_iteration': timesteps_per_iteration,
                        'iterations_per_era': iterations_per_era})

    grid_id = grid_feeder_tuple[0]
    feeder_id = grid_feeder_tuple[1]
    root_dir = r'U:\Software'
    mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)
    edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\feeder\{}'.format(grid_id, feeder_id)
    result_dir = 'results/{}/{}/{}'.format(objective+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), grid_id, feeder_id)

    os.makedirs(result_dir)
    config.to_csv(result_dir+'/config.csv')

    try:

        edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

        print('eDisGo object imported.')

        edisgo_obj = convert_impedances_to_mv(edisgo_orig)

        downstream_nodes_matrix = pd.read_csv(
            'grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id),
            index_col=0)

        print('Converted impedances to mv.')

        downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
        downstream_nodes_matrix = downstream_nodes_matrix.loc[
            edisgo_obj.topology.buses_df.index,
            edisgo_obj.topology.buses_df.index]
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
                flexibility_bands.columns.str.contains('power')]] + 1e-6).values
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

        # extract data for feeder
        mapping = mapping.loc[mapping.index.isin(edisgo_obj.topology.charging_points_df.index)]
        cp_identifier = ['_'.join([str(mapping.loc[cp, 'ags']),
                                   str(mapping.loc[cp, 'cp_idx']),
                                   mapping.loc[cp, 'use_case']])
                         for cp in mapping.index]
        flex_band_identifier = []
        for cp in cp_identifier:
            flex_band_identifier.append('lower_' + cp)
            flex_band_identifier.append('upper_' + cp)
            flex_band_identifier.append('power_' + cp)
        flexibility_bands = flexibility_bands[flex_band_identifier]

        check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
        print('Data checked. Please pay attention to warnings.')

        charging_start = None
        energy_level_start = None
        overlap_interations = 10
        energy_level = {}
        charging_ev = {}

        for iteration in range(
                int(len(
                    edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)):  # edisgo_obj.timeseries.timeindex.week.unique()

            print('Starting optimisation for week {}.'.format(iteration))
            # timesteps = edisgo_obj.timeseries.timeindex[
            #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
            start_time = (iteration * timesteps_per_iteration) % 672
            if iteration % iterations_per_era != iterations_per_era - 1:
                timesteps = edisgo_obj.timeseries.timeindex[
                            iteration * timesteps_per_iteration:(
                                                                            iteration + 1) * timesteps_per_iteration + overlap_interations]
                flexibility_bands_week = flexibility_bands.iloc[
                                         start_time:start_time + timesteps_per_iteration + overlap_interations].set_index(
                    timesteps)
            else:
                timesteps = edisgo_obj.timeseries.timeindex[
                            iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
                flexibility_bands_week = flexibility_bands.iloc[
                                         start_time:start_time + timesteps_per_iteration].set_index(
                    timesteps)
            # if week == 0:
            model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective=objective,
                                optimize_storage=False, optimize_ev_charging=True,
                                mapping_cp=mapping,
                                energy_band_charging_points=flexibility_bands_week,
                                pu=False, charging_start=charging_start,
                                energy_level_start=energy_level_start)
            print('Set up model for week {}.'.format(iteration))

            x_charge, soc, charging_ev[iteration], energy_level[iteration], curtailment_feedin, \
            curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
            v_bus, p_line, q_line = optimize(model, solver)
            if iteration % iterations_per_era != iterations_per_era - 1:
                charging_start = charging_ev[iteration].iloc[-overlap_interations]
                energy_level_start = energy_level[iteration].iloc[-overlap_interations]
            else:
                charging_start = None
                energy_level_start = None

            print('Finished optimisation for week {}.'.format(iteration))
            x_charge.astype(np.float16).to_csv(
                result_dir + '/x_charge_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            soc.astype(np.float16).to_csv(result_dir + '/soc_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            charging_ev[iteration].astype(np.float16).to_csv(
                result_dir + '/x_charge_ev_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            energy_level[iteration].astype(np.float16).to_csv(
                result_dir + '/energy_band_cp_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            curtailment_feedin.astype(np.float16).to_csv(
                result_dir + '/curtailment_feedin_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            curtailment_load.astype(np.float16).to_csv(
                result_dir + '/curtailment_load_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            curtailment_reactive_feedin.astype(np.float16).to_csv(
                result_dir + '/curtailment_reactive_feedin_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            curtailment_reactive_load.astype(np.float16).to_csv(
                result_dir + '/curtailment_reactive_load_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            v_bus.astype(np.float16).to_csv(
                result_dir + '/bus_voltage_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            p_line.astype(np.float16).to_csv(
                result_dir + '/line_active_power_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            q_line.astype(np.float16).to_csv(
                result_dir + '/line_reactive_power_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
            print('Saved results for week {}.'.format(iteration))

    except Exception as e:
        print('Something went wrong with feeder {} of grid {}'.format(feeder_id, grid_id))
        print(e)


if __name__ == '__main__':
    pool = mp.Pool(int(mp.cpu_count()/2))

    grid_ids = [1690]
    root_dir = r'U:\Software'
    grid_id_feeder_tuples = []
    for grid_id in grid_ids:
        edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\feeder'.format(grid_id)
        for feeder in os.listdir(edisgo_dir):
            grid_id_feeder_tuples.append((grid_id, feeder))
    results = pool.map_async(run_optimized_charging_feeder_parallel, grid_id_feeder_tuples).get()

    pool.close()

    print('SUCCESS')