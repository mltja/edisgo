# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping, \
    prepare_time_invariant_parameters, update_model
from edisgo.tools.tools import convert_impedances_to_mv
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative
import edisgo.flex_opt.charging_ev as cEV
import pandas as pd
import numpy as np
import os
import datetime

optimize_storage = False
optimize_ev = True
solver = 'gurobi'

from pathlib import Path

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 177
feeder_id = 7 # 1,6
root_dir = r'U:\Software'
mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)
edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)
result_dir = 'results/{}/{}'.format(grid_id, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
data_dir = Path(root_dir + r'\simbev_nep_2035_results\cp_standing_times_mapping')
objective='residual_load'

edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

print('eDisGo object imported.')

edisgo_obj = convert_impedances_to_mv(edisgo_orig)

print('Converted impedances to mv.')

try:
    downstream_nodes_matrix = pd.read_csv(
        'grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id),
        index_col=0)
except:
    downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj.topology)
    #downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix_{}.csv'.format(grid_id))

downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
downstream_nodes_matrix = downstream_nodes_matrix.loc[
        edisgo_obj.topology.buses_df.index,
        edisgo_obj.topology.buses_df.index]
print('Downstream node matrix imported.')

data_dir = r'U:\Software\eDisGo_mirror'
flexibility_bands = pd.DataFrame()
for use_case in ['home', 'work']:
    try:
        flexibility_bands_tmp = \
            pd.read_csv(os.path.join(
                data_dir, 'grid_data/ev_flexibility_bands_{}_{}.csv'.format(grid_id, use_case)),
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
        pd.read_csv(mapping_dir + '/cp_data_home_within_grid_{}.csv'.
                      format(grid_id)).set_index('edisgo_id')
mapping_work = \
    pd.read_csv(mapping_dir + '/cp_data_work_within_grid_{}.csv'.
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
    flex_band_identifier.append('lower_'+cp)
    flex_band_identifier.append('upper_'+cp)
    flex_band_identifier.append('power_'+cp)
flexibility_bands = flexibility_bands[flex_band_identifier]

check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
print('Data checked. Please pay attention to warnings.')

# Create dict with time invariant parameters
parameters = prepare_time_invariant_parameters(edisgo_obj, downstream_nodes_matrix, pu=False, optimize_storage=False,
                                               optimize_ev_charging=True, cp_mapping=mapping)
print('Time-invariant parameters extracted.')
# parameters['branches'][['bus0', 'bus1', 'r', 'x', 's_nom']].to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\176\feeder\6\topology\branches.csv')
# parameters['nodal_active_power'].T.to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\176\feeder\6\timeseries\nodal_active_power.csv')
# parameters['nodal_reactive_power'].T.to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\176\feeder\6\timeseries\nodal_reactive_power.csv')
# parameters['downstream_nodes_matrix'].to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\176\feeder\6\topology\BIBC.csv')

timesteps_per_iteration = 24*4
iterations_per_era = 7
charging_start = None
energy_level_start = None
overlap_interations = 24
energy_level = {}
charging_ev = {}
for iteration in range(0,
        int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)):  # edisgo_obj.timeseries.timeindex.week.unique()

    print('Starting optimisation for week {}.'.format(iteration))
    start_time = (iteration * timesteps_per_iteration) % 672
    if iteration % iterations_per_era != iterations_per_era - 1:
        timesteps = edisgo_obj.timeseries.timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration + overlap_interations]
        flexibility_bands_week = flexibility_bands.iloc[
                                 start_time:start_time + timesteps_per_iteration + overlap_interations].set_index(
            timesteps)
        energy_level_end = None
    else:
        timesteps = edisgo_obj.timeseries.timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
        flexibility_bands_week = flexibility_bands.iloc[start_time:start_time + timesteps_per_iteration].set_index(
            timesteps)

    if charging_start is not None:
        for cp_tmp in energy_level_start.index:
            flex_id_tmp = '_'.join([str(mapping.loc[cp_tmp, 'ags']),
                                    str(mapping.loc[cp_tmp, 'cp_idx']),
                                    mapping.loc[cp_tmp, 'use_case']])
            if energy_level_start[cp_tmp] > flexibility_bands_week.iloc[0]['upper_' + flex_id_tmp]:
                print('charging point {} violates upper bound.'.format(cp_tmp))
                if energy_level_start[cp_tmp] - flexibility_bands_week.iloc[0]['upper_' + flex_id_tmp] > 1e-5:
                    raise ValueError('Optimisation should not return values higher than upper bound. '
                                     'Problem for {}. Initial energy level is {}, but upper bound {}.'.format(
                        cp_tmp, energy_level_start[cp_tmp], flexibility_bands_week.iloc[0]['upper_' + flex_id_tmp]))
                else:
                    energy_level_start[cp_tmp] = flexibility_bands_week.iloc[0]['upper_' + flex_id_tmp] - 1e-6
            if energy_level_start[cp_tmp] < flexibility_bands_week.iloc[0]['lower_' + flex_id_tmp]:
                print('charging point {} violates lower bound.'.format(cp_tmp))
                if -energy_level_start[cp_tmp] + flexibility_bands_week.iloc[0]['lower_' + flex_id_tmp] > 1e-5:
                    raise ValueError('Optimisation should not return values lower than lower bound. '
                                     'Problem for {}. Initial energy level is {}, but lower bound {}.'.format(
                        cp_tmp, energy_level_start[cp_tmp], flexibility_bands_week.iloc[0]['lower_' + flex_id_tmp]))
                else:
                    energy_level_start[cp_tmp] = flexibility_bands_week.iloc[0]['lower_' + flex_id_tmp] + 1e-6
            if charging_start[cp_tmp] < 1e-5:
                print('Very small charging power: {}, set to 0.'.format(cp_tmp))
                charging_start[cp_tmp] = 0
    # if week == 0:
    try:
        model = update_model(model, timesteps, parameters, optimize_storage=optimize_storage,
                             optimize_ev=optimize_ev, energy_band_charging_points=flexibility_bands_week,
                             charging_start=charging_start, energy_level_start=energy_level_start,
                             energy_level_end=energy_level_end)
    except NameError:
        model = setup_model(parameters, timesteps, objective=objective,
                            optimize_storage=False, optimize_ev_charging=True,
                            energy_band_charging_points=flexibility_bands_week,
                            charging_start=charging_start,
                            energy_level_start=energy_level_start, energy_level_end=energy_level_end,
                            overlap_interations=overlap_interations)
    print('Set up model for week {}.'.format(iteration))

    result_dict = optimize(model, solver)
    charging_ev[iteration] = result_dict['x_charge_ev']
    energy_level[iteration] = result_dict['energy_level_cp']
    if iteration % iterations_per_era != iterations_per_era - 1:
        charging_start = charging_ev[iteration].iloc[-overlap_interations]
        charging_start.to_csv('results/tests/charging_start.csv')
        energy_level_start = energy_level[iteration].iloc[-overlap_interations]
        energy_level_start.to_csv('results/tests/energy_level_start.csv')
    else:
        charging_start = None
        energy_level_start = None

    if iteration==0:
        os.makedirs(result_dir)
    print('Finished optimisation for week {}.'.format(iteration))
    # x_charge.astype(np.float16).to_csv(
    #     result_dir+'/x_charge_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # soc.astype(np.float16).to_csv(result_dir + '/soc_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # charging_ev[iteration].astype(np.float16).to_csv(
    #     result_dir + '/x_charge_ev_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # energy_level[iteration].astype(np.float16).to_csv(
    #     result_dir + '/energy_band_cp_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # curtailment_feedin.astype(np.float16).to_csv(
    #     result_dir + '/curtailment_feedin_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # curtailment_load.astype(np.float16).to_csv(
    #     result_dir + '/curtailment_load_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # curtailment_reactive_feedin.astype(np.float16).to_csv(
    #     result_dir + '/curtailment_reactive_feedin_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # curtailment_reactive_load.astype(np.float16).to_csv(
    #     result_dir + '/curtailment_reactive_load_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # v_bus.astype(np.float16).to_csv(result_dir + '/bus_voltage_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # p_line.astype(np.float16).to_csv(result_dir + '/line_active_power_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    # q_line.astype(np.float16).to_csv(result_dir + '/line_reactive_power_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    print('Saved results for week {}.'.format(iteration))

print('SUCCESS')
