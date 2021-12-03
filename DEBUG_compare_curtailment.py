from edisgo.edisgo import import_edisgo_from_files
import pandas as pd
from results_helper_functions import relative_load, voltage_diff
from edisgo.tools.tools import extract_feeders_nx
from run_extract_optimized_edisgo_object import combine_results_for_grid
import os
import numpy as np
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping, prepare_time_invariant_parameters

grid_id = 176
feeder_id = 6
feeders = [feeder_id]
edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)
res_dir = r'U:\Software\buildings\eDisGo_mirror\results\residual_load_test2'


def add_curtailment_linopt(edisgo_dir, feeders, res_dir):
    edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    curtailment_ev = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_ev")
    curtailment_load = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_load")
    curtailment_feedin = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_feedin")

    # Todo: Replace when reactive curtailment is saved in code
    data_dir = r'U:\Software\eDisGo_mirror'
    root_dir = r'U:\Software'
    mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)

    downstream_nodes_matrix = pd.read_csv(os.path.join(
        data_dir, 'grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id)),
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
            pd.read_csv(os.path.join(
                data_dir, 'grid_data/ev_flexibility_bands_{}_{}.csv'.format(grid_id, use_case)),
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
        flex_band_identifier.append('lower_' + cp)
        flex_band_identifier.append('upper_' + cp)
        flex_band_identifier.append('power_' + cp)
    flexibility_bands = flexibility_bands[flex_band_identifier]

    check_mapping(mapping, edisgo_obj.topology, flexibility_bands)
    print('Data checked. Please pay attention to warnings.')

    # Create dict with time invariant parameters
    parameters = prepare_time_invariant_parameters(edisgo_obj, downstream_nodes_matrix, pu=False,
                                                   optimize_storage=False,
                                                   optimize_ev_charging=True, cp_mapping=mapping)

    tan_phi_load = (parameters['nodal_reactive_load'] / \
                    parameters['nodal_active_load']).fillna(0)
    tan_phi_feedin = (parameters['nodal_reactive_feedin'] / \
                      parameters['nodal_active_feedin']).fillna(0)

    curtailment_reactive_load = curtailment_load*tan_phi_load.T
    curtailment_reactive_feedin = curtailment_feedin*tan_phi_feedin.T

    edisgo_obj.timeseries.mode = 'manual'
    for node in curtailment_load:
        if curtailment_load[node].sum().sum()>0:
            edisgo_obj.add_component('Generator', ts_active_power=curtailment_load[node]+curtailment_ev[node],
                                     ts_reactive_power=curtailment_reactive_load[node], bus=node,
                                     generator_id=node, p_nom=curtailment_load[node].max(),
                                     generator_type='load_curtailment')
            print('Generator added for curtailment at bus {}'.format(node))
        if curtailment_feedin[node].sum().sum()>0:
            edisgo_obj.add_component('Load', ts_active_power=curtailment_feedin[node],
                                     ts_reactive_power=curtailment_reactive_feedin[node], bus=node,
                                     load_id=node, peak_load=curtailment_feedin[node].max(),
                                     annual_consumption=curtailment_feedin[node].sum(),
                                     sector='feedin_curtailment')
            print('Load added for curtailment at bus {}'.format(node))
    return edisgo_obj


def add_curtailment_julia(edisgo_dir, curtailment_dir, factor=1):
    edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    timeindex = edisgo_obj.timeseries.timeindex[672:]
    edisgo_obj.timeseries._timeindex = timeindex
    curtailment_load = pd.read_csv(curtailment_dir + r'\curtailment_load.csv').set_index(timeindex)
    curtailment_feedin = pd.read_csv(curtailment_dir + r'\curtailment_feedin.csv').set_index(timeindex)
    curtailment_reactive_load = pd.read_csv(curtailment_dir + r'\curtailment_reactive_load.csv').set_index(timeindex)
    curtailment_reactive_feedin = pd.read_csv(curtailment_dir + r'\curtailment_reactive_feedin.csv').set_index(timeindex)

    edisgo_obj.timeseries.mode = 'manual'
    for load in curtailment_load:
        name_load = "_".join(load.split("_")[1:])
        node = edisgo_obj.topology.loads_df.loc[name_load, "bus"]
        if curtailment_load[load].sum().sum() > 0:
            edisgo_obj.add_component('Load', ts_active_power=curtailment_load[load]*factor,
                                     ts_reactive_power=curtailment_reactive_load["Q"+load[1:]]*factor, bus=node,
                                     load_id=name_load, peak_load=curtailment_load[load].max()*factor,
                                     annual_consumption=curtailment_load[load].sum()*factor,
                                     sector='load_curtailment')
            print('Generator added for curtailment of load {}'.format(name_load))
    for gen in curtailment_feedin:
        name_gen = "_".join(gen.split("_")[1:])
        node = edisgo_obj.topology.generators_df.loc[name_gen, "bus"]
        if curtailment_feedin[gen].sum().sum() > 0:
            edisgo_obj.add_component('Load', ts_active_power=curtailment_feedin[gen]*factor,
                                     ts_reactive_power=curtailment_reactive_feedin["Q"+gen[1:]]*factor, bus=node,
                                     load_id=name_gen, peak_load=curtailment_feedin[gen].max()*factor,
                                     annual_consumption=curtailment_feedin[gen].sum()*factor,
                                     sector='feedin_curtailment')
            print('Load added for curtailment of generator {}'.format(name_gen))

    edisgo_obj.timeseries.residual_load.plot()
    return edisgo_obj


if __name__ == "__main__":
    # edisgo_obj = add_curtailment_julia(
    #     edisgo_dir,
    #     r'\\s4d-fs.d.ethz.ch\itet-home$\aheider\Desktop\Code Conor\Networks\Feeder6\results')
    edisgo_obj = add_curtailment_linopt(edisgo_dir, feeders, res_dir)
    edisgo_obj.analyze()
    rel_load = relative_load(edisgo_obj)
    print("Minimum voltage is {} p.u.".format(edisgo_obj.results.v_res.min().min()))
    print("Maximum voltage is {} p.u.".format(edisgo_obj.results.v_res.max().max()))
    print("Maximum loading is {} p.u.".format(rel_load.max().max()))

# edisgo_obj.analyze()
# edisgo_obj.save(r'U:\Software\curtailment\results\177')
# edisgo_obj = import_edisgo_from_files(r'U:\Software\curtailment\results\177', import_timeseries=True)
# edisgo_obj.analyze()
# # feeders = extract_feeders_nx(edisgo_obj)
# # feeders[0].save(r'U:\Software\curtailment\results\177')
#
# rel_load = relative_load(edisgo_obj)
# v_diff = voltage_diff(edisgo_obj)
# #
# rel_load.to_csv(r'U:\Software\curtailment\results\rel_load.csv')
# v_diff.to_csv(r'U:\Software\curtailment\results\v_diff.csv')

print('SUCCESS')
