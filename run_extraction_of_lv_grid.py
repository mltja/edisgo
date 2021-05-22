from edisgo import EDisGo
import numpy as np
import pandas as pd
import edisgo.flex_opt.optimization as opt
from edisgo.edisgo import import_edisgo_from_files

from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative


def convert_to_pu_system(grid, s_base=1, convert_timeseries=True,
                         timeseries_inplace=False):
    """
    Method to convert grid to pu-system. Can be used to run optimisation with
    it.

    :param grid:
    :param s_base:
    :param convert_timeseries: boolean, determines whether timeseries data
        should also be converted
    :param timeseries_inplace: boolean, determines whether timeseries data is
        changed directly inside the edisgo-object. Otherwise timeseries are
        returned as DataFrame. Note: Be careful with this option and only use
        when the whole object is converted or you only need one grid.
    :return:
    """
    v_base = grid.nominal_voltage
    z_base = np.square(v_base)/s_base
    grid.base_power = s_base
    grid.base_impedance = z_base
    # convert all components and add pu_columns
    pu_cols = {'lines': ['r_pu', 'x_pu', 's_nom_pu'],
               'generators': ['p_nom_pu'],
               'loads': ['peak_load_pu'],
               'storage_units': ['p_nom_pu']}
    for comp, cols in pu_cols.items():
        new_cols = [col for col in cols if col not in
                    getattr(grid.edisgo_obj.topology, comp+'_df').columns]
        for col in new_cols:
            getattr(grid.edisgo_obj.topology, comp+'_df')[col] = np.NaN
    grid.edisgo_obj.topology.lines_df.loc[grid.lines_df.index, 'r_pu'] = \
        grid.lines_df.r/grid.base_impedance
    grid.edisgo_obj.topology.lines_df.loc[grid.lines_df.index, 'x_pu'] = \
        grid.lines_df.x/grid.base_impedance
    grid.edisgo_obj.topology.lines_df.loc[grid.lines_df.index, 's_nom_pu'] = \
        grid.lines_df.s_nom/grid.base_power
    grid.edisgo_obj.topology.generators_df.loc[grid.generators_df.index,
                                               'p_nom_pu'] = \
        grid.generators_df.p_nom/grid.base_power
    grid.edisgo_obj.topology.loads_df.loc[grid.loads_df.index, 'peak_load_pu'] = \
        grid.loads_df.peak_load/grid.base_power
    grid.edisgo_obj.topology.storage_units_df.loc[grid.storage_units_df.index,
                                                  'p_nom_pu'] = \
        grid.storage_units_df.p_nom/grid.base_power
    if hasattr(grid, 'charging_points_df') and \
            not grid.charging_points_df.empty:
        if not 'p_nom_pu' in grid.charging_points_df.columns:
            grid.edisgo_obj.topology.charging_points_df['p_nom_pu'] = np.NaN
        grid.edisgo_obj.topology.charging_points_df.loc[
            grid.charging_points_df.index, 'p_nom_pu'] = \
            grid.charging_points_df.p_nom/grid.base_power
    # convert timeseries
    if convert_timeseries:
        if not hasattr(grid.edisgo_obj.timeseries, 'generators_active_power'):
            print('No data inside the timeseries object. Please provide '
                  'timeseries to convert to the pu-system. Process is '
                  'interrupted.')
            return
        timeseries = {}
        # pass if values would not change
        if grid.base_power == 1:
            if timeseries_inplace:
                return
        for component in ['generators', 'loads',
                          'storage_units', 'charging_points']:
            if hasattr(grid.edisgo_obj.timeseries,
                       component + '_active_power'):
                active_power = getattr(grid.edisgo_obj.timeseries,
                                       component + '_active_power')
                reactive_power = getattr(grid.edisgo_obj.timeseries,
                                         component + '_reactive_power')
                comp_names = getattr(grid, component + '_df').index
                active_power_pu = active_power[comp_names]/grid.base_power
                reactive_power_pu = reactive_power[comp_names]
                if timeseries_inplace:
                    active_power[comp_names] = active_power_pu
                    reactive_power[comp_names] = reactive_power_pu
                else:
                    timeseries[component+'_active_power_pu'] = active_power_pu
                    timeseries[component+'_reactive_power_pu'] = reactive_power_pu
        return timeseries


result_dir = r'\\192.168.10.221\Daten_flexibel_02\Anyas_Daten\EV_Optimisation\1056'

# grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
# edisgo = import_edisgo_from_files(grid_dir)
# get_downstream_nodes_matrix_iterative(edisgo)

grid_id = 1056

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_final\Electrification_2050\{}\reduced'.format(grid_id)#Todo: change back to final in the end
edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

print('eDisGo object imported.')

# downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj)
# downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix.csv', dtype=uint8)
downstream_nodes_matrix = pd.read_csv('grid_data/downstream_node_matrix.csv',
                                      index_col=0)
downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
print('Downstream node matrix imported.')

lv_grids = [grid for grid in edisgo.topology.mv_grid.lv_grids]

lv_grid = lv_grids[0]
timeseries = convert_to_pu_system(lv_grid)
downstream_node_matrix = downstream_nodes_matrix.loc[lv_grid.buses_df.index,
                                                     lv_grid.buses_df.index]
model = opt.setup_model(lv_grid, downstream_node_matrix,
                        optimize_storage=False, optimize_ev_charging=False)
x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
   curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
   v_bus, p_line, q_line = opt.optimize(model, 'glpk')

pypsa_network = edisgo.to_pypsa(mode='lv',
                                lv_grid_name='LVGrid_{}'.format(lv_grid.id))
pypsa_network.pf()
voltage = pypsa_network.buses_t.v_mag_pu
# compare results
diff = voltage-v_bus/lv_grid.nominal_voltage
max_diff = diff.abs().max().max()
print('Maximum deviation voltage: {}%'.format(max_diff*100))
print('SUCCESS')
