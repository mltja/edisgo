import pandas as pd
import numpy as np
from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.timeseries import get_component_timeseries
import edisgo.flex_opt.charging_ev as cEV
import edisgo.flex_opt.optimization as opt
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative

grid_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data'
ev_dir = 'C:/Users/Anya.Heider/DistFlex/AllgaeuNetz/data/' \
         'BEV_medium_00002_standing_times.csv'

timeindex = pd.date_range('2011-01-01', periods=8760, freq='h')
timeindex_charging = pd.date_range('2011-01-01', periods=8760, freq='15min')
storage_ts = pd.DataFrame({'Storage 1': 8760*[0]}, index=timeindex)

edisgo = import_edisgo_from_files(grid_dir)
get_component_timeseries(edisgo, timeseries_load='demandlib',
                timeseries_generation_fluctuating='oedb',
                timeseries_storage_units=storage_ts)
timesteps = edisgo.timeseries.timeindex[0:24]

ev_data = pd.read_csv(ev_dir, index_col=0)
charging_events = ev_data.loc[ev_data.chargingdemand>0]
energy_bands = cEV.get_ev_timeseries(charging_events, mode='single_week')

timeindex_charging = pd.date_range('2011-01-01', periods=8760, freq='15min')
energy_band_ev = energy_bands.iloc[210:]
energy_band_ev.set_index(timeindex_charging[0:len(energy_band_ev)], inplace=True)
energy_band_ev.upper = energy_band_ev.upper.astype(float)
energy_band_ev.lower = energy_band_ev.lower.astype(float)
energy_band_ev = energy_band_ev.resample('h').min()

cp_ags_idx = [[0, 0]]
mapping = pd.DataFrame(index=edisgo.topology.charging_points_df.index, columns=['ags', 'cp_idx'], data=cp_ags_idx)
downstream_node_matrix = get_downstream_nodes_matrix_iterative(edisgo.topology)
energy_band_charging_point = energy_band_ev.rename(columns=({
    'lower': 'lower_{}_{}'.format(cp_ags_idx[0][0], cp_ags_idx[0][1]),
    'upper': 'upper_{}_{}'.format(cp_ags_idx[0][0], cp_ags_idx[0][1]),
    'power': 'power_{}_{}'.format(cp_ags_idx[0][0], cp_ags_idx[0][1])
}))
energy_band_charging_point.plot()

model = opt.setup_model(edisgo, downstream_node_matrix=downstream_node_matrix, mapping_cp=mapping,
                        timesteps=timesteps, energy_band_charging_points=energy_band_charging_point,
                        print_model=True)

x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
               curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
               v_bus, p_line, q_line  = opt.optimize(model, 'glpk')

edisgo_obj = edisgo
flexibility_bands = energy_band_charging_point
downstream_nodes_matrix = downstream_node_matrix
timesteps_per_iteration = 3
iterations_per_era = 3
charging_start = None
energy_level_start = None
energy_level_beginning = None
overlap_interations = 2
energy_level = {}
charging_ev = {}
for iteration in range(
        int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)):  # edisgo_obj.timeseries.timeindex.week.unique()
    if iteration >= 3:
        break
    print('Starting optimisation for week {}.'.format(iteration))
    # timesteps = edisgo_obj.timeseries.timeindex[
    #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
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
        energy_level_end = pd.Series(index=['ChargingPoint'], data=[
            3])  # edisgo_obj.timeseries.charging_points_active_power.loc[:, mapping.index].sum()
    # if week == 0:
    model = opt.setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective='peak_load',
                            optimize_storage=False, optimize_ev_charging=True,
                            mapping_cp=mapping,
                            energy_band_charging_points=flexibility_bands_week,
                            pu=False, energy_level_end=energy_level_end, charging_start=charging_start,
                            energy_level_start=energy_level_start, energy_level_beginning=energy_level_beginning)
    print('Set up model for week {}.'.format(iteration))

    x_charge, soc, charging_ev[iteration], energy_level[iteration], curtailment_feedin, \
    curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
    v_bus, p_line, q_line = opt.optimize(model, 'gurobi')
    if iteration % iterations_per_era != iterations_per_era - 1:
        charging_start = charging_ev[iteration].iloc[-overlap_interations]
        energy_level_start = energy_level[iteration].iloc[-overlap_interations]
    else:
        charging_start = None
        energy_level_start = None

    if iteration % iterations_per_era == 0:
        energy_level_beginning = energy_level[iteration].iloc[0]