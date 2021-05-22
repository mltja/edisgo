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