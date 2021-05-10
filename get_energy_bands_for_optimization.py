from edisgo.flex_opt.charging_ev import get_ev_timeseries
from pathlib import Path
import edisgo.flex_opt.charging_ev as cEV
import pandas as pd
import numpy as np

grid_id = 1056
use_cases = ['work', 'home']

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\cp_standing_times_mapping",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)
edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_full\Electrification_2050\{}\reduced'.format(grid_id)

gdf_cps_total, df_standing_total = cEV.charging_existing_edisgo_object(data_dir, grid_id, edisgo_dir, [])

for use_case in use_cases:
    if use_case == 'home':
        df_standing_times = df_standing_total.loc[df_standing_total.use_case == 3]
    elif use_case == 'work':
        df_standing_times = df_standing_total.loc[df_standing_total.use_case == 4]
    else:
        raise Exception('Only home and work charging have flexibility.')

    cp_indices = df_standing_times.cp_idx.unique()
    weekly_bands = pd.DataFrame()
    for idx in cp_indices:
        charging = df_standing_times.loc[df_standing_times.cp_idx == idx]
        ags = charging.ags.unique()
        for ag in ags:
            charging_cp = charging.loc[charging.ags == ag]
            cp_sub_indices = charging_cp.cp_sub_idx.unique()
            weekly_bands_cp = pd.DataFrame()
            for sub_index in cp_sub_indices:
                charging_events = charging_cp.loc[charging_cp.cp_sub_idx == sub_index]
                weekly_energy_band = get_ev_timeseries(charging_events)
                weekly_bands_cp = pd.concat([weekly_bands_cp, weekly_energy_band], axis=1)
            if len(cp_sub_indices) > 1:
                weekly_bands['_'.join(['upper',str(ag), str(idx)])] = weekly_bands_cp['upper'].sum(axis=1)
                weekly_bands['_'.join(['lower',str(ag), str(idx)])] = weekly_bands_cp['lower'].sum(axis=1)
                weekly_bands['_'.join(['power',str(ag), str(idx)])] = weekly_bands_cp['power'].sum(axis=1)
            else:
                weekly_bands['_'.join(['upper',str(ag), str(idx)])] = weekly_bands_cp['upper']
                weekly_bands['_'.join(['lower',str(ag), str(idx)])] = weekly_bands_cp['lower']
                weekly_bands['_'.join(['power',str(ag), str(idx)])] = weekly_bands_cp['power']
    weekly_bands.to_csv('grid_data/ev_flexibility_bands_{}.csv'.format(use_case))
    print('Use case {} finished.'. format(use_case))

print('SUCCESS')