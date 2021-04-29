from edisgo.flex_opt.charging_ev import get_ev_data
from pathlib import Path
import edisgo.flex_opt.charging_ev as cEV
import pandas as pd
import numpy as np

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\cp_standing_times_mapping",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)
edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_full\Electrification_2050\1056\reduced'

gdf_cps_total, df_standing_total = cEV.charging_existing_edisgo_object(data_dir, edisgo_dir, [])

df_standing_times_home = df_standing_total.loc[df_standing_total.use_case == 3]
df_standing_times_work = df_standing_total.loc[df_standing_total.use_case == 4]

df_standing_times = df_standing_times_work
cp_indices = df_standing_times.cp_idx.unique()
weekly_bands = pd.DataFrame()
for idx in cp_indices:
    charging = df_standing_times.loc[df_standing_times.cp_idx == idx]
    cp_sub_indices = charging.cp_sub_idx.unique()
    weekly_bands_cp = pd.DataFrame()
    for sub_index in cp_sub_indices:
        charging_events = charging.loc[charging.cp_sub_idx == sub_index]
        weekly_energy_band = get_ev_data(charging_events)
        weekly_bands_cp = pd.concat([weekly_bands_cp, weekly_energy_band], axis=1)
    if len(cp_sub_indices) > 1:
        weekly_bands['upper_' + str(idx)] = weekly_bands_cp['upper'].sum(axis=1)
        weekly_bands['lower_' + str(idx)] = weekly_bands_cp['lower'].sum(axis=1)
        weekly_bands['power_' + str(idx)] = weekly_bands_cp['power'].sum(axis=1)
    else:
        weekly_bands['upper_' + str(idx)] = weekly_bands_cp['upper']
        weekly_bands['lower_' + str(idx)] = weekly_bands_cp['lower']
        weekly_bands['power_' + str(idx)] = weekly_bands_cp['power']
weekly_bands.to_csv('grid_data/ev_flexibility_bands_work.csv')

print('SUCCESS')