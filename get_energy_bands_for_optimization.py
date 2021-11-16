from edisgo.flex_opt.charging_ev import get_ev_timeseries
from pathlib import Path
import edisgo.flex_opt.charging_ev as cEV
import pandas as pd
import numpy as np
import multiprocessing as mp


def get_energy_bands_for_optimization_parallel_server(root_dir, grid_id, use_case):
    edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(grid_id)
    data_dir = Path(root_dir + r'\simbev_nep_2035_results\cp_standing_times_mapping')
    return {(grid_id, use_case):
                cEV.get_energy_bands_for_optimization(data_dir, edisgo_dir, grid_id, use_case,
                                                      time_offset=0)}


# if __name__ == '__main__':
#     root_dir = r'U:\Software'
#     grid_ids = [2534, 1811, 1690, 1056, 177, 176]
#     use_cases = ['work', 'home']
#     pool = mp.Pool(12)
#     results = [pool.apply(get_energy_bands_for_optimization_parallel_server, args=(root_dir, grid_id, use_case))
#                for use_case in use_cases for grid_id in grid_ids]
#
#     pool.close()
#
#     for result in results:
#         for (grid_id, use_case) in result.keys():
#             result[(grid_id, use_case)].to_csv('grid_data/ev_flexibility_bands_{}_{}_00.csv'.format(grid_id, use_case))
#
#     print('SUCCESS')
# get_energy_bands_for_optimization_parallel_server(r'U:\Software', 177, 'work')

for grid_id in [176, 177, 1690]:
    flexibility_bands = pd.DataFrame()
    for use_case in ['home', 'work']:
        flexibility_bands_tmp = \
            pd.read_csv(
                'grid_data/ev_flexibility_bands_{}_{}_00.csv'.format(grid_id,
                                                                     use_case),
                index_col=0)
        rename_dict = {col: col + '_{}'.format(use_case) for col in
                       flexibility_bands_tmp.columns}
        flexibility_bands_tmp.rename(columns=rename_dict, inplace=True)
        flexibility_bands = pd.concat([flexibility_bands, flexibility_bands_tmp],
                                      axis=1)

    columns_upper = [col for col in flexibility_bands.columns if 'upper' in col]
    columns_lower = [col for col in flexibility_bands.columns if 'lower' in col]
    columns_power = [col for col in flexibility_bands.columns if 'power' in col]
    aggregated_bands = pd.DataFrame()
    aggregated_bands['upper'] = flexibility_bands[columns_upper].sum(axis=1)
    aggregated_bands['lower'] = flexibility_bands[columns_lower].sum(axis=1)
    aggregated_bands['power'] = flexibility_bands[columns_power].sum(axis=1)
    aggregated_bands.to_csv(r'U:\Software\grids\{}\flex_ev.csv'.format(grid_id))
print("SUCCESS")
