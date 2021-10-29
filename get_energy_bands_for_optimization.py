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
get_energy_bands_for_optimization_parallel_server(r'U:\Software', 177, 'work')