# Script to extract and save independent feeders
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.tools import extract_feeders_nx
import multiprocessing as mp


def extract_feeders_parallel(grid_id):
    try:
        root_dir = r'U:\Software'
        edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id)
        save_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}'.format(grid_id)
        edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
        extract_feeders_nx(edisgo_obj, save_dir)
    except Exception as e:
        print('Problem in grid {}.'.format(grid_id))
        print(e)


# if __name__ == '__main__':
#     pool = mp.Pool(6)
#
#     grid_ids = [2534, 1811, 1690, 177, 176, 1056]
#     results = pool.map_async(extract_feeders_parallel, grid_ids).get()
#
#     pool.close()
#
#     print('SUCCESS')

extract_feeders_parallel(176)
