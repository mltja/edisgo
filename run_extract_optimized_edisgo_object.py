# Script to create new edisgo object with optimised charging timeseries
import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files
import multiprocessing as mp

res_dir = r'U:\Software\buildings\eDisGo_mirror\results\residual_load_test2'


def extract_edisgo_object(grid_id, res_dir, full_object=True, **kwargs):

    edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\feeder'.format(grid_id)
    if full_object:
        feeders=[]
        for feeder in os.listdir(edisgo_dir):
            feeders.append(feeder)
        edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id)
        edisgo_dir_new = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\optimised'.format(grid_id)
    else:
        feeder = kwargs.get('feeder')
        feeders = [feeder]
        edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder)
        edisgo_dir_new = edisgo_dir

    x_charge_ev_grid = combine_results_for_grid(feeders, grid_id, res_dir, 'x_charge_ev')

    # import original edisgo object and create new directory

    edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    os.makedirs(edisgo_dir_new, exist_ok=True)

    # update timeseries of original object
    tmp = edisgo.timeseries.charging_points_active_power
    tmp.update(x_charge_ev_grid)
    edisgo.timeseries.charging_points_active_power = tmp

    # save new object
    edisgo.save(edisgo_dir_new)


def combine_results_for_grid(feeders, grid_id, res_dir, res_name):
    res_grid = pd.DataFrame()
    for feeder_id in feeders:
        res_feeder = pd.DataFrame()
        for i in range(14):
            try:
                res_feeder_tmp = pd.read_csv(res_dir + '/{}/{}/{}_{}_{}_{}.csv'.format(
                    grid_id, feeder_id, res_name, grid_id, feeder_id, i),
                                                     index_col=0, parse_dates=True)
                res_feeder = pd.concat([res_feeder, res_feeder_tmp], sort=False)
            except:
                print('Results for feeder {} in grid {} could not be loaded.'.format(feeder_id, grid_id))
        try:
            res_grid = pd.concat([res_grid, res_feeder], axis=1, sort=False)
        except:
            print('Feeder {} not added'.format(feeder_id))
    res_grid = res_grid.loc[~res_grid.index.duplicated(keep='last')]
    return res_grid


# if __name__ == '__main__':
#     pool = mp.Pool(2)
#     mp.cpu_count()
#     grid_ids = [176, 1690]#1811,, 2534, 1056, 177
#     results = pool.map_async(extract_edisgo_object, grid_ids).get()
#     pool.close()
#     pool.join()
#
#     print('SUCCESS')
if __name__ == '__main__':
    # extract_edisgo_object(176, res_dir, full_object=False, feeder=6)
    # extract_edisgo_object(177, res_dir, full_object=False, feeder=7)
    extract_edisgo_object(2534, res_dir)
    print('SUCCESS')