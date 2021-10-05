# Script to create new edisgo object with optimised charging timeseries
import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files
import multiprocessing as mp

def extract_edisgo_object(grid_id):
    res_dir = r'U:\Software\eDisGo_mirror\results\residual_load_test2'
    feeders = []
    edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\feeder'.format(grid_id)
    for feeder in os.listdir(edisgo_dir):
        feeders.append(feeder)

    x_charge_ev_grid = pd.DataFrame()
    for feeder_id in feeders:
        x_charge_ev_feeder = pd.DataFrame()
        for i in range(14):
            try:
                x_charge_ev_feeder_tmp = pd.read_csv(res_dir + '/{}/{}/x_charge_ev_{}_{}_{}.csv'.format(
                    grid_id, feeder_id, grid_id, feeder_id, i),
                                                     index_col=0, parse_dates=True)
                x_charge_ev_feeder = pd.concat([x_charge_ev_feeder, x_charge_ev_feeder_tmp], sort=False)
            except:
                print('Results for feeder {} in grid {} could not be loaded.'.format(feeder_id, grid_id))
        try:
            x_charge_ev_grid = pd.concat([x_charge_ev_grid, x_charge_ev_feeder], axis=1, sort=False)
        except:
            print('Feeder {} not added'.format(feeder_id))

    x_charge_ev_grid = x_charge_ev_grid.loc[~x_charge_ev_grid.index.duplicated(keep='last')]

    # import original edisgo object and create new directory
    edisgo_dir=r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id)
    edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    edisgo_dir_new = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\optimised'.format(grid_id)
    os.makedirs(edisgo_dir_new, exist_ok=True)

    # update timeseries of original object
    tmp = edisgo.timeseries.charging_points_active_power
    tmp.update(x_charge_ev_grid)
    edisgo.timeseries.charging_points_active_power = tmp

    # save new object
    edisgo.save(edisgo_dir_new)


if __name__ == '__main__':
    pool = mp.Pool(2)

    grid_ids = [176, 1690]#1811,, 2534, 1056, 177
    results = pool.map_async(extract_edisgo_object, grid_ids).get()
    pool.close()
    pool.join()

    print('SUCCESS')

# extract_edisgo_object(1811)
# print('SUCCESS')