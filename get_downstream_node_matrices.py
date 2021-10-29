# Script to extract downstream node matrix
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative, get_downstream_nodes_matrix
import numpy as np
import os
import multiprocessing as mp


def get_downstream_node_matrix_parallel_server(grid_id):
    if os.path.isfile('grid_data/downstream_node_matrix_{}.csv'.format(grid_id)):
        return
    root_dir = r'U:\Software'
    edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id)
    edisgo_obj = import_edisgo_from_files(edisgo_dir)
    downstream_node_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj.topology)
    downstream_node_matrix.to_csv('grid_data/downstream_node_matrix_{}.csv'.format(grid_id))
    return


def get_downstream_node_matrix_feeders_parallel_server(grid_id_feeder_tuple):
    grid_id = grid_id_feeder_tuple[0]
    feeder_id = grid_id_feeder_tuple[1]
    if os.path.isfile('grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id)):
        return
    try:
        root_dir = r'U:\Software'
        edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)
        edisgo_obj = import_edisgo_from_files(edisgo_dir)
        downstream_node_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj.topology)
        downstream_node_matrix.to_csv('grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id))
    except Exception as e:
        print(e.args)
        print(e)
    return


if __name__ == '__main__':
    pool = mp.Pool(int(mp.cpu_count()/2))

    grid_ids = [2534, 1811, 1690, 1056, 176, 177]
    root_dir = r'U:\Software'
    grid_id_feeder_tuples = []
    for grid_id in grid_ids:
        edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\feeder'.format(grid_id)
        for feeder in os.listdir(edisgo_dir):
            grid_id_feeder_tuples.append((grid_id, feeder))
    results = pool.map_async(get_downstream_node_matrix_feeders_parallel_server, grid_id_feeder_tuples).get()

    pool.close()

    print('SUCCESS')