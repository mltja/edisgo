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
    edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(grid_id)
    edisgo_obj = import_edisgo_from_files(edisgo_dir)
    downstream_node_matrix = get_downstream_nodes_matrix(edisgo_obj) #Todo: change back to iterative? add .topology to edisgo
    downstream_node_matrix.to_csv('grid_data/downstream_node_matrix_{}.csv'.format(grid_id))
    return


if __name__ == '__main__':
    pool = mp.Pool(6)

    grid_ids = [2534, 1811, 1690, 177, 176, 1056]
    results = pool.map(get_downstream_node_matrix_parallel_server, grid_ids)

    pool.close()

    print('SUCCESS')