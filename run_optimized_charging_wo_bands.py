from edisgo.flex_opt.charging_ev import get_ev_timeseries
from pathlib import Path
import edisgo.flex_opt.charging_ev as cEV
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.tools import convert_impedances_to_mv
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative
import pandas as pd
import geopandas as gpd
import numpy as np

grid_ids = [176]

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results\calculations_for_anya\simbev_nep_2035_results\cp_standing_times_mapping",
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\cp_standing_times_mapping",
)


for grid_id in grid_ids:

    edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\calculations_for_anya\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(
        grid_id)

    gdf_cps_total, df_standing_total = cEV.charging_existing_edisgo_object(
        data_dir, grid_id, edisgo_dir, [])
    df_standing_total = df_standing_total.loc[df_standing_total.chargingdemand>0]
    df_standing_times_home = df_standing_total.loc[df_standing_total.use_case == 3]
    df_standing_times_work = df_standing_total.loc[df_standing_total.use_case == 4]

    edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
    edisgo_obj = convert_impedances_to_mv(edisgo_obj)

    print('Converted impedances to mv.')
    try:
        downstream_nodes_matrix = pd.read_csv(
            'grid_data/downstream_node_matrix_{}.csv'.format(grid_id),
            index_col=0)
    except:
        downstream_nodes_matrix = get_downstream_nodes_matrix_iterative(edisgo_obj)
        downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix_{}.csv'.format(grid_id), dtype=np.uint8)

    downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)

    mapping_home = \
        gpd.read_file('grid_data/cp_data_home_within_grid_{}.geojson'.
                      format(grid_id)).set_index('edisgo_id')
    mapping_work = \
        gpd.read_file('grid_data/cp_data_work_within_grid_{}.geojson'.
                      format(grid_id)).set_index('edisgo_id')
    mapping_home['use_case'] = 'home'
    mapping_work['use_case'] = 'work'
    mapping = pd.concat([mapping_work, mapping_home],
                        sort=False)  # , mapping_hpc, mapping_public
    print('Mapping imported.')