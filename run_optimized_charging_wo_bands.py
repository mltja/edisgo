from edisgo.flex_opt.charging_ev import get_ev_timeseries
from pathlib import Path
import edisgo.flex_opt.charging_ev as cEV
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.tools import convert_impedances_to_mv
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix_iterative
from edisgo.flex_opt.optimization import setup_model_wo_bands, optimize
import pandas as pd
import geopandas as gpd
import numpy as np

grid_ids = [176]

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results\calculations_for_anya\simbev_nep_2035_results\cp_standing_times_mapping",
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\cp_standing_times_mapping",
)


for grid_id in grid_ids:
    result_dir = 'grid_data/{}'.format(grid_id)
    edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\calculations_for_anya\eDisGo_object_files\simbev_nep_2\{}\reduced'.format(
        grid_id)

    gdf_cps_total, df_standing_total = cEV.charging_existing_edisgo_object(
        data_dir, grid_id, edisgo_dir, [])
    df_standing_total = df_standing_total.loc[df_standing_total.chargingdemand>0]
    df_standing_total.loc[df_standing_total.use_case == 3, 'use_case'] = 'home'
    df_standing_total.loc[df_standing_total.use_case == 4, 'use_case'] = 'work'
    df_standing_total = df_standing_total.loc[df_standing_total.use_case.isin(['home', 'work'])]

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
    timesteps_per_iteration = 24 * 4
    for iteration in range(int(len(
            edisgo_obj.timeseries.timeindex) / timesteps_per_iteration) - 1):  # edisgo_obj.timeseries.timeindex.week.unique()
        print('Starting optimisation for week {}.'.format(iteration))
        timesteps = edisgo_obj.timeseries.timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
    model = \
        setup_model_wo_bands(edisgo_obj, downstream_nodes_matrix, timesteps,
                             optimize_storage=False, mapping_cp=mapping,
                             standing_times=df_standing_total, pu=False)

    print('Set up model for iteration {}.'.format(iteration))
    x_charge, soc, x_charge_ev, energy_band_cp, curtailment_feedin, curtailment_load = optimize(
        model, 'glpk')
    print('Finished optimisation for week {}.'.format(iteration))
    x_charge.astype(np.float16).to_csv(
        result_dir + '/x_charge_{}.csv'.format(iteration))
    soc.astype(np.float16).to_csv(result_dir + '/soc_{}.csv'.format(iteration))
    x_charge_ev.astype(np.float16).to_csv(
        result_dir + '/x_charge_ev_{}.csv'.format(iteration))
    energy_band_cp.astype(np.float16).to_csv(
        result_dir + '/energy_band_cp_{}.csv'.format(iteration))
    curtailment_feedin.astype(np.float16).to_csv(
        result_dir + '/curtailment_feedin_{}.csv'.format(iteration))
    curtailment_load.astype(np.float16).to_csv(
        result_dir + '/curtailment_load_{}.csv'.format(iteration))
    print('Saved results for week {}.'.format(iteration))

