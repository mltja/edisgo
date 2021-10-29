from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, optimize, check_mapping
from edisgo.tools.tools import convert_impedances_to_mv, extract_feeders_nx

import pandas as pd
import numpy as np

# import edisgo objects and assign feeder ids
# feeders = {}
# for grid_id in [176]:
#     edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\dumb'.format(grid_id)
#     edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=False)
#     feeders[grid_id] = extract_feeders_nx(edisgo_obj)

grid_id = 2534
feeder_id = 0
root_dir = r'U:\Software'
edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)
edisgo_orig = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
edisgo_obj = convert_impedances_to_mv(edisgo_orig)
downstream_nodes_matrix = pd.read_csv(
    'grid_data/feeder_data/downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id),
    index_col=0)

downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
downstream_nodes_matrix = downstream_nodes_matrix.loc[
        edisgo_obj.topology.buses_df.index,
        edisgo_obj.topology.buses_df.index]

timesteps = edisgo_obj.timeseries.timeindex[0:96]
objective='residual_load'
model = setup_model(edisgo_obj, downstream_nodes_matrix, timesteps, objective=objective,
                            optimize_storage=False, optimize_ev_charging=False,
                            pu=False, v_min=1.0, v_max=1.0)

x_charge, soc, charging_ev, energy_level, curtailment_feedin, \
    curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
    v_bus, p_line, q_line, slack_charging, slack_energy = optimize(model, 'gurobi')