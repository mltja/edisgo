# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.optimization import setup_model, update_model, optimize
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
sns.set()

grid_id = 1056

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_final\Electrification_2050\{}\reduced'.format(grid_id)#Todo: change back to final in the end
edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

downstream_nodes_matrix = get_downstream_nodes_matrix(edisgo_obj)
downstream_nodes_matrix.to_csv('grid_data/downstream_node_matrix.csv')

flexibility_bands_home = \
    pd.read_csv('grid_data/ev_flexibility_bands_home.csv', index_col=0)
flexibility_bands_work = \
    pd.read_csv('grid_data/ev_flexibility_bands_work.csv', index_col=0)
flexibility_bands = pd.concat([flexibility_bands_work, flexibility_bands_home],
                              axis=1)
cp_mapping_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\eDisGo_charging_time_series\{}'.format(grid_id)
mapping_home = \
    gpd.read_file(cp_mapping_dir + '\cp_data_home_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file(cp_mapping_dir + '\cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping = pd.concat([mapping_work, mapping_home])

# timesteps_per_week = 672
# for week in range(int(len(edisgo_obj.timeseries.timeindex)/timesteps_per_week)-1):#edisgo_obj.timeseries.timeindex.week.unique()
#     timesteps = edisgo_obj.timeseries.timeindex[week*timesteps_per_week:(week+1)*timesteps_per_week]
#     # timesteps = edisgo_obj.timeseries.timeindex[
#     #     edisgo_obj.timeseries.timeindex.week == week] # Todo: adapt
#     flexibility_bands = flexibility_bands.iloc[:len(timesteps)].set_index(timesteps)
#     model = setup_model(edisgo_obj, timesteps, optimize_storage=False,
#                         mapping_cp=mapping, energy_band_charging_points=flexibility_bands)
