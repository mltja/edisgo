# Test to implement functionality of optimisation
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.tools import get_nodal_residual_load
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

edisgo_dir = r'\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_object_files_full\Electrification_2050\1056\reduced'
edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
flexibility_bands_home = \
    pd.read_csv('grid_data/ev_flexibility_bands.csv', index_col=0)
flexibility_bands_work = \
    pd.read_csv('grid_data/ev_flexibility_bands.csv', index_col=0)

timesteps = edisgo_obj.timeseries.timeindex[0:7*24*4]

flexibility_bands_home = \
    pd.read_csv('grid_data/ev_flexibility_bands.csv', index_col=0)