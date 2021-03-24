import pandas as pd
import geopandas as gpd
import contextily as ctx
import os
import logging
import warnings
import matplotlib.pyplot as plt
import matplotlib

from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.plots import mv_grid_topology
from pathlib import Path

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

font = {'family' : 'Latin Modern Roman',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

grid_id = "2534"

dir_dumb = Path(
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_curtailment_results\Electrification_2050\{}\dumb".format(
        grid_id
    )
)

edisgo_dumb = import_edisgo_from_files(
    directory=dir_dumb,
    import_topology=True,
    import_timeseries=True,
    import_results=True,
)

# dir_res = Path(
#     r"\\192.168.10.221\Daten_flexibel_02\simbev_results\eDisGo_curtailment_results\Electrification_2050\{}\residual".format(
#         grid_id
#     )
# )
#
# edisgo_residual = import_edisgo_from_files(
#     directory=dir_res,
#     import_topology=True,
#     import_timeseries=True,
#     import_results=True,
# )

print("breaker")

mv_grid_topology(
    edisgo_dumb,
    node_color="charging_park",
)