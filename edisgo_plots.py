import gc
import os
import pandas as pd
import logging
import warnings
import multiprocessing
import traceback

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

import curtailment as cur

from datetime import timedelta
from pathlib import Path
from time import perf_counter
from edisgo.edisgo import import_edisgo_from_files


gc.collect()

# os.sched_setaffinity(0,range(1000)) # TODO

num_threads = 1 # TODO

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

sub_dir = r"eDisGo_curtailment_results_test" # TODO

scenarios = [ # TODO
    "NEP_C_2035",
    # "Reference_2050",
    # "Szenarette_Kleinwagen_2050",
    # "Mobility_Transition_2050",
    # "Electrification_2050",
    # "Electrification_2050_sensitivity_low_work",
]

grid_ids = ["2534"]#, "177", "1056", "1690", "1811", "176"] # TODO

strategies = ["dumb"]#, "grouped", "reduced", "residual"] # TODO

data_dirs = [
    Path(os.path.join(data_dir, sub_dir, scenario, grid_id, strategy))
    for scenario in scenarios for grid_id in grid_ids for strategy in strategies
]

for d in data_dirs:
    try:
        edisgo_chunk = import_edisgo_from_files(
            directory=d,
            import_topology=True,
            import_timeseries=True,
            import_results=True,
        )

        edisgo_chunk.plot_mv_grid_topology(node_color="charging_park")

        print("breaker")

    except:
        traceback.print_exc()