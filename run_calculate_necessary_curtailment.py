import gc
import os.path
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from calculate_necessary_curtailment import calculate_curtailment, integrate_public_charging

gc.collect()

num_threads = 1

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

scenarios = [
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    "Mobility_Transition_2050_simbev_run",
    "Szenarette_Kleinwagen_2050_simbev_run",
    "Reference_2050_simbev_run",
    "NEP_C_2035_simbev_run",
]

sub_dir = r"eDisGo_charging_time_series"

grid_ids = ["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb", "grouped", "reduced", "residual"]

grid_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir, grid_id))
    for scenario in scenarios for grid_id in grid_ids
]

for grid_dir in grid_dirs:
    files = os.listdir(grid_dir)

    files.sort()

    grid_id = grid_dir.parts[-1]

    scenario = grid_dir.parts[-3][:-11]

    edisgo = integrate_public_charging(
        ding0_dir,
        grid_dir,
        grid_id,
        files,
        generator_scenario="ego100",
    )

    for strategy in strategies:

        print("breaker")