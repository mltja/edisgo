import gc
import os.path
import logging
import warnings
import multiprocessing
import traceback
import calculate_necessary_curtailment as cc

from copy import deepcopy
from pathlib import Path

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

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

def run_calculate_curtailment(
        grid_dir,
):
    try:
        files = os.listdir(grid_dir)

        files.sort()

        grid_id = grid_dir.parts[-1]

        scenario = grid_dir.parts[-3][:-11]

        edisgo = cc.integrate_public_charging(
            ding0_dir,
            grid_dir,
            grid_id,
            files,
            generator_scenario="ego100",
        )

        gc.collect()

        for strategy in strategies:
            edisgo_strategy = deepcopy(edisgo)

            edisgo_strategy = cc.integrate_private_charging(
                edisgo_strategy,
                grid_dir,
                files,
                strategy,
            )

            cc.calculate_curtailment(
                grid_dir,
                edisgo_strategy,
                strategy,
            )

            print("breaker")

            del edisgo_strategy

            gc.collect()

    except:
        traceback.print_exc()


if __name__ == "__main__":
    if num_threads == 1:
        for grid_dir in grid_dirs:
            run_calculate_curtailment(grid_dir)
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(
                run_calculate_curtailment,
                grid_dirs,
            )
