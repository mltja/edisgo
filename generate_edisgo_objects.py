import gc
import os.path
import logging
import warnings
import multiprocessing
import traceback
import calculate_necessary_curtailment as cc

from datetime import datetime
from pathlib import Path
from time import perf_counter
from numpy.random import default_rng

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

gc.collect()

num_threads = 10

rng = default_rng(seed=5)

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
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

rng.shuffle(grid_dirs)

def generate_edisgo_objects(
        grid_dir,
):
    try:
        t0 = perf_counter()

        grid_id = grid_dir.parts[-1]

        scenario = grid_dir.parts[-3][:-11]

        data_dir = grid_dir.parent.parent.parent

        files = os.listdir(grid_dir)

        files.sort()

        print("Scenario {} in grid {} is being processed.".format(scenario, grid_id))

        start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")

        for strategy in strategies:
            t1 = perf_counter()

            edisgo = cc.integrate_public_charging(
                ding0_dir,
                grid_dir,
                grid_id,
                files,
                date=start_date,
                generator_scenario="ego100",
            )

            gc.collect()

            print(
                "Public charging has been integrated in scenario {} in grid {}.".format(
                    scenario, grid_id
                ),
                "It took {} seconds.".format(round(perf_counter() - t1, 0)),
            )

            t1 = perf_counter()

            edisgo = cc.integrate_private_charging(
                edisgo,
                grid_dir,
                files,
                strategy,
            )

            gc.collect()

            print(
                "Private charging has been integrated for",
                "scenario {} in grid {} with strategy {}.".format(
                    scenario, grid_id, strategy
                ),
                "It took {} seconds.".format(round(perf_counter() - t1, 0)),
            )

            t1 = perf_counter()

            export_dir = Path(
                os.path.join(
                    data_dir,
                    "eDisGo_curtailment_results",
                    "scenario",
                    "grid_id",
                    "strategy",
                )
            )

            os.makedirs(
                export_dir,
                exist_ok=True,
            )

            edisgo.save(
                directory=export_dir,
            )

            print(
                "Scenario {} in grid {} with strategy {} has been saved.".format(
                    scenario, grid_id, strategy
                ),
                "It took {} seconds.".format(round(perf_counter() - t1, 0)),
            )

            del edisgo

            gc.collect()

        print(
            "Scenario {} in grid {} has been saved.".format(
                strategy, scenario, grid_id
            ),
            "It took {} seconds".format(round(perf_counter() - t0, 0))
        )

    except:
        traceback.print_exc()


if __name__ == "__main__":
    if num_threads == 1:
        for grid_dir in grid_dirs:
            generate_edisgo_objects(
                grid_dir,
            )
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(
                generate_edisgo_objects,
                grid_dirs,
            )
