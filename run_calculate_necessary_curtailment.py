import gc
import os.path
import logging
import warnings
import multiprocessing
import traceback
import calculate_necessary_curtailment as cc
import curtailment as cur

from datetime import datetime, timedelta
from random import shuffle
from copy import deepcopy
from pathlib import Path
from time import perf_counter

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

gc.collect()

num_threads = 1

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

shuffle(grid_dirs) # mix memory intense scenarios with not so intense scenarios

def run_calculate_curtailment(
        grid_dir,
):
    try:
        t1 = perf_counter()

        files = os.listdir(grid_dir)

        files.sort()

        grid_id = grid_dir.parts[-1]

        scenario = grid_dir.parts[-3][:-11]

        print("Scenario {} in grid {} is being processed.".format(scenario, grid_id))

        start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")

        offsets = [*range(73)]

        for day_offset in offsets:
            date = start_date + timedelta(days=int(day_offset*365/len(offsets)))

            edisgo = cc.integrate_public_charging(
                ding0_dir,
                grid_dir,
                grid_id,
                files,
                date=date,
                chunks=len(offsets),
                generator_scenario="ego100",
            )

            gc.collect()

            print("Public charging has been integrated for chunk Nr. {} in scenario {} in grid {}.".format(
                day_offset, scenario, grid_id
            ))

            for strategy in strategies:
                edisgo_strategy = deepcopy(edisgo)

                edisgo_strategy = cc.integrate_private_charging(
                    edisgo_strategy,
                    grid_dir,
                    files,
                    strategy,
                )

                gc.collect()

                cur.calculate_curtailment(
                    grid_dir,
                    edisgo_strategy,
                    strategy,
                    day_offset,
                )

                del edisgo_strategy

                gc.collect()

                print("Curtailment for strategy {} and chunk Nr. {} in scenario {} in grid {} has been calculated.".format(
                    strategy, day_offset, scenario, grid_id
                ))

            del edisgo

            gc.collect()

            print("Curtailment for Chunk Nr. {} in scenario {} in grid {} has been calculated.".format(
                day_offset, scenario, grid_id
            ))

        print("It took {} seconds for scenario {} in grid {}.".format(
            round(perf_counter()-t1, 1), scenario, grid_id
        ))

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
