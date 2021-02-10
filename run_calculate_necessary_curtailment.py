import gc
import os.path
import pandas as pd
import logging
import warnings
import multiprocessing
import traceback
import calculate_necessary_curtailment as cc
import curtailment as cur

from datetime import datetime, timedelta
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

global edisgo

num_threads = 32

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

scenarios = [
    # "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    # "Mobility_Transition_2050_simbev_run",
    # "Szenarette_Kleinwagen_2050_simbev_run",
    # "Reference_2050_simbev_run",
    # "NEP_C_2035_simbev_run",
]

sub_dir = r"eDisGo_charging_time_series"

grid_ids = ["1056"]#["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb", "grouped", "reduced", "residual"]

grid_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir, grid_id))
    for scenario in scenarios for grid_id in grid_ids
]

def run_calculate_curtailment(
        grid_dir,
        num_threads,
):
    try:
        global edisgo
        
        t0 = perf_counter()

        files = os.listdir(grid_dir)

        files.sort()

        grid_id = grid_dir.parts[-1]

        scenario = grid_dir.parts[-3][:-11]

        print("Scenario {} in grid {} is being processed.".format(scenario, grid_id))

        start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")

        offsets = [*range(365)]

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

            if num_threads == 1:
                for day_offset in offsets:
                    stepwise_curtailment(
                        day_offset,
                        start_date,
                        len(offsets),
                        strategy,
                    )

            else:
                data_tuples = [
                    (day_offset, start_date, len(offsets), strategy)
                    for day_offset in offsets
                ]

                mv_grid_id = int(grid_dir.parts[-1])

                if mv_grid_id == 176:
                    num_threads = 6
                elif mv_grid_id == 177:
                    num_threads = 20
                elif mv_grid_id == 1056:
                    pass
                elif mv_grid_id == 1690:
                    pass
                elif mv_grid_id == 1811:
                    pass
                elif mv_grid_id == 2534:
                    num_threads = 32
                else:
                    num_threads = 2

                with multiprocessing.Pool(num_threads) as pool:
                    pool.starmap(
                        stepwise_curtailment,
                        data_tuples,
                    )

            print(
                "Curtailment for strategy {} in scenario {} in grid {} has been calculated.".format(
                    strategy, day_offset, scenario, grid_id
                ),
                "It took {} seconds".format(round(perf_counter()-t0, 0))
            )

            del edisgo

            gc.collect()

    except:
        traceback.print_exc()


def stepwise_curtailment(
        day_offset,
        date,
        chunks,
        strategy,
):
    try:
        start = date + timedelta(days=int(day_offset * 365 / chunks))

        timeindex = pd.date_range(
            start,
            periods=int(365 / chunks * 24 * 4),
            freq="15min",
        )

        edisgo_chunk = deepcopy(edisgo)

        edisgo_chunk.timeseries.timeindex = timeindex

        edisgo_chunk.timeseries.generation_dispatchable = edisgo_chunk.timeseries.generation_dispatchable.loc[
            (edisgo_chunk.timeseries.generation_dispatchable.index >= timeindex[0]) &
            (edisgo_chunk.timeseries.generation_dispatchable.index <= timeindex[-1])
        ]

        edisgo_chunk.timeseries.generation_fluctuating = edisgo_chunk.timeseries.generation_fluctuating.loc[
            (edisgo_chunk.timeseries.generation_fluctuating.index >= timeindex[0]) &
            (edisgo_chunk.timeseries.generation_fluctuating.index <= timeindex[-1])
        ]

        edisgo_chunk.timeseries.load = edisgo_chunk.timeseries.load.loc[
            (edisgo_chunk.timeseries.load.index >= timeindex[0]) &
            (edisgo_chunk.timeseries.load.index <= timeindex[-1])
        ]

        edisgo_chunk.timeseries.storage_units_active_power = edisgo_chunk.timeseries.storage_units_active_power.loc[
            (edisgo_chunk.timeseries.storage_units_active_power.index >= timeindex[0]) &
            (edisgo_chunk.timeseries.storage_units_active_power.index <= timeindex[-1])
        ]

        edisgo_chunk.timeseries.storage_units_reactive_power = edisgo_chunk.timeseries.storage_units_reactive_power.loc[
            (edisgo_chunk.timeseries.storage_units_reactive_power.index >= timeindex[0]) &
            (edisgo_chunk.timeseries.storage_units_reactive_power.index <= timeindex[-1])
        ]

        gc.collect()

        cur.calculate_curtailment(
            grid_dir,
            edisgo_chunk,
            strategy,
            day_offset,
        )

        del edisgo_chunk

        gc.collect()

    except:
        traceback.print_exc()


if __name__ == "__main__":
    for grid_dir in grid_dirs:
        run_calculate_curtailment(
            grid_dir,
            num_threads,
        )

    # if num_threads == 1:
    #     for grid_dir in [grid_dirs[0]]:
    #         run_calculate_curtailment(grid_dir)
    # else:
    #     with multiprocessing.Pool(num_threads) as pool:
    #         pool.map(
    #             run_calculate_curtailment,
    #             grid_dirs,
    #         )
