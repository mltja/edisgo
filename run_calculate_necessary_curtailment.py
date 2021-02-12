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
from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files


# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

gc.collect()

global edisgo

num_threads = 2

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

sub_dir = r"eDisGo_curtailment_results"

scenarios = [
    # "NEP_C_2035",
    # "Reference_2050",
    # "Szenarette_Kleinwagen_2050",
    # "Mobility_Transition_2050",
    # "Electrification_2050",
    "Electrification_2050_sensitivity_low_work",
]

grid_ids = ["1056"]#["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb"]#, "grouped", "reduced", "residual"]

data_dirs = [
    Path(os.path.join(data_dir, sub_dir, scenario, grid_id, strategy))
    for scenario in scenarios for grid_id in grid_ids for strategy in strategies
]


def run_calculate_curtailment(
        directory,
        num_threads,
):
    try:
        global edisgo

        t0 = perf_counter()

        strategy = directory.parts[-1]

        grid_id = directory.parts[-2]

        scenario = directory.parts[-3]

        print("Scenario {} with strategy {} in grid {} is being processed.".format(scenario, strategy, grid_id))

        start_date = datetime.strptime("2011-01-01", "%Y-%m-%d")

        edisgo = import_edisgo_from_files(
            directory=directory,
            import_topology=True,
            import_timeseries=True,
            import_results=True,
        )

        print(
            "EDisGo Object for scenario {} with strategy {} in grid {} has been loaded.".format(
                scenario, strategy, grid_id
            ),
            "It took {} seconds.".format(round(perf_counter() - t0, 0)),
        )

        offsets = [*range(73)]

        if num_threads == 1:
            for day_offset in offsets:
                stepwise_curtailment(
                    directory,
                    day_offset,
                    start_date,
                    len(offsets),
                    strategy,
                )

        else:
            data_tuples = [
                (directory, day_offset, start_date, len(offsets), strategy)
                for day_offset in offsets
            ]

            if grid_id == 176:
                num_threads = 5
            elif grid_id == 177:
                num_threads = 12
            elif grid_id == 1056:
                num_threads = 2
            elif grid_id == 1690:
                num_threads = 2
            elif grid_id == 1811:
                num_threads = 2
            elif grid_id == 2534:
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
                strategy, scenario, grid_id
            ),
            "It took {} seconds".format(round(perf_counter()-t0, 0))
        )

        gc.collect()

    except:
        traceback.print_exc()


def stepwise_curtailment(
        directory,
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
            directory,
            edisgo_chunk,
            strategy,
            day_offset,
        )

        del edisgo_chunk

        gc.collect()

    except:
        traceback.print_exc()


if __name__ == "__main__":
    for d in data_dirs:
        run_calculate_curtailment(
            d,
            num_threads,
        )

