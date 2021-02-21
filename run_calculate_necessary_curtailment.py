import gc
import os.path
import pandas as pd
import logging
import warnings
import multiprocessing
import traceback
import curtailment as cur

from datetime import timedelta
from pathlib import Path
from time import perf_counter
from edisgo.edisgo import import_edisgo_from_files


# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

gc.collect()

num_threads = 2

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

sub_dir = r"eDisGo_curtailment_results"

scenarios = [
    "NEP_C_2035",
    "Reference_2050",
    "Szenarette_Kleinwagen_2050",
    "Mobility_Transition_2050",
    "Electrification_2050",
    "Electrification_2050_sensitivity_low_work",
]

grid_ids = ["2534", "177", "1056", "1690", "1811", "176"]

strategies = ["dumb", "grouped", "reduced", "residual"]

data_dirs = [
    Path(os.path.join(data_dir, sub_dir, scenario, grid_id, strategy))
    for grid_id in grid_ids for scenario in scenarios for strategy in strategies
]


def run_calculate_curtailment(
        directory,
        num_threads,
):
    try:
        t0 = perf_counter()

        strategy = directory.parts[-1]

        grid_id = directory.parts[-2]

        scenario = directory.parts[-3]

        print("Scenario {} with strategy {} in grid {} is being processed.".format(scenario, strategy, grid_id))

        mode = "days"

        days = get_days(
            grid_id,
            mode=mode, # TODO
        )

        if mode == "days":
            ts_count = 96
        elif mode == "weeks":
            ts_count = 7*96

        # days = pd.date_range(
        #     '2011-01-01',
        #     periods=365/5, # TODO
        #     freq='5d',
        # ).tolist()

        if num_threads == 1:
            for day in days:
                stepwise_curtailment(
                    directory,
                    day,
                    ts_count,
                )

        else:
            if grid_id == "176":
                num_threads = 11
            elif grid_id == "177":
                num_threads = 21
            elif grid_id == "1056":
                num_threads = 14
            elif grid_id == "1690":
                num_threads = 12
            elif grid_id == "1811":
                num_threads = 6
            elif grid_id == "2534":
                num_threads = 32
            else:
                num_threads = 2

            num_threads = min(num_threads, len(days), 7) # TODO

            data_tuples = [
                (directory, day, ts_count)
                for day in days
            ]

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
        day,
        len_day,
):
    try:
        t1 = perf_counter()

        strategy = directory.parts[-1]

        edisgo_chunk = import_edisgo_from_files(
            directory=directory,
            import_topology=True,
            import_timeseries=True,
            import_results=True,
        )

        timeindex = pd.date_range(
            day,
            periods=len_day,
            freq="15min",
        )

        # FIXME:
        edisgo_chunk.topology.generators_df["type"] = ["solar"] * len(
            edisgo_chunk.topology.generators_df
        )

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

        print(
            "EDisGo Object for day {} has been loaded.".format(
                day,
            ),
            "It took {} seconds.".format(round(perf_counter() - t1, 0)),
        )

        t1 = perf_counter()

        cur.calculate_curtailment(
            directory,
            edisgo_chunk,
            strategy,
            day,
        )

        del edisgo_chunk

        gc.collect()

        print(
            "Curtailment for day {} has been calculated.".format(
                day
            ),
            "It took {} seconds.".format(round(perf_counter() - t1, 0)),
        )

    except:
        traceback.print_exc()


def get_days(
        grid_id,
        mode="days",
):
    try:
        s = pd.read_csv(
            os.path.join(
                data_dir,
                sub_dir,
                "extreme_weeks.csv",
            ),
            index_col=[0],
            parse_dates=[1,2,3,4],
        ).loc[int(grid_id)]

        if mode == "days":
            days = []

            for ts in [s.start_week_low, s.start_week_high]:
                for i in range(7):
                    days.append(ts + timedelta(days=i))

        elif mode == "weeks":
            days = [s.start_week_low, s.start_week_high]

        return days
    except:
        traceback.print_exc()


if __name__ == "__main__":
    for d in data_dirs:
        run_calculate_curtailment(
            d,
            num_threads,
        )

        break # TODO

