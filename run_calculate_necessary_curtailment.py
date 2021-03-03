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

os.sched_setaffinity(0,range(1000)) # TODO

num_threads = 1 # TODO

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

sub_dir = r"eDisGo_curtailment_results" # TODO

scenarios = [ # TODO
    "NEP_C_2035",
    # "Reference_2050",
    # "Szenarette_Kleinwagen_2050",
    # "Mobility_Transition_2050",
    # "Electrification_2050",
    # "Electrification_2050_sensitivity_low_work",
]

grid_ids = ["176"]#["2534", "177", "1056", "1690", "1811", "176"] # TODO

strategies = ["residual"]#, "grouped", "reduced", "residual"] # TODO

data_dirs = [
    Path(os.path.join(data_dir, sub_dir, scenario, grid_id, strategy))
    for scenario in scenarios for grid_id in grid_ids for strategy in strategies
]


def run_calculate_curtailment(
        directory,
):
    try:
        strategy = directory.parts[-1]

        grid_id = directory.parts[-2]

        scenario = directory.parts[-3]

        print("Scenario {} with strategy {} in grid {} is being processed.".format(scenario, strategy, grid_id))

        mode = "days" # TODO

        if mode == "days":
            ts_count = 96
        elif mode == "weeks":
            ts_count = 7*96
        else:
            ts_count = None

        if ts_count is not None:
            days = get_days(
                grid_id,
                mode=mode,
            )
        else:
            days = [None]

        for day in days:
            stepwise_curtailment(
                directory,
                day,
                ts_count,
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

        edisgo_chunk.timeseries.residual_load.to_csv(
            os.path.join(directory, "residual_load.csv")
        )

        # FIXME:
        if "type" not in edisgo_chunk.topology.generators_df.columns.tolist():
            edisgo_chunk.topology.generators_df["type"] = ["solar"] * len(
                edisgo_chunk.topology.generators_df
            )

        if len_day is not None:
            timeindex = pd.date_range(
                day,
                periods=len_day,
                freq="15min",
            )

            edisgo_chunk.timeseries.timeindex = timeindex

            edisgo_chunk.timeseries.storage_units_active_power = edisgo_chunk.timeseries.storage_units_active_power.loc[
                edisgo_chunk.timeseries.storage_units_active_power.index.isin(timeindex)
            ]

            edisgo_chunk.timeseries.storage_units_reactive_power = edisgo_chunk.timeseries\
                .storage_units_reactive_power.loc[
                edisgo_chunk.timeseries.storage_units_reactive_power.index.isin(timeindex)
            ]

            edisgo_chunk.timeseries.charging_points_active_power = edisgo_chunk.timeseries\
                .charging_points_active_power.loc[
                edisgo_chunk.timeseries.charging_points_active_power.index.isin(timeindex)
            ]

            edisgo_chunk.timeseries.charging_points_reactive_power = edisgo_chunk.timeseries\
                .charging_points_reactive_power.loc[
                edisgo_chunk.timeseries.charging_points_reactive_power.index.isin(timeindex)
            ]

            edisgo_chunk.timeseries.loads_active_power = edisgo_chunk.timeseries.loads_active_power.round(5).loc[
                                                         :, (
                                                                    edisgo_chunk.timeseries.loads_active_power != 0
                                                            ).any(axis=0)
                                                         ]

            load_new_connectors = edisgo_chunk.timeseries.loads_active_power.columns.tolist()

            edisgo_chunk.topology.loads_df = edisgo_chunk.topology.loads_df.loc[
                edisgo_chunk.topology.loads_df.index.isin(load_new_connectors)
            ]

            drop_cols = [
                col for col in edisgo_chunk.timeseries._loads_reactive_power.columns if col not in load_new_connectors
            ]

            edisgo_chunk.timeseries._loads_reactive_power.drop(
                columns=drop_cols,
                inplace=True,
            )

            edisgo_chunk.topology.loads_df = edisgo_chunk.topology.loads_df.loc[
                edisgo_chunk.topology.loads_df.index.isin(load_new_connectors)
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
    if num_threads == 1:
        for d in data_dirs:
            t0 = perf_counter()

            run_calculate_curtailment(d)

            gc.collect()

            strategy = d.parts[-1]

            grid_id = d.parts[-2]

            scenario = d.parts[-3]

            print(
                "Curtailment for strategy {} in scenario {} in grid {} has been calculated.".format(
                    strategy, scenario, grid_id
                ),
                "It took {} seconds".format(round(perf_counter()-t0, 0))
            )
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(
                run_calculate_curtailment,
                data_dirs,
            )

