import gc
import os.path
import logging
import warnings
import multiprocessing
import traceback
import calculate_necessary_curtailment as cc
import pandas as pd

from datetime import datetime, timedelta
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

num_threads = 4 # TODO

rng = default_rng(seed=5)

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

scenarios = [ # TODO
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    "Reference_2050_simbev_run",
    "NEP_C_2035_simbev_run",
]

# "Mobility_Transition_2050_simbev_run",
# "Szenarette_Kleinwagen_2050_simbev_run",

sub_dir = r"eDisGo_charging_time_series"

grid_ids = ["1056"]#["176", "177", "1056", "1690", "1811", "2534"] # TODO

strategies = ["dumb", "grouped", "reduced", "residual", "no_charging"] # TODO

grid_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir, grid_id))
    for scenario in scenarios for grid_id in grid_ids
]

rng.shuffle(grid_dirs)


def get_days(
        grid_id,
        mode="days",
):
    try:
        s = pd.read_csv(
            os.path.join(
                data_dir,
                # sub_dir,
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

        days = get_days(
            grid_id,
            mode="weeks",
        )

        for count_strategies, strategy in enumerate(strategies):
            t2 = perf_counter()

            export_dir = Path(
                os.path.join(
                    data_dir,
                    "eDisGo_object_files_final", # TODO
                    scenario,
                    grid_id,
                    strategy,
                )
            )

            os.makedirs(
                export_dir,
                exist_ok=True,
            )

            if count_strategies == 0:
                edisgo = cc.integrate_public_charging(
                    ding0_dir,
                    grid_dir,
                    grid_id,
                    files,
                    date=start_date,
                    generator_scenario="ego100",
                    days=days,
                )

                gc.collect()

                print(
                    "Public charging with strategy {} has been integrated for scenario {} in grid {}.".format(
                        strategy, scenario, grid_id
                    ),
                    "It took {} seconds.".format(round(perf_counter() - t0, 0)),
                )

                t1 = perf_counter()

                edisgo, df_matching_home, df_matching_work = cc.integrate_private_charging(
                    edisgo,
                    grid_dir,
                    files,
                    strategy,
                )

                edisgo.topology.lines_df["check"] = 1 / edisgo.topology.lines_df.length.divide(0.001)

                if not edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].empty:
                    edisgo.topology.lines_df[
                        edisgo.topology.lines_df.check > 1
                        ] = edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].assign(
                        length=edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].length.multiply(
                            edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].check
                        ),
                        r=edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].r.multiply(
                            edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].check
                        ),
                        x=edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].x.multiply(
                            edisgo.topology.lines_df[edisgo.topology.lines_df.check > 1].check
                        ),
                    )

                edisgo.topology.lines_df.drop(
                    columns=["check"],
                    inplace=True,
                )

                gc.collect()

                edisgo.aggregate_components()

                edisgo = cc.reinforce_transformers_and_lines(
                    edisgo,
                    by=0.3,  # TODO
                    mode="lv",  # TODO
                )

                gc.collect()

                print(
                    "Private charging has been integrated for",
                    "scenario {} in grid {} with strategy {}.".format(
                        scenario, grid_id, strategy
                    ),
                    "It took {} seconds.".format(round(perf_counter() - t1, 0)),
                )
            elif count_strategies == 4:
                for col in edisgo.timeseries.charging_points_active_power.columns:
                    edisgo.timeseries._charging_points_active_power[col].values[:] = 0
                    edisgo.timeseries.charging_points_active_power[col].values[:] = 0

            else:
                for col in df_matching_home.edisgo_id.tolist():
                    edisgo.timeseries._charging_points_active_power[col].values[:] = 0
                    edisgo.timeseries.charging_points_active_power[col].values[:] = 0
                for col in df_matching_work.edisgo_id.tolist():
                    edisgo.timeseries._charging_points_active_power[col].values[:] = 0
                    edisgo.timeseries.charging_points_active_power[col].values[:] = 0

                ts_files = [
                    Path(os.path.join(grid_dir, f)) for f in files
                    if ("h5" in f and strategy in f and ("home" in f or "work" in f))
                ]

                ts_files.sort()

                for count_files, ts_f in enumerate(ts_files):
                    if count_files == 0:
                        df_matching = df_matching_home.copy()
                    elif count_files == 1:
                        df_matching = df_matching_work.copy()

                    df = pd.read_hdf(
                        ts_f,
                        key="df_load",
                    )

                    temp_timeindex = pd.date_range(
                        "2011-01-01",
                        periods=len(df),
                        freq="15min",
                    )

                    df.index = temp_timeindex

                    timeindex = edisgo.timeseries.timeindex

                    df = df.loc[timeindex].divide(1000)  # kW -> MW

                    for edisgo_id, ags, cp_idx in list(
                        zip(
                            df_matching.edisgo_id.tolist(),
                            df_matching.ags.tolist(),
                            df_matching.cp_idx.tolist(),
                        )
                    ):
                        edisgo.timeseries._charging_points_active_power[edisgo_id] = df.loc[
                                                                                     :, (ags, cp_idx)
                                                                                     ].values
                        edisgo.timeseries.charging_points_active_power[edisgo_id] = df.loc[
                                                                                     :, (ags, cp_idx)
                                                                                     ].values

            t1 = perf_counter()

            edisgo.save(
                directory=export_dir,
            )

            print(
                "Scenario {} in grid {} with strategy {} has been saved.".format(
                    scenario, grid_id, strategy
                ),
                "It took {} seconds.".format(round(perf_counter() - t2, 0)),
            )

            print(
                "Total charging demand in scenario {} in grid {} with strategy {}: {} MWh".format(
                    scenario, grid_id, strategy, round(edisgo.timeseries.charging_points_active_power.sum().sum()/4, 0)
                )
            )

            gc.collect()

        print(
            "Scenario {} in grid {} has been saved.".format(
                scenario, grid_id
            ),
            "It took {} seconds".format(round(perf_counter() - t0, 0))
        )

        del edisgo

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
