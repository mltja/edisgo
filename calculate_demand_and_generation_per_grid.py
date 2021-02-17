import gc
import os.path
import numpy as np
import pandas as pd
import logging
import warnings
import traceback

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

num_threads = 1

base_dir = Path( # TODO: set dir
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

grid_ids = ["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb", "grouped", "reduced", "residual"]

data_dirs = [
    Path(os.path.join(base_dir, sub_dir, scenario, grid_id, strategy))
    for grid_id in grid_ids for scenario in scenarios for strategy in strategies
]

idx = pd.MultiIndex.from_product(
    [
        scenarios,
        strategies,
        [
            "demand",
            "generation",
        ],
        [
            "low_residual",
            "high_residual",
        ],
    ]
)

df_demand_load = pd.DataFrame(
    data=np.nan,
    columns=grid_ids,
    index=idx,
)


def calculate_demand_and_generation(
        directory,
):
    try:
        t0 = perf_counter()

        strategy = directory.parts[-1]

        grid_id = directory.parts[-2]

        scenario = directory.parts[-3]

        print("Scenario {} with strategy {} in grid {} is being processed.".format(scenario, strategy, grid_id))

        days = get_days(grid_id)

        edisgo = import_edisgo_from_files(
            directory=directory,
            import_topology=False,
            import_timeseries=True,
            import_results=False,
        )

        generation_low = edisgo.timeseries.generators_active_power.loc[
                         days.start_week_low:days.end_week_low - timedelta(minutes=15)
                         ].sum().sum()

        generation_high = edisgo.timeseries.generators_active_power.loc[
                          days.start_week_high:days.end_week_high - timedelta(minutes=15)
                          ].sum().sum()

        demand_low = edisgo.timeseries.loads_active_power.loc[
                     days.start_week_low:days.end_week_low - timedelta(minutes=15)
                     ].sum().sum() +\
                     edisgo.timeseries.charging_points_active_power.loc[
                     days.start_week_low:days.end_week_low - timedelta(minutes=15)
                     ].sum().sum()

        demand_high = edisgo.timeseries.loads_active_power.loc[
                     days.start_week_high:days.end_week_high - timedelta(minutes=15)
                     ].sum().sum() + \
                     edisgo.timeseries.charging_points_active_power.loc[
                     days.start_week_high:days.end_week_high - timedelta(minutes=15)
                     ].sum().sum()

        df_demand_load.at[(scenario, strategy, "generation", "low_residual"), grid_id] = generation_low
        df_demand_load.at[(scenario, strategy, "generation", "high_residual"), grid_id] = generation_high

        df_demand_load.at[(scenario, strategy, "demand", "low_residual"), grid_id] = demand_low
        df_demand_load.at[(scenario, strategy, "demand", "high_residual"), grid_id] = demand_high

        print(
            "Demand and generation for strategy {} in scenario {} in grid {} has been calculated.".format(
                strategy, scenario, grid_id
            ),
            "It took {} seconds".format(round(perf_counter()-t0, 0))
        )

        del edisgo

    except:
        traceback.print_exc()


def get_days(
        grid_id,
):
    try:
        s = pd.read_csv(
            os.path.join(
                base_dir,
                sub_dir,
                "extreme_weeks.csv",
            ),
            index_col=[0],
            parse_dates=[1,2,3,4],
        ).loc[int(grid_id)]

        return s
    except:
        traceback.print_exc()


if __name__ == "__main__":
    for d in data_dirs:
        calculate_demand_and_generation(
            d,
        )

        gc.collect()

    results_dir = Path(
        os.path.join(
            base_dir,
            "eDisGo_curtailment_analysis",
        )
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    df_demand_load.to_csv(
        os.path.join(
            results_dir,
            "grid_demand_load.csv",
        )
    )

