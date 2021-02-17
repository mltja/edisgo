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
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
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

grid_ids = ["2534", "file_count",]#["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb", "grouped", "reduced", "residual"]

sub_sub_dir = r"curtailment_weeks"

data_dirs = [
    Path(os.path.join(base_dir, sub_dir, scenario, grid_id, strategy, sub_sub_dir))
    for grid_id in grid_ids for scenario in scenarios for strategy in strategies
]

idx = pd.MultiIndex.from_product(
    [
        scenarios,
        strategies,
        [
            "curtailment_demand_convergence",
            "curtailment_demand_lv",
            "curtailment_demand_mv",
            "curtailment_generation_convergence",
            "curtailment_generation_lv",
            "curtailment_generation_mv",
        ],
        [
            "low_residual",
            "high_residual",
        ],
    ]
)

df_curtailment = pd.DataFrame(
    data=np.nan,
    columns=grid_ids,
    index=idx,
)


def join_curtailment(
        directory,
):
    try:
        t0 = perf_counter()

        strategy = directory.parts[-2]

        grid_id = directory.parts[-3]

        scenario = directory.parts[-4]

        print("Scenario {} with strategy {} in grid {} is being processed.".format(scenario, strategy, grid_id))

        low_rl_days, high_rl_days = get_days(grid_id)

        files = os.listdir(directory)

        cur_files = [f for f in files if not "_ts_" in f]

        cur_files.sort()

        low_rl_files = [
            os.path.join(directory, f) for f in cur_files if any([day in f for day in low_rl_days])
        ]
        high_rl_files = [
            os.path.join(directory, f) for f in cur_files if any([day in f for day in high_rl_days])
        ]

        print("breaker")

        # df_demand_load.at[(scenario, strategy, "generation", "low_residual"), grid_id] = generation_low
        # df_demand_load.at[(scenario, strategy, "generation", "high_residual"), grid_id] = generation_high
        #
        # df_demand_load.at[(scenario, strategy, "demand", "low_residual"), grid_id] = demand_low
        # df_demand_load.at[(scenario, strategy, "demand", "high_residual"), grid_id] = demand_high

        print(
            "Demand and generation for strategy {} in scenario {} in grid {} has been calculated.".format(
                strategy, scenario, grid_id
            ),
            "It took {} seconds".format(round(perf_counter()-t0, 0))
        )

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

        low_rl_days = []
        high_rl_days = []

        for i in range(7):
            low_rl_days.append(s.start_week_low + timedelta(days=i))
            high_rl_days.append(s.start_week_high + timedelta(days=i))

        low_rl_days = [day.strftime("%Y-%m-%d") for day in low_rl_days]
        high_rl_days = [day.strftime("%Y-%m-%d") for day in high_rl_days]

        return low_rl_days, high_rl_days
    except:
        traceback.print_exc()


if __name__ == "__main__":
    for d in data_dirs:
        join_curtailment(
            d,
        )

        gc.collect()

    results_dir = Path(
        os.path.join(
            base_dir,
            "eDisGo_curtailment_analysis",
            "2534"
        )
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    df_curtailment.to_csv(
        os.path.join(
            results_dir,
            "curtailment.csv",
        )
    )

