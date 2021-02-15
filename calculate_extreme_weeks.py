import gc
import os.path
import numpy as np
import pandas as pd
import logging
import warnings
import multiprocessing
import traceback
import curtailment as cur

from edisgo import EDisGo
from datetime import datetime, timedelta
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

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

grid_ids = ["2534"]#["176", "177", "1056", "1690", "1811", "2534"]

grid_dirs = [
    Path(os.path.join(ding0_dir, grid_id)) for grid_id in grid_ids
]

global df_weeks

df_weeks = pd.DataFrame(
    data=np.nan,
    columns=[
        "start_week_low",
        "end_week_low",
        "start_week_high",
        "end_week_high",
    ],
    index=grid_ids,
)

def calculate_extreme_weeks(
        grid_dir,
):
    try:
        global df_weeks

        timeindex = pd.date_range(
            '2011-01-01',
            periods=8760,
            freq='H',
        )

        p_bio = 9983  # MW
        e_bio = 50009  # GWh

        vls_bio = e_bio / (p_bio / 1000)

        share = vls_bio / 8760

        timeseries_generation_dispatchable = pd.DataFrame(
            {
                "biomass": [share] * len(timeindex),
                "coal": [1] * len(timeindex),
                "other": [1] * len(timeindex),
            },
            index=timeindex,
        )

        generator_scenario="ego100"

        edisgo = EDisGo(
            ding0_grid=grid_dir,
            generator_scenario=generator_scenario,
            timeseries_load="demandlib",
            timeseries_generation_fluctuating="oedb",
            timeseries_generation_dispatchable=timeseries_generation_dispatchable,
            timeindex=timeindex,
        )

        df_residual_resample = edisgo.timeseries.residual_load.copy().resample("W-FRI").mean()

        min_idx_end = df_residual_resample.iloc[:-1].idxmin() + timedelta(days=1)

        min_idx_start = min_idx_end - timedelta(weeks=1)

        max_idx_end = df_residual_resample.iloc[:-1].idxmax() + timedelta(days=1)

        max_idx_start = max_idx_end - timedelta(weeks=1)

        df_weeks.at[grid_dir.parts[-1], "start_week_low"] = min_idx_start
        df_weeks.at[grid_dir.parts[-1], "end_week_low"] = min_idx_end
        df_weeks.at[grid_dir.parts[-1], "start_week_high"] = max_idx_start
        df_weeks.at[grid_dir.parts[-1], "end_week_high"] = max_idx_end

    except:
        traceback.print_exc()


if __name__ == "__main__":
    for d in grid_dirs:
        calculate_extreme_weeks(
            d,
        )

    df_weeks.to_csv(
        os.path.join(
            data_dir,
            "eDisGo_curtailment_results",
            "extreme_weeks.csv",
        )
    )

