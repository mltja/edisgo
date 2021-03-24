import gc
import os.path
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from statistics import mean, stdev

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

sub_dir = r"eDisGo_charging_time_series"

grid_ids = ["176", "177", "1056", "1690", "1811", "2534"]

strategies = ["dumb"]#, "grouped", "reduced", "residual"]

use_cases = [
    "home",
    "work",
    "public",
    "hpc",
]

df_connection_ratings = pd.DataFrame(
    data=0,
    columns=[
        "count_charging_parks",
        "mean_connection_rating",
        "std_connection_rating",
        "max_connection_rating",
    ],
    index=use_cases,
)


if __name__ == "__main__":
    if num_threads == 1:
        scenarios = [
            "Electrification_2050_simbev_run",
            "Electrification_2050_sensitivity_low_work_simbev_run",
            "Mobility_Transition_2050_simbev_run",
            "Szenarette_Kleinwagen_2050_simbev_run",
            "Reference_2050_simbev_run",
            "NEP_C_2035_simbev_run",
        ]

        for scenario in scenarios:
            grid_dirs = [
                Path(os.path.join(data_dir, scenario, sub_dir, grid_id))
                for grid_id in grid_ids
            ]

            home_ratings = []
            work_ratings = []
            public_ratings = []
            hpc_ratings = []

            for grid_dir in grid_dirs:
                files = os.listdir(grid_dir)

                files.sort()

                geo_files = [
                    Path(os.path.join(grid_dir, f)) for f in files
                    if "geojson" in f
                ]

                if len(geo_files) == 4:
                    use_cases_grid = [
                        "home",
                        "hpc",
                        "public",
                        "work",
                    ]
                else:
                    use_cases_grid = [
                        "home",
                        "public",
                        "work",
                    ]

                for geo_f, use_case in list(
                        zip(
                            geo_files,
                            use_cases_grid,
                        )
                ):
                    gdf = gpd.read_file(
                        geo_f,
                    )

                    cp_connection_ratings = gdf.cp_connection_rating.tolist()

                    if use_case == "home":
                        home_ratings.extend(cp_connection_ratings)
                    elif use_case == "work":
                        work_ratings.extend(cp_connection_ratings)
                    elif use_case == "public":
                        public_ratings.extend(cp_connection_ratings)
                    else:
                        hpc_ratings.extend(cp_connection_ratings)

            df_connection_ratings.count_charging_parks = [
                len(home_ratings),
                len(work_ratings),
                len(public_ratings),
                len(hpc_ratings),
            ]

            df_connection_ratings.mean_connection_rating = [
                mean(home_ratings),
                mean(work_ratings),
                mean(public_ratings),
                mean(hpc_ratings),
            ]

            df_connection_ratings.std_connection_rating = [
                stdev(home_ratings),
                stdev(work_ratings),
                stdev(public_ratings),
                stdev(hpc_ratings),
            ]

            df_connection_ratings.max_connection_rating = [
                max(home_ratings),
                max(work_ratings),
                max(public_ratings),
                max(hpc_ratings),
            ]

            df_connection_ratings.to_csv("connection_ratings_{}.csv".format(scenario[:-11]))

            print("done")

