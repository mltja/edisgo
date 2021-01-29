import gc
import os.path
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path

gc.collect()

num_threads = 1

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
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

grid_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir, grid_id))
    for scenario in scenarios for grid_id in grid_ids
]

strategies = ["dumb", "grouped", "reduced", "residual"]

use_cases = ["home", "work"]

ts_data_paths = [
    Path(os.path.join(grid_dir, "{}_charging_timeseries_{}.h5".format(strategy, use_case)))
    for grid_dir in grid_dirs for use_case in use_cases for strategy in strategies
]

rows = [
    (scenario, grid_id, use_case, strategy)
    for scenario in scenarios for grid_id in grid_ids for use_case in use_cases for strategy in strategies
]

cols = ["max_simultaneousness", "e_sum"]

arr_base = np.empty(
    shape=(
        len(cols),
        len(rows),
    ),
    dtype=float,
)

arr_base[:] = 0

df_plau = pd.DataFrame(
    data=arr_base,
)

df_plau.columns = pd.MultiIndex.from_tuples(rows)

df_plau = df_plau.T

df_plau.rename_axis(['scenario', 'grid_id', "strategy", "use_case"]),

df_plau.columns = cols

for count, path in enumerate(ts_data_paths):
    file = path.parts[-1]
    grid_id = path.parts[-2]
    if "home" in file:
        gdf = gpd.read_file(
            os.path.join(
                path.parent,
                "cp_data_home_within_grid_{}.geojson".format(grid_id)
            )
        )
    else:
        gdf = gpd.read_file(
            os.path.join(
                path.parent,
                "cp_data_work_within_grid_{}.geojson".format(grid_id)
            )
        )

    df = pd.read_hdf(
        path,
        key="df_load",
    )

    df_max = pd.DataFrame(
        df.max().rename_axis(['ags', 'cp_idx']),
        columns=["p_max"],
    )

    df_simultan = gdf[["ags", "cp_idx", "cp_capacity", "cp_connection_rating"]].merge(
        df_max,
        left_on=['ags', 'cp_idx'],
        right_index=True,

    )

    df_simultan = df_simultan.assign(
        simultaneousness=(df_simultan.p_max / df_simultan.cp_connection_rating * 100).round(1)
    )

    df_simultan = df_simultan[df_simultan.cp_capacity >= 300]

    max_sim = df_simultan.simultaneousness.max()

    e_sum = df.sum().sum()

    df_plau.iloc[count] = [max_sim, e_sum]

    print("{} %".format(round((count+1)/len(ts_data_paths) * 100, 1)))

    if count % 10 == 0:
        gc.collect()

df_plau.to_csv(
    os.path.join(
        data_dir,
        r"plausibility_check.csv",
    )
)