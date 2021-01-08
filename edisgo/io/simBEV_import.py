import numpy as np
import pandas as pd
import os.path
import traceback
import glob

from pathlib import Path
from numpy.random import default_rng

def get_ags(
        data_dir,
):
    try:
        ags_dirs = glob.glob(f'{data_dir}/*/')

        ags_dirs = [
            Path(directory) for directory in ags_dirs
        ]

        ags_lst = [
            directory.parts[-1] for directory in ags_dirs
        ]

        return ags_lst, ags_dirs

    except:
        traceback.print_exc()


def get_ags_data(
        ags_dir,
        NumberOfUseCases = 4,
):
    try:
        car_csvs = os.listdir(ags_dir)

        car_csvs = [
            Path(os.path.join(ags_dir, car_csv)) for car_csv in car_csvs
        ]

        cols = [
            "netto_charging_capacity",
            "chargingdemand",
            "charge_start",
            "charge_end",
            "car_idx",
        ]

        use_case_dfs = [
            pd.DataFrame(
                columns=cols,
            ) for _ in range(NumberOfUseCases)
        ]

        for count_cars, car_csv in enumerate(car_csvs[:10]): # TODO: remove limit
            df = pd.read_csv(
                car_csv,
                index_col=[0],
            )

            df["car_idx"] = count_cars

            df_lst = [0] * NumberOfUseCases

            df_lst[0] = df.copy()[
                (df.use_case == "private") &
                (df.location == "6_home")
            ][cols]

            df_lst[1] = df.copy()[
                (df.use_case == "private") &
                (df.location == "0_work")
            ][cols]

            df_lst[2] = df.copy()[
                (df.use_case == "public") &
                (df.location != "7_charging_hub")
                ][cols]

            df_lst[3] = df.copy()[
                df.location == "7_charging_hub"
                ][cols]

            for count, use_case_df in enumerate(use_case_dfs):
                use_case_df = use_case_df.append(
                    df_lst[count],
                    ignore_index=True
                )

                use_case_dfs[count] = use_case_df

        dtypes = [
            np.float32,
            np.float32,
            np.uint32,
            np.uint32,
            np.uint32,
        ]

        for count, use_case_df in enumerate(use_case_dfs):
            for count_cols, col in enumerate(cols):
                use_case_df[col] = use_case_df[col].astype(dtypes[count_cols])

            use_case_dfs[count] = use_case_df

        return tuple(use_case_dfs)

    except:
        traceback.print_exc()


def get_charging_points(
        localiser_data_dir,
        ags_dir,
):
    try:
        files = os.listdir(localiser_data_dir)

        ags = ags_dir.parts[-1]

        files = [
            Path(os.path.join(localiser_data_dir, file)) for file in files if ags in file
        ]

        # uc1-fast/hpc, uc2-public, uc3-home, uc4-work
        files.sort()

        cp_dfs = [0] * len(files)

        for count, file in enumerate(files):
            cp_dfs[count] = pd.read_csv(
                file,
                index_col=[0],
            )

        return tuple(cp_dfs)

    except:
        traceback.print_exc()

def distribute_demand(
        use_case,
        df_standing,
        df_cp,
):
    try:
        if use_case == "home":
            distribute_demand_home(
                df_standing,
                df_cp,
            )
        # elif use_case == "work":
        #     distribute_demand_work(
        #         df_standing,
        #         df_cp,
        #     )
        # elif use_case == "public":
        #     distribute_demand_public(
        #         df_standing,
        #         df_cp,
        #     )
        # else:
        #     distribute_demand_hpc(
        #         df_standing,
        #         df_cp,
        #     )


    except:
        traceback.print_exc()


def distribute_demand_home(
        df_standing,
        df_cp,
):
    try:
        rng = default_rng(
            seed=25588,
        )

        car_idxs = df_standing.car_idx.unique()

        total_population = df_cp.Einwohner.sum()

        max_cp_per_location = int(np.ceil(len(car_idxs) / len(df_cp))) + 1

        cols = [
            "cp_{:02d}".format(x+1) for x in range(max_cp_per_location)
        ]

        df_cp_mapping = df_cp.copy()

        df_cp_mapping = pd.concat(
            [
                df_cp_mapping,
                pd.DataFrame(
                    columns=cols,
                )
            ]
        )

        population_list = df_cp.Einwohner.tolist()

        weights = [
            population/total_population for population in population_list
        ]

        print(sum(weights))

        cp_idxs = df_cp.index.tolist()

        for car_idx in car_idxs:
            df_car = df_standing[
                df_standing.car_idx == car_idx
            ]

            # TODO: check if it is faster with or without this line
            df_standing = df_standing[
                df_standing.car_idx != car_idx
            ]

            cp_idx = rng.choice(
                cp_idxs,
                p=weights,
            )

            # recalculate weights
            population_list[cp_idx] -= (population_list[cp_idx] / max_cp_per_location)

            total_population = sum(population_list)

            weights = [
                population / total_population for population in population_list
            ]

            print(sum(weights))

            cp_number = df_cp_mapping.columns[df_cp_mapping.loc[cp_idx].isna()].tolist()[0]

            df_cp_mapping.at[cp_idx, cp_number] = df_car.iat[0, 0]

            # TODO: store standing times per CP

        df_cp_mapping = df_cp_mapping[df_cp_mapping.cp_01.notna()]

        return df_cp_mapping

    except:
        traceback.print_exc()