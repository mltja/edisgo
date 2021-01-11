import numpy as np
import pandas as pd
import os.path
import traceback
import glob
import gc

from pathlib import Path
from numpy.random import default_rng


def run_simBEV_import(
        data_dir,
        localiser_data_dir,
):
    try:
        # get ags numbers
        ags_lst, ags_dirs = get_ags(data_dir)

        for count, ags_dir in enumerate(ags_dirs):

            print("AGS Nr. {} is processing.".format(ags_dir.parts[-1]))

            (df_standing_times_home, df_standing_times_work,
             df_standing_times_public, df_standing_times_hpc) = get_ags_data(ags_dir)

            df_cp_hpc, df_cp_public, df_cp_home, df_cp_work = get_charging_points(
                localiser_data_dir,
                ags_dir,
            )

            use_cases = [
                "home",
                "work",
                "public",
                "hpc",
            ]

            for use_case in use_cases:
                if use_case == "home" and len(df_standing_times_home) > 0:
                    distribute_demand(
                        use_case,
                        df_standing_times_home,
                        df_cp_home,
                        ags_dir,
                    )
                    del df_standing_times_home, df_cp_home
                elif use_case == "work" and len(df_standing_times_work) > 0:
                    distribute_demand(
                        use_case,
                        df_standing_times_work,
                        df_cp_work,
                        ags_dir,
                    )
                    del df_standing_times_work, df_cp_work
                elif use_case == "public" and len(df_standing_times_public) > 0:
                    distribute_demand(
                        use_case,
                        df_standing_times_public,
                        df_cp_public,
                        ags_dir,
                    )
                    del df_standing_times_public, df_cp_public
                elif use_case == "hpc" and len(df_standing_times_hpc) > 0:
                    distribute_demand(
                        use_case,
                        df_standing_times_hpc,
                        df_cp_hpc,
                        ags_dir,
                    )
                    del df_standing_times_hpc, df_cp_hpc
                else:
                    pass

            gc.collect()
            print("AGS Nr. {} has been processed.".format(ags_dir.parts[-1]))
            break  # TODO: remove this when all data is available
    except:
        traceback.print_exc()


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
        NumberOfUseCases=4,
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

        # for count_cars, car_csv in enumerate([car_csv for count, car_csv in enumerate(car_csvs) if count in [317, 771, 955]]):
        for count_cars, car_csv in enumerate(car_csvs):#[50:200]):  # TODO: remove limit
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

            print(count_cars/len(car_csvs)*100)

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

            if count >= 2:
                use_case_df = use_case_df.sort_values(
                    by=["charge_start"],
                    ascending=True,
                )

                use_case_df.reset_index(
                    drop=True,
                    inplace=True,
                )

            use_case_dfs[count] = use_case_df

        print("Standing times for AGS Nr. {} have been read in.".format(ags_dir.parts[-1]))

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
                dtype={
                    "Einwohner": np.uint32,
                }
            )

            cp_dfs[count] = cp_dfs[count].sort_values(
                by=["Einwohner"],
                ascending=False,
            )

        return tuple(cp_dfs)

    except:
        traceback.print_exc()


def distribute_demand(
        use_case,
        df_standing,
        df_cp,
        ags_dir,
):
    try:
        export = False
        if use_case == "home" or use_case == "work" and len(df_standing) > 0:
            df_standing, df_cp = distribute_demand_private(
                df_standing,
                df_cp,
            )
            export = True
        elif use_case == "public" or use_case == "hpc" and len(df_standing) > 0:
            df_standing, df_cp = distribute_demand_public(
                df_standing,
                df_cp,
                use_case,
            )
            export = True
        else:
            pass
            # print("Use case '{}' wasn't found".format(use_case))

        if export:
            data_to_csv(
                use_case,
                df_standing,
                df_cp,
                ags_dir,
            )

        print("Demand for use case {} for AGS Nr. {} has been distributed.".format(use_case, ags_dir.parts[-1]))

    except:
        traceback.print_exc()


def distribute_demand_private(
        df_standing,
        df_cp,
):
    try:
        rng = default_rng(
            seed=25588,
        )

        df_standing, df_cp, weights, car_idxs, cp_idxs, population_list, max_cp_per_location = data_preprocessing(
            df_standing,
            df_cp,
        )

        population_list_init = population_list.copy()

        for car_idx in car_idxs:
            cp_idx, weights, total_population, population_list = get_weighted_rnd_cp(
                rng,
                weights,
                cp_idxs,
                population_list,
                population_list_init,
                max_cp_per_location,
            )
            # try:
            cp_number = df_cp.columns[df_cp.loc[cp_idx].isna()].tolist()[0]
            # except:
            #     a = df_cp.loc[cp_idx]
            #     traceback.print_exc()
            #     print("breaker")

            df_cp.at[cp_idx, cp_number] = df_standing[
                df_standing.car_idx == car_idx
                ].iat[0, 0]

            df_standing[df_standing.car_idx == car_idx] = df_standing[df_standing.car_idx == car_idx].assign(
                cp_idx=cp_idx
            )

        return df_standing, df_cp

    except:
        traceback.print_exc()


def distribute_demand_public(
        df_standing,
        df_cp,
        use_case,
        utilisation=0.8,
):
    try:
        rng = default_rng(
            seed=25588,
        )

        df_standing, df_cp, weights, _, cp_idxs, population_list, max_cp_per_location = data_preprocessing(
            df_standing,
            df_cp,
            use_case=use_case,
        )

        population_list_init = population_list.copy()

        cols = [
            "cp_idx",
            "netto_charging_capacity",
            "ts_last",
        ]

        df_generated_cps = pd.DataFrame(
            columns=cols,
        )

        if use_case == "hpc" and len(df_standing) > 0:
            count_150 = len(
                df_standing[
                    (df_standing.netto_charging_capacity > 120) &
                    (df_standing.netto_charging_capacity < 150)
                ]
            )
            count_350 = len(
                df_standing[
                    (df_standing.netto_charging_capacity > 150)
                    ]
            )

            if count_350 == 0:
                prob_150 = 1
                prob_350 = 0
            elif count_150 == 0:
                prob_150 = 0
                prob_350 = 1
            else:
                prob_150 = count_150 / (count_150 + count_350)
                prob_350 = 1 - prob_150

        for standing_idx, cap in df_standing.netto_charging_capacity.iteritems():
            start = df_standing.charge_start.at[standing_idx]

            if cap > 100 and cap < 120:
                cap = rng.choice(
                    [150, 350],
                    p=[prob_150, prob_350]
                ) * 0.9 # FIXME: automate this to use the right efficiency eta_cp

            df_matching = df_generated_cps.copy()[
                (df_generated_cps.netto_charging_capacity.round(0) == round(cap, 0)) &
                (df_generated_cps.ts_last < start)
                ]

            last_ts = get_last_ts(
                df_standing,
                standing_idx,
                start,
                utilisation,
            )

            if len(df_matching) > 0:
                cp_idx = int(df_matching.iat[0, 0])

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                df_generated_cps.loc[
                    (df_generated_cps.cp_idx == cp_idx)
                ] = df_generated_cps.loc[
                    (df_generated_cps.cp_idx == cp_idx)
                ].assign(
                    ts_last=last_ts,
                )

            else:
                cp_idx, weights, total_population, population_list = get_weighted_rnd_cp(
                    rng,
                    weights,
                    cp_idxs,
                    population_list,
                    population_list_init,
                    max_cp_per_location,
                    use_case=use_case,
                    capacity=cap,
                )

                try:
                    cp_number = df_cp.columns[df_cp.loc[cp_idx].isna()].tolist()[0]
                except:
                    a = df_cp.loc[cp_idx]
                    traceback.print_exc()
                    print("breaker")


                df_cp.at[cp_idx, cp_number] = cap

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                s = pd.Series(
                    [
                        cp_idx,
                        cap,
                        last_ts,
                    ],
                    index=cols,
                )

                df_generated_cps = df_generated_cps.append(
                    s,
                    ignore_index=True,
                )

        return df_standing, df_cp

    except:
        traceback.print_exc()


def get_last_ts(
        df_standing,
        standing_idx,
        start,
        utilisation,
):
    try:
        stop = df_standing.charge_end.at[standing_idx]

        standing_time = stop - start + 1

        last_ts = max(
            int(stop + (1 - utilisation) * standing_time),
            int(stop + 1),
        )

        return last_ts

    except:
        traceback.print_exc()


def get_weighted_rnd_cp(
        rng,
        weights,
        cp_idxs,
        population_list,
        population_list_init,
        max_cp_per_location,
        use_case=None,
        capacity=None,
):
    try:
        if use_case == "hpc" and capacity > 150:
            weights_350 = weights.copy()
            weights_350.reverse()
            cp_idx = rng.choice(
                cp_idxs,
                p=weights_350,
            )
        else:
            cp_idx = rng.choice(
                cp_idxs,
                p=weights,
            )

        # recalculate weights
        population_list[cp_idx] = max(
            population_list[cp_idx] - np.ceil(population_list_init[cp_idx] / max_cp_per_location),
            0,
        )

        total_population = sum(population_list)

        weights = [
            population / total_population for population in population_list
        ]

        return cp_idx, weights, total_population, population_list

    except:
        traceback.print_exc()


def data_preprocessing(
        df_standing,
        df_cp,
        use_case = None,
):
    try:
        car_idxs = df_standing.car_idx.unique()

        total_population = df_cp.Einwohner.sum()

        if use_case == "public":
            max_cp_per_location = min(
                max(
                    len(car_idxs),
                    4,
                ),
                8,
            )
        elif use_case == "hpc":
            max_cp_per_location = min(
                max(
                    len(car_idxs),
                    4,
                ),
                16,
            )
        else:
            max_cp_per_location = max(
                int(np.ceil(len(car_idxs) / len(df_cp))),
                1,
            ) * 2

        cols = [
            "cp_{:02d}".format(x + 1) for x in range(max_cp_per_location)
        ]

        df_cp = pd.concat(
            [
                df_cp,
                pd.DataFrame(
                    columns=cols,
                )
            ],
            sort=False,
        )

        df_cp.Einwohner = df_cp.Einwohner.astype(np.int32)

        cp_idxs = df_cp.index.tolist()

        df_standing["cp_idx"] = np.nan

        population_list = df_cp.Einwohner.tolist()

        weights = [
            population / total_population for population in population_list
        ]

        return df_standing, df_cp, weights, car_idxs, cp_idxs, population_list, max_cp_per_location

    except:
        traceback.print_exc()


def data_to_csv(
        use_case,
        df_standing,
        df_cp,
        ags_dir,
):
    try:
        df_cp = df_cp.drop(
            [
                "Einwohner",
            ],
            axis="columns",
        )
        df_cp = df_cp.dropna(
            axis="columns",
            how="all",
        )
        df_cp = df_cp[df_cp.cp_01.notna()].fillna(0)
        df_cp = df_cp.sort_index()

        df_standing.cp_idx = df_standing.cp_idx.astype(np.int32)
        df_standing = df_standing.drop(
            [
                "car_idx",
            ],
            axis="columns",
        )

        ags = ags_dir.parts[-1]
        data_dir = ags_dir.parent.parent

        export_dir = Path(
            os.path.join(
                data_dir,
                "cp_standing_times_mapping",
                ags,
            )
        )

        os.makedirs(
            export_dir,
            exist_ok=True,
        )

        files = [
            r"cp_data_{}.csv".format(use_case),
            r"cp_standing_times_mapping_{}.csv".format(use_case),
        ]

        for count, file in enumerate(files):
            if count == 0:
                export_df = df_cp.copy()
            elif count == 1:
                export_df = df_standing.copy()

            export_path = os.path.join(
                export_dir,
                file,
            )

            export_df.to_csv(
                export_path,
            )

        print("Data has been exported for AGS Nr. {}.".format(ags_dir.parts[-1]))
    except:
        traceback.print_exc()

