import numpy as np
import pandas as pd
import geopandas as gpd
import os.path
import traceback
import glob
import gc

from pathlib import Path
from numpy.random import default_rng
from time import perf_counter


def run_simBEV_import(
        data_dir,
        localiser_data_dir,
):
    try:
        # get ags numbers
        ags_lst, ags_dirs = get_ags(data_dir)

        for count, ags_dir in enumerate(ags_dirs):

            print("AGS Nr. {} is being processed in scenario {}.".format(ags_dir.parts[-1], ags_dir.parts[-3]))

            t1 = perf_counter()

            df_standing_times_home, df_standing_times_work, \
                df_standing_times_public, df_standing_times_hpc = get_ags_data(ags_dir)

            print(
                "It took {} seconds to read in the data in AGS Nr. {}.".format(
                    perf_counter() - t1, ags_dir.parts[-1]
                )
            )

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
                t1 = perf_counter()
                if use_case == "home" and len(df_standing_times_home) > 0:
                    distribute_demand(
                        use_case,
                        df_standing_times_home,
                        df_cp_home,
                        ags_dir,
                    )
                    del df_standing_times_home, df_cp_home
                elif use_case == "work" and len(df_standing_times_work) > 0:
                    if len(df_cp_work) > 0:
                        distribute_demand(
                            use_case,
                            df_standing_times_work,
                            df_cp_work,
                            ags_dir,
                        )
                    else:
                        # use public charging points
                        distribute_demand(
                            use_case,
                            df_standing_times_work,
                            df_cp_public,
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
                    if len(df_cp_hpc) > 0:
                        distribute_demand(
                            use_case,
                            df_standing_times_hpc,
                            df_cp_hpc,
                            ags_dir,
                        )
                    else:
                        # use public charging points
                        distribute_demand(
                            use_case,
                            df_standing_times_hpc,
                            df_cp_public,
                            ags_dir,
                        )
                    del df_standing_times_hpc, df_cp_hpc
                else:
                    print("Use case {} is not in AGS Nr. {}.".format(use_case, ags_dir.parts[-1]))

                print(
                    "It took {} seconds to distribute the demand for use case {} in AGS Nr. {}.".format(
                        perf_counter() - t1, use_case, ags_dir.parts[-1]
                    )
                )

            gc.collect()
            print("AGS Nr. {} has been processed in scenario {}.".format(ags_dir.parts[-1], ags_dir.parts[-3]))
            # break  # TODO: remove this when all data is available
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
        for count_cars, car_csv in enumerate(car_csvs):#[:10]):  # TODO: remove limit
            df = pd.read_csv(
                car_csv,
                index_col=[0],
                engine="c",
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

            if count >= 2:
                use_case_df = use_case_df.sort_values(
                    by=["charge_start"],
                    ascending=True,
                )

                use_case_df.reset_index(
                    drop=True,
                    inplace=True,
                )

            use_case_dfs[count] = compress(
                use_case_df,
                verbose=False,
            )

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
            cp_dfs[count] = gpd.read_file(
                file,
            )

            if count == 0 and len(cp_dfs[count]) > 0:
                cp_dfs[count] = cp_dfs[count].drop(
                    [
                        "name",
                    ],
                    axis="columns",
                )
                # TODO: hpc has no weights @Johannes
            elif count == 1 and len(cp_dfs[count]) > 0:
                cp_dfs[count] = cp_dfs[count].rename(
                    columns={
                        "sum_pois": "weight",
                    }
                )
                cp_dfs[count].weight = cp_dfs[count].weight.astype(np.int32)
                cp_dfs[count] = cp_dfs[count].sort_values(
                    by=["weight"],
                    ascending=False,
                )
            elif count == 2 and len(cp_dfs[count]) > 0:
                cp_dfs[count] = cp_dfs[count].rename(
                    columns={
                        "building_wohneinheiten": "weight",
                    }
                )
                cp_dfs[count].weight = cp_dfs[count].weight.astype(np.int32)
                cp_dfs[count] = cp_dfs[count].sort_values(
                    by=["weight"],
                    ascending=False,
                )
            elif count == 3 and len(cp_dfs[count]) > 0:
                cp_dfs[count] = cp_dfs[count].drop(
                    [
                        "landuse",
                        "area",
                    ],
                    axis="columns",
                )
                cp_dfs[count].weight = cp_dfs[count].weight.astype(np.float32)
                cp_dfs[count] = cp_dfs[count].sort_values(
                    by=["weight"],
                    ascending=False,
                )
            else:
                pass

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
        if (use_case == "home" or use_case == "work") and len(df_standing) > 0:
            df_standing, df_cp = distribute_demand_private(
                df_standing,
                df_cp,
            )
            export = True
        elif use_case == "public" and len(df_standing) > 0:
            df_standing, df_cp = distribute_demand_public(
                df_standing,
                df_cp,
                use_case,
            )
            export = True
        elif use_case == "hpc" and len(df_standing) > 0:
            df_standing, df_cp = distribute_demand_hpc(
                df_standing,
                df_cp,
                use_case,
            )
            export = True
        else:
            pass

        if export:
            data_to_hdf(
                use_case,
                df_standing,
                df_cp,
                ags_dir,
            )

            # print("Demand for use case {} for AGS Nr. {} has been distributed.".format(use_case, ags_dir.parts[-1]))

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

        df_standing, df_cp, weights, car_idxs, cp_idxs, weight_list, max_cp_per_location = data_preprocessing(
            df_standing,
            df_cp,
        )

        weight_list_init = weight_list.copy()

        for count, car_idx in enumerate(car_idxs):
            cp_idx, weights, weight_list = get_weighted_rnd_cp(
                rng,
                cp_idxs,
                max_cp_per_location,
                weight_list,
                weight_list_init,
                weights,
            )

            cp_number = df_cp.columns[df_cp.loc[cp_idx].isna()].tolist()[0]

            df_cp.at[cp_idx, cp_number] = df_standing[
                df_standing.car_idx == car_idx
                ].iat[0, 0]


            df_standing[df_standing.car_idx == car_idx] = df_standing[df_standing.car_idx == car_idx].assign(
                cp_idx=cp_idx
            )

        return compress(df_standing, verbose=False), compress(df_cp, verbose=False)

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

        df_standing, df_cp, weights, _, cp_idxs, weight_list, max_cp_per_location = data_preprocessing(
            df_standing,
            df_cp,
            use_case=use_case,
        )

        weight_list_init = weight_list.copy()

        cols = [
            "cp_idx",
            "netto_charging_capacity",
            "ts_last",
        ]

        df_generated_cps = pd.DataFrame(
            columns=cols,
        )

        for standing_idx, cap in df_standing.netto_charging_capacity.iteritems():
            start = df_standing.charge_start.at[standing_idx]

            df_matching = df_generated_cps[
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
                row_matching = df_matching.iloc[0]

                cp_idx = int(row_matching.iat[0])

                matching_idx = row_matching.name

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                df_generated_cps.at[matching_idx, "ts_last"] = last_ts

            else:
                cp_idx, weights, weight_list = get_weighted_rnd_cp(
                    rng,
                    cp_idxs,
                    max_cp_per_location,
                    weight_list,
                    weight_list_init,
                    weights,
                )

                cp_number = df_cp.columns[df_cp.loc[cp_idx].isna()].tolist()[0]

                df_cp.at[cp_idx, cp_number] = cap

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                df_generated_cps.loc[len(df_generated_cps)] = [
                    cp_idx,
                    cap,
                    last_ts,
                ]

        return compress(df_standing, verbose=False), compress(df_cp, verbose=False)

    except:
        traceback.print_exc()


def distribute_demand_hpc(
        df_standing,
        df_cp,
        use_case,
        utilisation=0.8,
):
    try:
        rng = default_rng(
            seed=25588,
        )

        #df_standing, df_cp, weights, _, cp_idxs, weight_list, max_cp_per_location = data_preprocessing(
        df_standing, df_cp, car_idxs, cp_idxs, max_cp_per_location = data_preprocessing(
            df_standing,
            df_cp,
            use_case=use_case,
        )

        # weight_list_init = weight_list.copy()

        cols = [
            "cp_idx",
            "netto_charging_capacity",
            "ts_last",
        ]

        df_generated_cps = pd.DataFrame(
            columns=cols,
        )

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
                row_matching = df_matching.iloc[0]

                cp_idx = int(row_matching.iat[0])

                matching_idx = row_matching.name

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                df_generated_cps.at[matching_idx, "ts_last"] = last_ts

            else:
                # cp_idx, weights, weight_list = get_weighted_rnd_cp(
                #     rng,
                #     cp_idxs,
                #     max_cp_per_location,
                #     weight_list,
                #     weight_list_init,
                #     weights,
                # )

                cp_idx = rng.choice(
                    cp_idxs,
                )

                cp_number = df_cp.columns[df_cp.loc[cp_idx].isna()].tolist()[0]

                df_cp.at[cp_idx, cp_number] = cap

                df_standing.at[standing_idx, "cp_idx"] = cp_idx

                df_generated_cps.loc[len(df_generated_cps)] = [
                    cp_idx,
                    cap,
                    last_ts,
                ]

        return compress(df_standing, verbose=False), compress(df_cp, verbose=False)

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
        cp_idxs,
        max_cp_per_location,
        weight_list,
        weight_list_init,
        weights,
):
    try:
        if sum(weights) < 0.9:
            cp_idx = rng.choice(
                cp_idxs,
            )
        else:
            cp_idx = rng.choice(
                cp_idxs,
                p=weights,
            )

        weight_idx = cp_idxs.index(cp_idx)

        # recalculate weights
        weight_list[weight_idx] = max(
            weight_list[weight_idx] - np.ceil((weight_list_init[weight_idx] / max_cp_per_location)*1000)/1000,
            0,
        )

        total_weight = max(
            sum(weight_list),
            1,
        )

        weights = [
            weight / total_weight for weight in weight_list
        ]

        if sum(weights) > 0:
            # sometimes the sum is too far away from 1 and needs to be standardized
            weights = [
                weight / sum(weights) for weight in weights
            ]

        return cp_idx, weights, weight_list

    except:
        traceback.print_exc()


def data_preprocessing(
        df_standing,
        df_cp,
        use_case=None,
):
    try:
        car_idxs = df_standing.car_idx.unique()

        if use_case == "public" or use_case == "hpc":
            # max_cp_per_location = len(car_idxs) * 0.2
            max_cp_per_location = max(
                int(np.ceil(len(car_idxs) / len(df_cp))),
                1,
            ) * 5
        else:
            max_cp_per_location = max(
                int(np.ceil(len(car_idxs) / len(df_cp))),
                1,
            ) * 3

        cols = [
            "cp_{:05d}".format(x + 1) for x in range(max_cp_per_location)
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

        cp_idxs = df_cp.index.tolist()

        df_standing["cp_idx"] = np.nan

        if use_case != "hpc":
            total_weight = df_cp.weight.sum()

            # df_cp.weight = df_cp.weight.astype(np.int32)

            weight_list = df_cp.weight.tolist()

            weights = [
                weight / total_weight for weight in weight_list
            ]

            if sum(weights) > 0:
                weights = [
                    weight * 1 / (sum(weights)) for weight in weights
                ]

            return df_standing, df_cp, weights, car_idxs, cp_idxs, weight_list, max_cp_per_location

        else:
            return df_standing, df_cp, car_idxs, cp_idxs, max_cp_per_location

    except:
        traceback.print_exc()


def data_to_hdf(
        use_case,
        df_standing,
        df_cp,
        ags_dir,
):
    try:
        if "weight" in df_cp.columns:
            df_cp = df_cp.drop(
                [
                    "weight",
                ],
                axis="columns",
            )
        df_cp = df_cp.dropna(
            axis="columns",
            how="all",
        )
        df_cp = df_cp[df_cp.cp_00001.notna()].fillna(0)
        df_cp = df_cp.sort_index()
        df_cp.reset_index(inplace=True)

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
            r"cp_data_{}.geojson".format(use_case),
            r"cp_standing_times_mapping_{}.h5".format(use_case),
        ]

        for count, file in enumerate(files):
            export_path = os.path.join(
                export_dir,
                file,
            )

            if count == 0:
                df_cp.to_file(
                    export_path,
                    driver="GeoJSON",
                )
            elif count == 1:
                df_standing.to_hdf(
                    path_or_buf=export_path,
                    key="export_df",
                    mode="w",
                    format="table",
                )

        # print("Data for use case {} has been exported for AGS Nr. {}.".format(use_case, ags_dir.parts[-1]))
    except:
        traceback.print_exc()

def compress(
        df,
        verbose=True,
):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

