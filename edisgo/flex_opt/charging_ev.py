import numpy as np
import pandas as pd
import os.path
import traceback
import gc

from edisgo.io.simBEV_import import get_ags
from pathlib import Path
from time import perf_counter


def charging(
        data_dir,
):
    try:
        setup_dict = get_setup(data_dir)

        # get ags numbers
        ags_lst, ags_dirs = get_ags(data_dir)

        for count_ags, ags_dir in enumerate(ags_dirs):
            t1 = perf_counter()
            data_dfs, use_cases = get_ags_data(ags_dir)
            print(
                "It took {} seconds to read in the data in AGS Nr. {}.".format(
                    perf_counter() - t1, ags_dir.parts[-1]
                )
            )

            for count_cases, use_case in enumerate(use_cases):
                for strategy in ["dumb", "reduced"]:
                    if strategy == "dumb" or (strategy == "reduced" and (use_case == "home" or use_case == "work")):
                        t1 = perf_counter()
                        grid_independent_charging(
                            data_dfs[count_cases],
                            setup_dict,
                            use_case,
                            ags_dir,
                            strategy=strategy,
                        )
                        print(
                            "It took {} seconds to generate the time series for".format(round(perf_counter() - t1, 1)),
                            "use case {} in AGS Nr. {} with a {} charging strategy.".format(
                                use_case, ags_dir.parts[-1], strategy,
                            )
                        )
                        gc.collect()

    except:
        traceback.print_exc()


def grid_independent_charging(
        data,
        setup_dict,
        use_case,
        ags_dir,
        strategy="dumb",
):
    try:
        cp_idxs = data.cp_idx.unique().tolist()

        cp_idxs.sort()

        cp_load = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(cp_idxs),
            ),
            dtype=np.float64,
        )

        cp_load[:] = 0

        time_factor = setup_dict["stepsize"] / 60

        if strategy == "dumb":
            data["stop"] = (data.chargingdemand / data.netto_charging_capacity.multiply(time_factor))\
                               .apply(np.ceil).astype(np.int32) + data.charge_start

            for count_cps, cp_idx in enumerate(cp_idxs):
                df_cp = data.copy()[data.cp_idx == cp_idx]
                for cap, start, stop in zip(
                    df_cp.netto_charging_capacity.tolist(),
                    df_cp.charge_start.tolist(),
                    df_cp.stop.tolist(),
                ):
                    cp_load[start:stop, count_cps] += (cap / setup_dict["eta_CP"])

        elif strategy == "reduced":
            data["time"] = data.charge_end - data.charge_start + 1

            data["stop"] = (data.chargingdemand / data.netto_charging_capacity.multiply(time_factor))\
                .apply(np.ceil).astype(np.int32)

            data.chargingdemand = data.stop.multiply(time_factor) * data.netto_charging_capacity

            data["min_cap"] = data.netto_charging_capacity.multiply(0.1)

            data["reduced_min_cap"] = data.chargingdemand / data.time.multiply(time_factor)

            data["reduced_cap"] = data[["min_cap", "reduced_min_cap"]].max(axis=1)

            data["stop"] = (data.chargingdemand / data.reduced_cap.multiply(time_factor))\
                               .round(0).astype(np.int32) + data.charge_start

            for count_cps, cp_idx in enumerate(cp_idxs):
                df_cp = data.copy()[data.cp_idx == cp_idx]

                for cap, start, stop in zip(
                    df_cp.reduced_cap.tolist(),
                    df_cp.charge_start.tolist(),
                    df_cp.stop.tolist(),
                ):
                    cp_load[start:stop, count_cps] += (cap / setup_dict["eta_CP"])

        else:
            raise ValueError("Strategy '{}' does not exist.".format(strategy))

        print(np.sum(cp_load))

        time_series_to_hdf(
            cp_load,
            cp_idxs,
            use_case,
            strategy,
            ags_dir,
        )

    except:
        traceback.print_exc()


def time_series_to_hdf(
        cp_load,
        cp_idxs,
        use_case,
        strategy,
        ags_dir,
):
    try:
        df_load = pd.DataFrame(
            data=cp_load,
            columns=cp_idxs,
        )

        df_load = df_load.iloc[:365*24*4]

        df_load = compress(
            df_load,
            verbose=False,
        )

        file_name = "{}_charging_timeseries_{}.h5".format(strategy, use_case,)

        sub_dir = "eDisGo_timeseries"

        export_path = Path(
            os.path.join(
                ags_dir.parent.parent,
                sub_dir,
                ags_dir.parts[-1],
                file_name,
            )
        )

        os.makedirs(
            export_path.parent,
            exist_ok=True,
        )

        df_load.to_hdf(
            export_path,
            key="df_load",
        )

    except:
        traceback.print_exc()


def get_setup(
        data_dir,
):
    try:
        setup_data = "config_data.csv"

        setup_data_path = Path(
            os.path.join(
                data_dir.parent,
                setup_data,
            )
        )

        setup_dict = pd.read_csv(
            setup_data_path,
            index_col=[0]
        )["0"].to_dict()

        attribute_list = [
            "random.seed",
            "num_threads",
            "stepsize",
            "year",
            "days",
            "car_limit",
            "eta_CP",
        ]

        for attribute in attribute_list:
            if attribute == "eta_CP":
                setup_dict[attribute] = float(setup_dict[attribute])
            else:
                setup_dict[attribute] = int(setup_dict[attribute])

        return setup_dict

    except:
        traceback.print_exc()


def get_ags_data(
        ags_dir
):
    try:
        files = os.listdir(ags_dir)

        files.sort()

        files = files[int(len(files)/2):]

        file_dirs = [
            Path(os.path.join(ags_dir, file)) for file in files
        ]

        use_cases = [
            file.split("_")[-1].split(".")[0] for file in files
        ]

        use_cases = list(dict.fromkeys(use_cases))

        dfs = [0] * len(file_dirs)

        for count_files, file_dir in enumerate(file_dirs):
            df = pd.read_csv(
                file_dir,
                index_col=[0], # this throws a numpy warning https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
                # nrows=100, # TODO: remove after testing
                engine="c",
            )

            df = compress( # FIXME: this saves memory but is pretty slow
                df,
                verbose=False,
            )

            dfs[count_files] = df

        return dfs, use_cases


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

