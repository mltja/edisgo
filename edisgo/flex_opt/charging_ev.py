import numpy as np
import pandas as pd
import geopandas as gpd
import os.path
import traceback
import gc

from edisgo import EDisGo
from edisgo.io.simBEV_import import get_ags, get_grid_data, hasNumbers
from pathlib import Path
from time import perf_counter


def charging(
        data_dir,
        ding0_dir,
):
    try:
        setup_dict = get_setup(data_dir)

        df_grid_data = get_grid_data()

        for grid_idx, grid_id in df_grid_data.grid_id.iteritems():

            print("Grid Nr. {} in scenario {} is being processed.".format(grid_id, data_dir.parts[-2]))

            ags_list = df_grid_data.ags.at[grid_idx]

            ags_list.sort()

            ags_dirs = [
                Path(os.path.join(data_dir, ags)) for ags in ags_list
            ]

            t1 = perf_counter()

            gdf_cps_total, df_standing_total = get_grid_cps_and_charging_processes(
                ags_dirs,
                setup_dict["eta_CP"],
            )

            print("It took {} seconds to read in the data for grid {}.".format(
                round(perf_counter() - t1, 1), grid_id
            ))

            t1 = perf_counter()

            edisgo, gdf_cps_total = integrate_cps(
                gdf_cps_total,
                ding0_dir,
                grid_id,
                worst_case_analysis="worst-case",
                # generator_scenario="ego100",
            )

            gc.collect()

            print("It took {} seconds to generate the eDisGo object for grid {}.".format(
                round(perf_counter() - t1, 1), grid_id
            ))

            use_cases = gdf_cps_total.use_case.unique().tolist()

            use_cases.sort()

            for count_cases, use_case in enumerate(use_cases):
                df_standing_data = df_standing_total.copy()[df_standing_total.use_case == use_case]
                gdf_cp_data = gdf_cps_total.copy()[gdf_cps_total.use_case == use_case]

                gdf_to_geojson(
                    gdf_cp_data,
                    use_case,
                    grid_id,
                    data_dir,
                )

                for strategy in ["grouped"]:#["dumb", "reduced", "grouped"]:
                    if strategy == "dumb" or (strategy == "reduced" and (use_case == 3 or use_case == 4)):
                        t1 = perf_counter()
                        grid_independent_charging(
                            df_standing_data,
                            gdf_cp_data,
                            setup_dict,
                            use_case,
                            grid_id,
                            data_dir,
                            strategy=strategy,
                        )
                        print(
                            "It took {} seconds to generate the time series for".format(round(perf_counter() - t1, 1)),
                            "use case {} in grid {} with charging strategy {}.".format(
                                get_use_case_name(use_case), grid_id, strategy,
                            )
                        )
                        gc.collect()
                    elif strategy == "grouped" and (use_case == 3 or use_case == 4):
                        t1 = perf_counter()
                        grouped_charging(
                            edisgo,
                            df_standing_data,
                            gdf_cp_data,
                            setup_dict,
                            use_case,
                            grid_id,
                            data_dir,
                            strategy=strategy,
                        )
                        print(
                            "It took {} seconds to generate the time series for".format(round(perf_counter() - t1, 1)),
                            "use case {} in grid {} with charging strategy {}.".format(
                                get_use_case_name(use_case), grid_id, strategy,
                            )
                        )
                        gc.collect()

    except:
        traceback.print_exc()


def grouped_charging(
        edisgo,
        df_standing,
        gdf_cps,
        setup_dict,
        use_case,
        grid_id,
        data_dir,
        strategy="grouped",
):
    try:
        gdf_cps = pd.merge(
            edisgo.topology.charging_points_df["bus"],
            gdf_cps,
            left_index=True,
            right_on="edisgo_id",
        )

        for bus in gdf_cps.bus.unique():
            df_bus = gdf_cps.copy()[gdf_cps.bus == bus]



            print("breaker")

    except:
        traceback.print_exc()


def integrate_cps(
        gdf_cps_total,
        ding0_dir,
        grid_id,
        worst_case_analysis="worst-case",
        generator_scenario=None,
):
    try:
        edisgo = EDisGo(
            ding0_grid=os.path.join(
                ding0_dir, str(grid_id)
            ),
            worst_case_analysis=worst_case_analysis,
            generator_scenario=generator_scenario,
        )

        time_idx = edisgo.timeseries.timeindex

        comp_type = "ChargingPoint"

        ts_active_power = pd.Series(
            data=[0, 0],
            index=time_idx,
            name="ts_active_power",
        )

        ts_reactive_power = pd.Series(
            data=[0, 0],
            index=time_idx,
            name="ts_reactive_power",
        )

        cp_edisgo_id = [
            EDisGo.integrate_component(
                edisgo,
                comp_type=comp_type,
                geolocation=geolocation,
                voltage_level=None,
                mode="mv",
                add_ts=True,
                ts_active_power=ts_active_power,
                ts_reactive_power=ts_reactive_power,
                p_nom=p_nom,
            ) for geolocation, p_nom in list(
                zip(
                    gdf_cps_total.geometry.tolist(),
                    gdf_cps_total.cp_capacity.divide(1000).tolist(), # kW -> MW
                )
            )
        ]

        gdf_cps_total.insert(0, "edisgo_id", cp_edisgo_id)

        return edisgo, gdf_cps_total

    except:
        traceback.print_exc()


def grid_independent_charging(
        df_standing,
        gdf_cps_total,
        setup_dict,
        use_case,
        grid_id,
        data_dir,
        strategy="dumb",
):
    try:
        cp_ags_list = list(
            zip(
                gdf_cps_total.ags.tolist(),
                gdf_cps_total.cp_idx.tolist(),
            )
        )

        cp_ags_list_unique = list(set(cp_ags_list))

        cp_ags_list_unique.sort()

        cp_load = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(cp_ags_list_unique),
            ),
            dtype=np.float64,
        )

        cp_load[:] = 0

        time_factor = setup_dict["stepsize"] / 60

        if strategy == "dumb":
            df_standing = df_standing.assign(
                stop=(df_standing.chargingdemand / df_standing.netto_charging_capacity.multiply(time_factor))\
                         .apply(np.ceil).astype(np.int32) + df_standing.charge_start
            )

            for count_cps, (ags, cp_idx) in enumerate(cp_ags_list_unique):
                df_cp = df_standing.copy()[
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ]
                for cap, start, stop in list(
                    zip(
                        df_cp.netto_charging_capacity.tolist(),
                        df_cp.charge_start.tolist(),
                        df_cp.stop.tolist(),
                    )
                ):
                    cp_load[start:stop, count_cps] += (cap / setup_dict["eta_CP"])

        elif strategy == "reduced":
            df_standing = df_standing.assign(
                time=df_standing.charge_end - df_standing.charge_start + 1
            )

            df_standing = df_standing.assign(
                stop=(df_standing.chargingdemand / df_standing.netto_charging_capacity.multiply(time_factor))\
                    .apply(np.ceil).astype(np.int32)
            )

            df_standing = df_standing.assign(
                chargingdemand=df_standing.stop.multiply(time_factor) * df_standing.netto_charging_capacity
            )

            df_standing = df_standing.assign(
                min_cap=df_standing.netto_charging_capacity.multiply(0.1)
            )

            df_standing = df_standing.assign(
                reduced_min_cap=df_standing.chargingdemand / df_standing.time.multiply(time_factor)
            )

            df_standing = df_standing.assign(
                reduced_cap=df_standing[["min_cap", "reduced_min_cap"]].max(axis=1)
            )

            df_standing = df_standing.assign(
                stop=(df_standing.chargingdemand / df_standing.reduced_cap.multiply(time_factor))\
                         .round(0).astype(np.int32) + df_standing.charge_start
            )

            for count_cps, (ags, cp_idx) in enumerate(cp_ags_list_unique):
                df_cp = df_standing.copy()[
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ]

                for cap, start, stop in list(
                    zip(
                        df_cp.reduced_cap.tolist(),
                        df_cp.charge_start.tolist(),
                        df_cp.stop.tolist(),
                    )
                ):
                    cp_load[start:stop, count_cps] += (cap / setup_dict["eta_CP"])

        else:
            raise ValueError("Strategy '{}' does not exist.".format(strategy))

        df_standing["time"] = (df_standing.chargingdemand / df_standing.netto_charging_capacity.divide(4)).apply(np.ceil).astype(int)
        df_standing["up"] = df_standing.time * df_standing.netto_charging_capacity.divide(4)

        print("before:", df_standing.up.sum() / 0.9)
        print("after:", np.sum(cp_load) / 4)

        time_series_to_hdf(
            cp_load,
            cp_ags_list_unique,
            use_case,
            strategy,
            grid_id,
            data_dir,
        )

    except:
        traceback.print_exc()


def time_series_to_hdf(
        cp_load,
        cp_idxs,
        use_case,
        strategy,
        grid_id,
        data_dir,
):
    try:
        df_load = pd.DataFrame(
            data=cp_load,
            # columns=cp_idxs,
        )

        df_load.columns = pd.MultiIndex.from_tuples(cp_idxs)

        df_load = df_load.iloc[:365*24*4]

        df_load = compress(
            df_load,
            verbose=False,
        )

        use_case_name = get_use_case_name(use_case)

        file_name = "{}_charging_timeseries_{}.h5".format(strategy, use_case_name)

        sub_dir = "eDisGo_timeseries"

        export_path = Path(
            os.path.join(
                data_dir.parent,
                sub_dir,
                str(grid_id),
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
            mode="w",
        )

    except:
        traceback.print_exc()


def gdf_to_geojson(
        gdf,
        use_case,
        grid_id,
        data_dir,
):
    try:
        use_case_name = get_use_case_name(use_case)

        file_name = "cp_data_{}_within_grid_{}.geojson".format(use_case_name, grid_id)

        sub_dir = "eDisGo_timeseries"

        export_path = Path(
            os.path.join(
                data_dir.parent,
                sub_dir,
                str(grid_id),
                file_name,
            )
        )

        os.makedirs(
            export_path.parent,
            exist_ok=True,
        )

        gdf = gdf.loc[:, (gdf != 0).any(axis=0)]

        gdf.pop("use_case")

        gdf = compress(
            gdf.copy(),
            verbose=False,
        )

        gdf.to_file(
            export_path,
            driver="GeoJSON",
        )

    except:
        traceback.print_exc()


def get_grid_cps_and_charging_processes(
        ags_dirs,
        eta_cp,
):
    try:
        df_standing_total = pd.DataFrame()

        gdf_cps_total = gpd.GeoDataFrame()

        for ags_dir in ags_dirs:
            files = os.listdir(ags_dir)

            cp_files = [
                file for file in files if "geojson" in file if hasNumbers(file)
            ]

            use_cases = [
                file.split("_")[2] for file in cp_files
            ]

            cp_paths = [
                Path(os.path.join(ags_dir, file)) for file in cp_files
            ]

            standing_times_paths = [
                Path(os.path.join(ags_dir, file)) for file in files if "h5" in file
            ]

            for count_use_cases, use_case in enumerate(use_cases):
                if use_case == "hpc":
                    use_case_nr = 1
                elif use_case == "public":
                    use_case_nr = 2
                elif use_case == "home":
                    use_case_nr = 3
                elif use_case == "work":
                    use_case_nr = 4
                else:
                    raise ValueError("Use case {} is not valid.".format(use_case))

                gdf_cps = gpd.read_file(
                    cp_paths[count_use_cases],
                )

                if "index" in gdf_cps.columns:
                    gdf_cps = gdf_cps.rename(
                        columns={
                            "index": "cp_idx",
                        }
                    )

                gdf_cps["use_case"] = use_case_nr
                gdf_cps["ags"] = int(ags_dir.parts[-1])

                cps_list = gdf_cps.cp_idx.tolist()

                # FIXME: I forgot the data_columns=True flag in the last run
                # FIXME: in new runs the hdf file can be queried directly by using "where=..."
                df_standing = pd.read_hdf(
                    standing_times_paths[count_use_cases],
                    key="export_df",
                )

                df_standing = df_standing[df_standing.cp_idx.isin(cps_list)]

                df_standing["use_case"] = use_case_nr
                df_standing["ags"] = int(ags_dir.parts[-1])

                df_standing_total = pd.concat(
                    [df_standing_total, df_standing],
                    axis = 0,
                    ignore_index=True,
                )

                gdf_cps_total = pd.concat(
                    [gdf_cps_total, gdf_cps],
                    axis=0,
                    ignore_index=True,
                    sort=False,
                )

        gdf_cps_total = gdf_cps_total.fillna(0)
        df_standing_total = df_standing_total.fillna(0)

        ags_series = gdf_cps_total.pop("ags")
        geometry_series = gdf_cps_total.pop("geometry")

        gdf_cps_total.insert(0, "ags", ags_series)

        gdf_cps_total = gdf_cps_total.assign(
            geometry=geometry_series,
        )

        cols = gdf_cps_total.columns.difference(["ags", "cp_idx", "geometry", "use_case"])

        cp_count = gdf_cps_total[cols].gt(0).sum(axis=1)

        gdf_cps_total.insert(2, "cp_count", cp_count)

        cp_capacity = gdf_cps_total[cols].sum(axis=1).divide(eta_cp)

        gdf_cps_total.insert(3, "cp_capacity", cp_capacity)

        use_case_series = gdf_cps_total.pop("use_case")

        gdf_cps_total.insert(1, "use_case", use_case_series)

        gdf_cps_total.cp_capacity = gdf_cps_total.cp_capacity.round(1)

        df_standing_total = df_standing_total.sort_values(
            by=["ags", "use_case", "cp_idx"],
            ascending=True,
        )
        df_standing_total.reset_index(
            drop=True,
            inplace=True,
        )

        gdf_cps_total = gdf_cps_total.sort_values(
            by=["ags", "use_case", "cp_idx"],
            ascending=True,
        )
        gdf_cps_total.reset_index(
            drop=True,
            inplace=True,
        )

        gdf_cps_total = compress(gdf_cps_total, verbose=False)

        return compress(gdf_cps_total, verbose=False), compress(df_standing_total, verbose=False)

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

        files = [
            file for file in files if "h5" in file
        ]

        file_dirs = [
            Path(os.path.join(ags_dir, file)) for file in files
        ]

        use_cases = [
            file.split("_")[-1].split(".")[0] for file in files
        ]

        use_cases = list(dict.fromkeys(use_cases))

        dfs = [0] * len(file_dirs)

        for count_files, file_dir in enumerate(file_dirs):
            df = pd.read_hdf(
                file_dir,
                key="export_df",
            )

            df = compress(
                df,
                verbose=False,
            )

            dfs[count_files] = df

        return dfs, use_cases

    except:
        traceback.print_exc()


def get_use_case_name(
        use_case_nr,
):
    if use_case_nr == 1:
        use_case_name = "hpc"
    elif use_case_nr == 2:
        use_case_name = "public"
    elif use_case_nr == 3:
        use_case_name = "home"
    elif use_case_nr == 4:
        use_case_name = "work"
    else:
        raise ValueError("Use case {} is not valid.".format(use_case_nr))

    return use_case_name


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

