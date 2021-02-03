import numpy as np
import pandas as pd
import geopandas as gpd
import os.path
import traceback
import gc

from edisgo import EDisGo
from edisgo.io.simBEV_import import get_grid_data, hasNumbers
from pathlib import Path
from time import perf_counter
from itertools import cycle
from datetime import timedelta


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

            # FIXME
            df_standing_total.netto_charging_capacity = df_standing_total.netto_charging_capacity.astype(float) \
                .divide(setup_dict["eta_CP"]).round(1).multiply(setup_dict["eta_CP"])

            print("It took {} seconds to read in the data for grid {} for scenario {}.".format(
                round(perf_counter() - t1, 1), grid_id, data_dir.parts[-2]
            ))

            t1 = perf_counter()

            edisgo, gdf_cps_total, s_residual_load = integrate_cps(
                gdf_cps_total,
                ding0_dir,
                grid_id,
                worst_case_analysis="worst-case",
                generator_scenario="ego100",
            )

            gc.collect()

            print("It took {} seconds to generate the eDisGo object for grid {} for scenario {}.".format(
                round(perf_counter() - t1, 1), grid_id, data_dir.parts[-2]
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

                for strategy in ["dumb", "reduced", "grouped"]:
                    if strategy == "dumb" or (strategy == "reduced" and (use_case == 3 or use_case == 4)):
                        t1 = perf_counter()
                        grid_independent_charging(
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
                            "use case {} in grid {} for scenario {} with charging strategy {}.".format(
                                get_use_case_name(use_case), grid_id, data_dir.parts[-2], strategy,
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
                            "use case {} in grid {} for scenario {} with charging strategy {}.".format(
                                get_use_case_name(use_case), grid_id, data_dir.parts[-2], strategy,
                            )
                        )
                        gc.collect()

            del edisgo

            df_residual_load = get_residual_load_with_evs(
                s_residual_load,
                setup_dict,
                data_dir,
                grid_id,
            )

            gc.collect()

            strategy = "residual"

            use_cases = [3, 4]

            df_standing_data = df_standing_total.copy()[df_standing_total.use_case.isin(use_cases)]
            gdf_cp_data = gdf_cps_total.copy()[gdf_cps_total.use_case.isin(use_cases)]

            t1 = perf_counter()
            residual_load_charging(
                df_standing_data,
                df_residual_load,
                gdf_cp_data,
                setup_dict,
                grid_id,
                data_dir,
                strategy=strategy,
            )
            print(
                "It took {} seconds to generate the time series for".format(round(perf_counter() - t1, 1)),
                "grid {} for scenario {} with charging strategy {}.".format(
                    grid_id, data_dir.parts[-2], strategy,
                )
            )
            gc.collect()

            # break # TODO

    except:
        traceback.print_exc()


def residual_load_charging(
        df_standing,
        df_residual_load,
        gdf_cps,
        setup_dict,
        grid_id,
        data_dir,
        strategy="residual",
):
    try:
        df_standing, cp_ags_list_home, cp_ags_list_work, cp_load_home, cp_load_work = data_preprocessing_residual(
            df_standing,
            gdf_cps,
            setup_dict,
        )

        arr_residual = np.copy(df_residual_load.residual_unflex.to_numpy())

        for cap_kw, cap_mw, start, stop, use_case, t_need, t_flex, cp_count in list(
            zip(
                df_standing.cap.tolist(),
                df_standing.cap.divide(1000).tolist(),
                df_standing.charge_start.tolist(),
                (df_standing.charge_end + 1).tolist(),
                df_standing.use_case.tolist(),
                df_standing.time_needed.tolist(),
                df_standing.time.tolist(),
                df_standing.cp_mapping.tolist(),
            )
        ):
            if t_flex > 0:
                # print(start, stop, t_need, t_flex)
                idx_ts = np.argpartition(arr_residual[start:stop], t_need-1)[:t_need] + start

                arr_residual[idx_ts] += cap_mw

                if use_case == 3:
                    cp_load_home[idx_ts, cp_count] += cap_kw
                elif use_case == 4:
                    cp_load_work[idx_ts, cp_count] += cap_kw
                else:
                    raise ValueError("Use case {} is not in residual load charging.".format(use_case))

            else:
                arr_residual[start:start + t_need] += cap_mw

                if use_case == 3:
                    cp_load_home[start:start + t_need, cp_count] += cap_kw
                elif use_case == 4:
                    cp_load_work[start:start + t_need, cp_count] += cap_kw
                else:
                    raise ValueError("Use case {} is not in residual load charging.".format(use_case))

        # df_standing["time"] = (df_standing.chargingdemand / df_standing.netto_charging_capacity.divide(4))\
        #     .astype(float).round(1).apply(np.ceil).astype(int)
        # df_standing["up"] = df_standing.time * df_standing.netto_charging_capacity.divide(4)
        #
        # print("before:", df_standing.up.sum() / 0.9)
        print(f"{round((np.sum(cp_load_home) + np.sum(cp_load_work)) / 4, 0)} kWh",
              f"in grid {str(grid_id)} and scenario {data_dir.parts[-2]}.")

        df_residual_load = df_residual_load.assign(
            flex_residual=arr_residual
        )

        export_path = os.path.join(
            data_dir.parent,
            "eDisGo_charging_time_series",
            str(grid_id),
            "residual_load.csv",
        )

        df_residual_load.to_csv(export_path)

        time_series_to_hdf(
            cp_load_home,
            cp_ags_list_home,
            3,
            strategy,
            grid_id,
            data_dir,
        )

        time_series_to_hdf(
            cp_load_work,
            cp_ags_list_work,
            4,
            strategy,
            grid_id,
            data_dir,
        )

    except:
        traceback.print_exc()


def data_preprocessing_residual(
        df_standing,
        gdf_cps,
        setup_dict,
):
    try:
        time_factor = setup_dict["stepsize"] / 60

        df_standing = df_standing.assign(
            time_needed=(df_standing.chargingdemand / df_standing.netto_charging_capacity.multiply(time_factor))
                .astype(float).round(1).apply(np.ceil).astype(np.int32),
            time=df_standing.charge_end - df_standing.charge_start + 1,
            cap=df_standing.netto_charging_capacity.divide(setup_dict["eta_CP"]),
        )

        df_standing.time = df_standing.time - df_standing.time_needed

        df_standing = df_standing.sort_values(
            by=["time", "time_needed", "charge_start"],
            ascending=True,
        )

        df_standing = df_standing.assign(
            cp_mapping=0
        )

        gdf_cps.reset_index(
            drop=True,
            inplace=True,
        )

        use_cases = [3, 4]

        cp_ags_lists = [0] * len(use_cases)

        for count, use_case in enumerate(use_cases):
            gdf_cps_use_case = gdf_cps.copy()[gdf_cps.use_case == use_case]

            cp_ags_list = list(
                zip(
                    gdf_cps_use_case.ags.tolist(),
                    gdf_cps_use_case.cp_idx.tolist(),
                )
            )

            cp_ags_list_unique = list(set(cp_ags_list))

            cp_ags_list_unique.sort()

            cp_ags_lists[count] = cp_ags_list_unique.copy()

            for count_cp, (ags, cp_idx) in enumerate(cp_ags_list_unique):
                df_standing[
                    (df_standing.use_case == use_case) &
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ] = df_standing[
                    (df_standing.use_case == use_case) &
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ].assign(
                    cp_mapping=count_cp
                )

        df_standing.cp_mapping = df_standing.cp_mapping.astype(int)

        cp_load_home = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(cp_ags_lists[0]),
            ),
            dtype=float,
        )

        cp_load_home[:] = 0

        cp_load_work = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(cp_ags_lists[1]),
            ),
            dtype=float,
        )

        cp_load_work[:] = 0

        return df_standing, cp_ags_lists[0], cp_ags_lists[1], cp_load_home, cp_load_work

    except:
        traceback.print_exc()


def get_residual_load_with_evs(
        s_residual_load,
        setup_dict,
        data_dir,
        grid_id,
):
    try:
        files = [
            r"dumb_charging_timeseries_public.h5",
            r"dumb_charging_timeseries_hpc.h5",
        ]

        sub_dir = "eDisGo_charging_time_series"

        files = [
            os.path.join(data_dir.parent, sub_dir, str(grid_id), f) for f in files
        ]

        df_residual_load = s_residual_load.to_frame(
            name="residual_load"
        )

        df_residual_load = df_residual_load.assign(
            residual_unflex=df_residual_load.residual_load.multiply(-1),
        )

        for f in files:
            arr = pd.read_hdf(
                f,
                key="df_load",
            ).sum(axis=1).divide(1000).to_numpy()

            df_residual_load.residual_unflex += arr

        days = int(setup_dict["days"])

        time_factor = setup_dict["stepsize"] / 60

        if days == 365:
            pass
        elif days < 365:
            df_residual_load = df_residual_load.iloc[:int(days * 24 / time_factor)]
        else:
            df_prolong = df_residual_load.copy().iloc[:int((days - 365) * 24 / time_factor)]
            df_prolong.index = [
                ts + timedelta(days=365) for ts in df_prolong.index[:len(df_prolong)]
            ]

            df_residual_load = df_residual_load.append(df_prolong)

        return df_residual_load

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
        gdf_cps = gdf_cps.copy().loc[:, (gdf_cps != 0).any(axis=0)]

        gdf_cps = pd.merge(
            edisgo.topology.charging_points_df["bus"],
            gdf_cps,
            left_index=True,
            right_on="edisgo_id",
        )

        gdf_cps.reset_index(
            drop=True,
            inplace=True,
        )

        time_factor = setup_dict["stepsize"] / 60

        df_standing.chargingdemand = df_standing.chargingdemand.astype(float)

        df_standing = get_groups(
            gdf_cps,
            df_standing,
            time_factor,
        )

        cp_ags_list = list(
            zip(
                df_standing.ags.tolist(),
                df_standing.cp_idx.tolist(),
            )
        )

        cp_ags_list_unique = list(set(cp_ags_list))

        del cp_ags_list

        cp_ags_list_unique.sort()

        cp_load = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(gdf_cps),
            ),
            dtype=float,
        )

        cp_load[:] = 0

        df_standing = df_standing.assign(
            cap=df_standing.netto_charging_capacity.divide(setup_dict["eta_CP"])
        )

        for count_cps, (ags, cp_idx) in enumerate(cp_ags_list_unique):
            df_cp = df_standing.copy()[
                (df_standing.ags == ags) &
                (df_standing.cp_idx == cp_idx)
                ]

            for cap, start, stop in list(
                zip(
                    df_cp.cap.tolist(),
                    df_cp.charge_start.tolist(),
                    (df_cp.stop + 1).tolist(),
                )
            ):
                cp_load[start:stop:2, count_cps] += cap

            df_unfulfilled = df_cp.copy()[df_cp.demand_left > 0]

            if not df_unfulfilled.empty:
                for cap, start, stop in list(
                        zip(
                            df_cp.cap.tolist(),
                            df_cp.start_next.tolist(),
                            (df_cp.stop_next + 1).tolist(),
                        )
                ):
                    cp_load[start:stop:2, count_cps] += cap

        # df_standing["time"] = (df_standing.chargingdemand / df_standing.netto_charging_capacity.divide(4))\
        #     .astype(float).round(1).apply(np.ceil).astype(int)
        # df_standing["up"] = df_standing.time * df_standing.netto_charging_capacity.divide(4)
        #
        # print("before:", df_standing.up.sum() / 0.9)
        print(f"{round(np.sum(cp_load) / 4, 0)} kWh in grid {str(grid_id)} and scenario {data_dir.parts[-2]}.")

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


def get_groups(
        gdf,
        df_standing,
        time_factor,
):
    try:
        cols = [
            col for col in gdf.columns.tolist() if "cp_0" in col
        ]

        mask = np.empty(
            shape=(
                gdf.shape[0],
                len(cols),
            ),
            dtype=np.bool_,
        )

        mask[:] = 0

        iterator = cycle(range(2))

        for bus in gdf.bus.unique():
            df_bus = gdf.copy()[gdf.bus == bus]

            df_bus_cap = df_bus[cols].astype(float).round(1) # FIXME: automate this to len of cols named "cp_#####"

            unique_caps = list(pd.unique(df_bus_cap.values.ravel()))

            if 0 in unique_caps:
                unique_caps.remove(0)

            for cap in unique_caps:
                cap_idxs = getIndexes(df_bus_cap, cap)

                for (row, col) in cap_idxs:
                    mask[row, col] = next(iterator)

        df_standing = df_standing.assign(
            group=[
                mask[
                    gdf.index[
                        (gdf.ags == ags) &
                        (gdf.cp_idx == cp_idx)
                        ][0],
                    cp_sub_idx,
                ] for ags, cp_idx, cp_sub_idx in list(
                    zip(
                        df_standing.ags.tolist(),
                        df_standing.cp_idx.tolist(),
                        (df_standing.cp_sub_idx - 1).tolist(),
                    )
                )
            ]
        )

        df_standing = df_standing.assign(
            start_on=[
                0 if ((group and start == 0) or (not group and start == 1)) else 1 for group, start in list(
                    zip(
                        df_standing.group.tolist(),
                        (df_standing.charge_start % 2).tolist(),
                    )
                )
            ]
        )

        df_standing.start_on = df_standing.charge_start + df_standing.start_on
        df_standing = df_standing.assign(
            stop=(df_standing.chargingdemand / df_standing.netto_charging_capacity.multiply(time_factor) - 1)
                .astype(float).round(1).apply(np.ceil).multiply(2).astype(int) + df_standing.start_on
        )

        df_standing.stop = df_standing[["charge_end", "stop"]].min(axis=1)

        df_standing = df_standing.assign(
            check=(df_standing.stop - df_standing.start_on) % 2
        )

        df_standing.stop = df_standing.stop - df_standing.check

        df_standing.pop("check")

        df_standing = df_standing.assign(
            demand_left=df_standing.chargingdemand - ((df_standing.stop - df_standing.start_on)
                .astype(float).divide(2).apply(np.floor) + 1)
                        * df_standing.netto_charging_capacity.multiply(time_factor)
        )

        df_standing.demand_left = df_standing.demand_left.round(5)

        df_standing.demand_left[df_standing.demand_left < 0] = 0

        df_standing = df_standing.assign(
            start_next=[
                charge_start if charge_start < start_on else start_on + 1 for charge_start, start_on in list(
                    zip(
                        df_standing.charge_start.tolist(),
                        df_standing.start_on.tolist(),
                    )
                )
            ],
        )

        df_standing = df_standing.assign(
            stop_next=(df_standing.demand_left / df_standing.netto_charging_capacity.multiply(time_factor) - 1) \
                     .astype(float).round(1).apply(np.ceil).multiply(2).astype(int) + df_standing.start_next,
        )

        return df_standing

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
        timeindex = pd.date_range(
            '2011-01-01',
            periods=8760,
            freq='H',
        )

        p_bio = 9983 # MW
        e_bio = 50009 # GWh

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

        edisgo_residual = EDisGo(
            ding0_grid=os.path.join(
                ding0_dir, str(grid_id)
            ),
            generator_scenario=generator_scenario,
            timeseries_load="demandlib",
            timeseries_generation_fluctuating="oedb",
            timeseries_generation_dispatchable=timeseries_generation_dispatchable,
            timeindex=timeindex,
        )

        s_residual_load = get_residual(
            edisgo_residual,
            freq="15min",
        )

        print("Residual Load in grid {} is done.".format(grid_id))

        del edisgo_residual, timeseries_generation_dispatchable, share, vls_bio, e_bio, p_bio, timeindex

        edisgo_wc = EDisGo(
            ding0_grid=os.path.join(
                ding0_dir, str(grid_id)
            ),
            worst_case_analysis=worst_case_analysis,
            generator_scenario=generator_scenario,
        )

        timeindex = edisgo_wc.timeseries.timeindex

        comp_type = "ChargingPoint"

        ts_active_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_active_power",
        )

        ts_reactive_power = pd.Series(
            data=[0] * len(timeindex),
            index=timeindex,
            name="ts_reactive_power",
        )

        cp_edisgo_id = [
            EDisGo.integrate_component(
                edisgo_wc,
                comp_type=comp_type,
                geolocation=geolocation,
                use_case=get_edisgo_use_case_name(use_case),
                voltage_level=None,
                add_ts=True,
                ts_active_power=ts_active_power,
                ts_reactive_power=ts_reactive_power,
                p_nom=p_nom,
            ) for geolocation, p_nom, use_case in list(
                zip(
                    gdf_cps_total.geometry.tolist(),
                    gdf_cps_total.cp_connection_rating.divide(1000).tolist(),  # kW -> MW
                    gdf_cps_total.use_case.tolist(),
                )
            )
        ]

        gdf_cps_total.insert(0, "edisgo_id", cp_edisgo_id)

        return edisgo_wc, gdf_cps_total, s_residual_load

    except:
        traceback.print_exc()


# def resample_edisgo_timeseries(
def get_residual(
        edisgo,
        freq="15min",
):
    try:
        timeindex_old = edisgo.timeseries.timeindex

        timeindex = pd.date_range(
            "2011-01-01",
            periods=len(timeindex_old) * 60 / int(freq[:2]),
            freq=freq,
        )

        edisgo.timeseries.timeindex = timeindex

        # edisgo.timeseries.generators_active_power = edisgo.timeseries.generators_active_power.ffill()
        # edisgo.timeseries.generators_reactive_power = edisgo.timeseries.generators_reactive_power.ffill()
        # edisgo.timeseries.loads_active_power = edisgo.timeseries.loads_active_power.ffill()
        # edisgo.timeseries.loads_reactive_power = edisgo.timeseries.loads_reactive_power.ffill()
        #
        # edisgo.timeseries._generators_active_power = edisgo.timeseries.generators_active_power.copy()
        # edisgo.timeseries._generators_reactive_power = edisgo.timeseries.generators_reactive_power.copy()
        # edisgo.timeseries._loads_active_power = edisgo.timeseries.loads_active_power.copy()
        # edisgo.timeseries._loads_reactive_power = edisgo.timeseries.loads_reactive_power.copy()

        s_residual_load = edisgo.timeseries.residual_load.copy()
        s_residual_load = s_residual_load.ffill()

        # edisgo.timeseries._residual_load = s_residual_load.copy()
        # edisgo.timeseries.residual_load = s_residual_load

        # return edisgo, s_residual_load
        return s_residual_load

    except:
        traceback.print_exc()


def grid_independent_charging(
        edisgo,
        df_standing,
        gdf_cps,
        setup_dict,
        use_case,
        grid_id,
        data_dir,
        strategy="dumb",
):
    try:
        gdf_cps = pd.merge(
            gdf_cps,
            edisgo.topology.charging_points_df["bus"],
            left_on="edisgo_id",
            right_index=True,
        )

        gdf_cps = pd.merge(
            gdf_cps,
            edisgo.topology.buses_df["v_nom"],
            left_on="bus",
            right_index=True,
        )

        df_standing = pd.merge(
            df_standing,
            gdf_cps[["ags", "cp_idx", "v_nom"]],
            left_on=["ags", "cp_idx"],
            right_on=["ags", "cp_idx"],
        )

        cp_ags_list = list(
            zip(
                gdf_cps.ags.tolist(),
                gdf_cps.cp_idx.tolist(),
            )
        )

        cp_ags_list_unique = list(set(cp_ags_list))

        cp_ags_list_unique.sort()

        cp_load = np.empty(
            shape=(
                int(setup_dict["days"] * (60 / setup_dict["stepsize"]) * 24),
                len(cp_ags_list_unique),
            ),
            dtype=float,
        )

        cp_load[:] = 0

        time_factor = setup_dict["stepsize"] / 60

        df_standing = df_standing.assign(
            t_given=df_standing.charge_end - df_standing.charge_start + 1,
            stop_dumb=(df_standing.chargingdemand / df_standing.netto_charging_capacity.multiply(time_factor)) \
                .astype(float).round(1).apply(np.ceil).astype(int),
        )

        df_standing = df_standing.assign(
            demand_reduced=df_standing.stop_dumb.multiply(time_factor) * df_standing.netto_charging_capacity,
            min_cap_tech=df_standing.netto_charging_capacity.multiply(0.1),
        )

        df_standing = df_standing.assign(
            min_cap_demand=df_standing.demand_reduced / df_standing.t_given.multiply(time_factor),
        )

        df_standing = df_standing.assign(
            cap_reduced=df_standing[["min_cap_tech", "min_cap_demand"]].max(axis=1),
        )

        df_standing = df_standing.assign(
            stop_reduced=(df_standing.demand_reduced / df_standing.cap_reduced.multiply(time_factor))
                     .round(0).astype(np.int32) + df_standing.charge_start,
        )

        df_standing.stop_dumb += df_standing.charge_start

        if strategy == "dumb":

            for count_cps, (ags, cp_idx) in enumerate(cp_ags_list_unique):
                df_cp = df_standing.copy()[
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ]

                for cap_dumb, cap_reduced, start, stop_dumb, stop_reduced, v in list(
                    zip(
                        df_cp.netto_charging_capacity.divide(setup_dict["eta_CP"]).tolist(),
                        df_cp.cap_reduced.divide(setup_dict["eta_CP"]).tolist(),
                        df_cp.charge_start.tolist(),
                        df_cp.stop_dumb.tolist(),
                        df_cp.stop_reduced.tolist(),
                        df_cp.v_nom.tolist(),
                    )
                ):
                    if use_case == 4 or use_case == 3: # TODO: für work UND home?
                        if v < 1:
                            cp_load[start:stop_dumb, count_cps] += cap_dumb
                        else:
                            cp_load[start:stop_reduced, count_cps] += cap_reduced
                    else:
                        cp_load[start:stop_dumb, count_cps] += cap_dumb

        elif strategy == "reduced":

            for count_cps, (ags, cp_idx) in enumerate(cp_ags_list_unique):
                df_cp = df_standing.copy()[
                    (df_standing.ags == ags) &
                    (df_standing.cp_idx == cp_idx)
                ]

                for cap, start, stop in list(
                    zip(
                        df_cp.cap_reduced.divide(setup_dict["eta_CP"]).tolist(),
                        df_cp.charge_start.tolist(),
                        df_cp.stop_reduced.tolist(),
                    )
                ):
                    cp_load[start:stop, count_cps] += cap

        else:
            raise ValueError("Strategy '{}' does not exist.".format(strategy))

        # df_standing["time"] = (df_standing.chargingdemand / df_standing.netto_charging_capacity.divide(4))\
        #     .astype(float).round(1).apply(np.ceil).astype(int)
        # df_standing["up"] = df_standing.time * df_standing.netto_charging_capacity.divide(4)
        #
        # print("before:", df_standing.up.sum() / 0.9)
        print(f"{round(np.sum(cp_load) / 4, 0)} kWh in grid {str(grid_id)} and scenario {data_dir.parts[-2]}.")

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
        )

        df_load.columns = pd.MultiIndex.from_tuples(cp_idxs)

        df_load = df_load.iloc[:365*24*4]

        use_case_name = get_use_case_name(use_case)

        file_name = "{}_charging_timeseries_{}.h5".format(strategy, use_case_name)

        sub_dir = "eDisGo_charging_time_series"

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

        sub_dir = "eDisGo_charging_time_series"

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

        cp_idx = gdf.pop("cp_idx")

        gdf = gdf.loc[:, (gdf != 0).any(axis=0)] # TODO: this also removes cp_idx if all are 0

        gdf.insert(2, "cp_idx", cp_idx)

        gdf.pop("use_case")

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
        df_standing_total.netto_charging_capacity = df_standing_total.netto_charging_capacity.astype(float)
        df_standing_total.chargingdemand = df_standing_total.chargingdemand.astype(float)

        ags_series = gdf_cps_total.pop("ags")
        geometry_series = gdf_cps_total.pop("geometry")

        gdf_cps_total.insert(0, "ags", ags_series)

        gdf_cps_total = gdf_cps_total.assign(
            geometry=geometry_series,
        )

        cols = gdf_cps_total.columns.difference(["ags", "cp_idx", "geometry", "use_case"])

        cp_count = gdf_cps_total[cols].gt(0).sum(axis=1)

        gdf_cps_total.insert(2, "cp_count", cp_count)

        cp_capacity = gdf_cps_total[cols].sum(axis=1).astype(float).divide(eta_cp).round(1)

        gdf_cps_total.insert(3, "cp_capacity", cp_capacity)

        cp_connection_rating = [
            get_connection_rating_factor(cap, use_case) * cap for cap, use_case in list(
                zip(
                    cp_capacity,
                    gdf_cps_total.use_case.tolist(),
                )
            )
        ]

        gdf_cps_total.insert(4, "cp_connection_rating", cp_connection_rating)

        use_case_series = gdf_cps_total.pop("use_case")

        gdf_cps_total.insert(1, "use_case", use_case_series)

        gdf_cps_total.cp_capacity = gdf_cps_total.cp_capacity.round(1)

        df_standing_total = df_standing_total.sort_values(
            by=["ags", "use_case", "cp_idx", "cp_sub_idx"],
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

        return gdf_cps_total, df_standing_total

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


def get_edisgo_use_case_name(
        use_case_nr,
):
    if use_case_nr == 1:
        use_case_name = "fast"
    elif use_case_nr == 2:
        use_case_name = "public"
    elif use_case_nr == 3:
        use_case_name = "home"
    elif use_case_nr == 4:
        use_case_name = "work"
    else:
        raise ValueError("Use case {} is not valid.".format(use_case_nr))

    return use_case_name


def getIndexes(
        dfObj,
        value,
):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, int(col[3:]) - 1))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos


def get_connection_rating_factor(
        p,
        use_case,
        p_upper_limit=1000,
        p_lower_limit=300,
        f_upper=0.5,
):
    if use_case == 1 or use_case == 2:
        return 1
    else:
        if p < p_lower_limit:
            return 1
        elif p > p_upper_limit:
            return f_upper
        else:
            return - (0.5 / 700) * p + (17 / 14)

