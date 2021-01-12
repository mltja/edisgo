import numpy as np
import pandas as pd
import os.path
import traceback
import gc

from edisgo.io.simBEV_import import get_ags
from pathlib import Path
from time import perf_counter


def dumb_charging(
        data_dir,
):
    try:
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
                t1 = perf_counter()
                unflexible_charging(
                    data_dfs[count_cases],
                    use_case,
                    ags_dir,
                )
                print(
                    "It took {} to generate the timeseries for use case {} AGS Nr. {}.".format(
                        perf_counter() - t1, use_case, ags_dir.parts[-1]
                    )
                )
                data_dfs[count_cases] = 0

                gc.collect()

    except:
        traceback.print_exc()


def unflexible_charging(
        data,
        use_case,
        ags_dir,
):
    try:
        cp_idxs = data.cp_idx.unique().tolist()

        cp_idxs.sort()

        cp_load = np.empty(
            shape=(371*24*4, len(cp_idxs)),
            dtype=np.float64,
        )

        # TODO: automate this for other stepsizes than 15min
        data["stop"] = (data.chargingdemand / data.netto_charging_capacity.divide(4)).apply(np.ceil)\
                           .astype(np.uint32) + data.charge_start

        for count_cps, cp_idx in enumerate(cp_idxs):
            df_cp = data.copy()[data.cp_idx == cp_idx]
            for cap, start, stop in zip(
                df_cp.netto_charging_capacity.tolist(),
                df_cp.charge_start.tolist(),
                df_cp.stop.tolist(),
            ):
                # TODO: automate this for other efficiencies than 90%
                cp_load[start:stop, count_cps] += (cap / 0.9)

        df_load = pd.DataFrame(
            data=cp_load,
            columns=cp_idxs,
        )

        df_load = df_load.iloc[:365*24*4]

        file_name = "dumb_charging_timeseries_{}.csv".format(use_case)

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

        df_load.to_csv(
            export_path,
        )

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
                index_col=[0],
                # nrows=100, # TODO: remove after testing
                engine="c",
            )

            dfs[count_files] = df

        return dfs, use_cases


    except:
        traceback.print_exc()