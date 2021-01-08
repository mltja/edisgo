import numpy as np
import pandas as pd
import os.path
import gc
import multiprocessing
import traceback
import edisgo.io.simBEV_import as sB

from pathlib import Path

gc.collect()

data_dir = Path(
    r"\\FS01\Daten_flexibel_02\simbev_results\NEP_C_2035_simbev_run\standing_times_looped"
)

localiser_data_dir = Path(
    r"\\FS01\Daten_flexibel_02\simbev_results\Beispieldaten"
)

# exchange factor stepsize in min to stepsize in h
stepsize = 15
time_factor = stepsize / 60
days = 371
timesteps = int(days * 24 / time_factor)

# get ags numbers
ags_lst, ags_dirs = sB.get_ags(data_dir)

for count, ags_dir in enumerate(ags_dirs):

    (df_standing_times_home, df_standing_times_work,
     df_standing_times_public, df_standing_times_hpc) = sB.get_ags_data(ags_dir)

    gc.collect()

    df_cp_hpc, df_cp_public, df_cp_home, df_cp_work = sB.get_charging_points(
        localiser_data_dir,
        ags_dir,
    )

    use_cases = [
        "home",
        "work",
        "public",
        "HPC",
    ]

    for use_case in use_cases:
        if use_case == "home":
            sB.distribute_demand(
                use_case,
                df_standing_times_home,
                df_cp_home,
            )
        elif use_case == "work":
            sB.distribute_demand(
                use_case,
                df_standing_times_work,
                df_cp_work,
            )
        elif use_case == "public":
            sB.distribute_demand(
                use_case,
                df_standing_times_public,
                df_cp_public,
            )
        else:
            sB.distribute_demand(
                use_case,
                df_standing_times_hpc,
                df_cp_hpc,
            )



    break