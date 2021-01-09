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

if __name__ == "__main__":
    sB.run_simBEV_import(
        data_dir,
        localiser_data_dir,
    )