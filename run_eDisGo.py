import gc
import edisgo.io.simBEV_import as sB

from pathlib import Path

gc.collect()

data_dir = Path(
    r"\\FS01\Daten_flexibel_02\simbev_results\Electrification_2050_simbev_run\standing_times"
)

localiser_data_dir = Path(
    r"\\FS01\Daten_flexibel_02\simbev_results\Beispieldaten"
)

if __name__ == "__main__":
    sB.run_simBEV_import(
        data_dir,
        localiser_data_dir,
    )