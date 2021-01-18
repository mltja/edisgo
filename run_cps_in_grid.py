import gc
import os.path
import edisgo.io.simBEV_import as sB
import multiprocessing

from pathlib import Path


gc.collect()

num_threads = 1

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

scenarios = [
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    "Mobility_Transition_2050_simbev_run",
    "Szenarette_Kleinwagen_2050_simbev_run",
    "Reference_2050_simbev_run",
    "NEP_C_2035_simbev_run",
]

sub_dir = "cp_standing_times_mapping"

data_dirs = Path(
    os.path.join(
        data_dir,
        scenarios[5],
        sub_dir,
    )
)

if __name__ == "__main__":
    if num_threads == 1:
        sB.run_cps_in_grid(
            data_dirs,
        )