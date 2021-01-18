import gc
import os.path
import edisgo.flex_opt.charging_ev as cEV
import multiprocessing

from pathlib import Path

gc.collect()

num_threads = 1

# TODO: set dir
data_dir = r"\\FS01\Daten_flexibel_02\simbev_results"
# data_dir = r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results"

scenarios = [
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    "Mobility_Transition_2050_simbev_run",
    "Szenarette_Kleinwagen_2050_simbev_run",
    "Reference_2050_simbev_run",
    "NEP_C_2035_simbev_run",
]

sub_dir = "cp_standing_times_mapping"

data_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir)) for scenario in scenarios
]

if __name__ == "__main__":
    if num_threads == 1:
        cEV.charging(
            data_dirs[0],
        )
    else:
        pass