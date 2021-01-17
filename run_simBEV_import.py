import gc
import os.path
import edisgo.io.simBEV_import as sB
import multiprocessing

from pathlib import Path

gc.collect()

num_threads = 6

data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

localiser_data_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results\2021-01-15_cp_locations_Elia_Neuberechnung_Rohdaten\raw",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results/2021-01-15_cp_locations_Elia_Neuberechnung_Rohdaten/raw",
)

scenarios = [
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
    "Mobility_Transition_2050_simbev_run",
    "Szenarette_Kleinwagen_2050_simbev_run",
    "Reference_2050_simbev_run",
    "NEP_C_2035_simbev_run",
]

sub_dir = "standing_times_looped"

data_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir)) for scenario in scenarios
]

data_tuples = [
    (directory, localiser_data_dir) for directory in data_dirs
]

if __name__ == "__main__":
    if num_threads == 1:
        sB.run_simBEV_import(
            data_dirs[0],
            localiser_data_dir,
        )
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.starmap(
                sB.run_simBEV_import,
                data_tuples,
            )