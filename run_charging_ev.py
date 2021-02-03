import gc
import os.path
import logging
import edisgo.flex_opt.charging_ev as cEV
import multiprocessing
import warnings

from pathlib import Path


# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

gc.collect()

num_threads = 1

data_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ding0_dir = Path( # TODO: set dir
    r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    # r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
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

data_dirs = [
    Path(os.path.join(data_dir, scenario, sub_dir)) for scenario in scenarios
]

data_tuples = [
    (directory, ding0_dir) for directory in data_dirs
]

if __name__ == "__main__":
    if num_threads == 1:
        cEV.charging(
            data_dirs[5],
            ding0_dir,
        )
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.starmap(
                cEV.charging,
                data_tuples,
            )

