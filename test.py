import os.path
import logging
import warnings
import pandas as pd

from pathlib import Path
from edisgo import EDisGo


# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# suppress warnings
# disable for development
warnings.filterwarnings("ignore")

ding0_dir = Path( # TODO: set dir
    # r"\\192.168.10.221\Daten_flexibel_01\ding0\20200812180021_merge",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_daten_flexibel_01/ding0/20200812180021_merge",
)

grid_id = "1056"

len_timeindex = 8760

timeindex = pd.date_range(
    "2011-01-01",
    periods=len_timeindex,
    freq='H',
)

p_bio = 9983  # MW
e_bio = 50009  # GWh

vls_bio = e_bio / (p_bio / 1000)

share = vls_bio / 8760

timeseries_generation_dispatchable = pd.DataFrame(
    {
        "biomass": [share] * len_timeindex,
        "coal": [1] * len_timeindex,
        "other": [1] * len_timeindex,
    },
    index=timeindex,
)

for _ in range(100):
    edisgo = EDisGo(
        ding0_grid=os.path.join(
            ding0_dir, str(grid_id)
        ),
        generator_scenario="ego100",
        timeseries_load="demandlib",
        timeseries_generation_fluctuating="oedb",
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex,
    )

    df = edisgo.topology.lines_df[edisgo.topology.lines_df.index.str.contains("3964241")]

    print(len(df))