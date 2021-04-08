import os
import gc
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product


base_dir = Path(
    # r"\\192.168.10.221\Daten_flexibel_02\simbev_results",
    r"/home/local/RL-INSTITUT/kilian.helfenbein/RLI_simulation_results/simbev_results",
)

ags_in_grid_path = Path(
    r"ags_in_grid.csv"
)

scenarios = [
    "NEP_C_2035_simbev_run",
    "Reference_2050_simbev_run",
    "Electrification_2050_simbev_run",
    "Electrification_2050_sensitivity_low_work_simbev_run",
]

sub_dir = "standing_times"

data_dirs = [
    Path(os.path.join(base_dir, scenario, sub_dir))
    for scenario in scenarios
]

df_ags_in_grid = pd.read_csv(ags_in_grid_path, index_col=[0])

for count, ags_list in enumerate(df_ags_in_grid.ags_list):
    ags_list = ags_list.replace(r"{", "").replace(r"}", "").replace(r" ", "")

    ags_list = ags_list.split(",")

    ags_list_new = []

    for string in ags_list:
        ags_list_new.append(string.split(":")[0].zfill(8))

    df_ags_in_grid.iat[count, 0] = ags_list_new.copy()

grid_ids = [176, 1056, 1690, 1811, 177]
df_ags_in_grid = df_ags_in_grid.loc[grid_ids]

df_total_demand = pd.DataFrame(data=0, index=scenarios, columns=grid_ids)
df_flex_demand = df_total_demand.copy()

for idx, ags_list in list(zip(df_ags_in_grid.index.tolist(), df_ags_in_grid.ags_list.tolist())):

    for scenario in scenarios:
        total_demand = 0
        flex_demand = 0

        main_dir = Path(os.path.join(base_dir, scenario, sub_dir))

        for ags in ags_list:
            ags_dir = Path(os.path.join(main_dir, ags))

            car_list = os.listdir(ags_dir)

            for count_cars, car in enumerate(car_list):
                car_dir = Path(os.path.join(ags_dir, car))

                df_car = pd.read_csv(car_dir, index_col=[0])

                df_car = df_car.loc[df_car.chargingdemand > 0]

                df_car["min_charge_time"] = df_car.chargingdemand.divide(
                    df_car.netto_charging_capacity.divide(4)).apply(np.ceil).astype(int)

                df_flex = df_car.loc[(df_car.use_case == "private") & (df_car.min_charge_time < df_car.charge_time)]

                total_demand += df_car.chargingdemand.sum()

                flex_demand += df_flex.chargingdemand.sum()

                if count_cars % 50 == 0:
                    print(round(count_cars / len(car_list) * 100, 1), "% of cars")

            print(ags, "is done")

        df_total_demand.at[scenario, idx] = total_demand
        df_flex_demand.at[scenario, idx] = flex_demand

        gc.collect()

        print(scenario, "in", idx, "is done")
    print(idx, "is done")

df_total_demand.to_csv(os.path.join(base_dir, "total_charging_demand_per_grid_per_week.csv")
df_flex_demand.to_csv(os.path.join(base_dir, "flex_charging_demand_per_grid_per_week.csv")