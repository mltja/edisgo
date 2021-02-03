import pandas as pd
import logging

from edisgo.network.timeseries import import_load_timeseries

elia_logger = logging.getLogger('elia_project')


def get_rated_generator_timeseries(path, timeindex):
    # import installed capacities
    installed_capacities = pd.read_csv(
        path,
    ).loc[[0]]

    for col in installed_capacities.columns[2:]:
        installed_capacities[col] = installed_capacities[col].astype(float)

    # extract capacities of fluctuating and dispatchable generation
    installed_capacities_fluct = extract_capacities_of_fluctuating_generation(
        installed_capacities)

    installed_capacities_disp = extract_capacities_of_dispatchable_generation(
        installed_capacities)
    elia_logger.debug('Installed generator capacities imported.')

    # import timeseries
    ts_path = r"timeseries_of_production.csv"
    timeseries_production = pd.read_csv(
        ts_path,
        nrows=len(timeindex)+1,
    ).drop(0).drop(
        'Unnamed: 0',
        axis=1,
    ).dropna(
        axis=1,
        how='all',
    ).dropna()

    for col in timeseries_production.columns:
        timeseries_production[col] = timeseries_production[col].astype(float)

    # get rated onshore wind and PV timeseries
    timeseries_fluctuating = extract_rated_production_of_fluctuating_generation(
        installed_capacities_fluct, timeindex, timeseries_production)
    elia_logger.debug('Timeseries of fluctuating generators imported.')
    # get rated timeseries of dispatchable plants
    timeseries_dispatchable = \
        extract_rated_production_of_dispatchable_generation(
            installed_capacities_disp, timeindex, timeseries_production)
    elia_logger.debug('Timeseries of dispatchable generators imported.')

    return timeseries_fluctuating, timeseries_dispatchable


def extract_rated_production_of_dispatchable_generation(
        installed_capacities_disp, timeindex, timeseries_production):
    disp_production_dict = {'GAS MWh exp': 'gas',
                            'H. ROR MWh exp': 'hydro_run',
                            'Other RES': 'other_res',
                            'Other non-RES': 'other_non_res'}
    timeseries_dispatchable = timeseries_production[
        disp_production_dict.keys()]. \
        rename(columns=disp_production_dict).set_index(timeindex).divide(
        installed_capacities_disp.values)
    return timeseries_dispatchable


def extract_rated_production_of_fluctuating_generation(
        installed_capacities_fluct, timeindex, timeseries_production):
    fluct_production_dict = {'Wind Onshore': 'wind', 'PV': 'solar'}
    timeseries_fluctuating = timeseries_production[
        fluct_production_dict.keys()]. \
        rename(columns=fluct_production_dict).set_index(
        timeindex).divide(installed_capacities_fluct.values)
    return timeseries_fluctuating


def extract_capacities_of_dispatchable_generation(installed_capacities):
    disp_capacity_dict = {'Gas': 'gas', 'Hydro-run': 'hydro_run',
                          'Other RES': 'other_res',
                          'Othernon-RES': 'other_non_res'}
    installed_capacities_disp = installed_capacities[
        disp_capacity_dict.keys()]. \
        rename(columns=disp_capacity_dict)
    return installed_capacities_disp


def extract_capacities_of_fluctuating_generation(installed_capacities):
    fluct_capacity_dict = {'Wind-on-shore': 'wind', 'Solar-PV': 'solar'}
    installed_capacities_fluct = installed_capacities[
        fluct_capacity_dict.keys()]. \
        rename(columns=fluct_capacity_dict)
    return installed_capacities_fluct


def get_rated_load_timeseries(path, timeindex, edisgo_obj):
    conventional_load_column = 'Conventional load'

    # get load timeseries
    timeseries_load = pd.read_csv(
        path,
        nrows=len(timeindex),
    ).rename(
        columns={
            conventional_load_column : "load",
        },
    )

    timeseries_load.index = timeindex

    # annual consumption in MWh
    annual_consumption = 475.9933765 * 1e6
    # correction ego - Elia
    annual_consumption_ego = 506 * 1e6
    correction_factor = annual_consumption/annual_consumption_ego

    # timeseries of load in distribution grids
    timeseries_load.load *= correction_factor

    # get sectoral share of load from ego for every timestep
    annual_consumption_per_sector_ego = {
        'agricultural': 47469064.68581086,
        'industrial': 204023507.5606358,
        'residential': 131166982.09864908,
        'retail': 92446937.04619695,
    }

    sectoral_share_of_load_timeseries = \
        get_sectoral_share_of_load_timeseries(
            edisgo_obj, timeindex, annual_consumption_per_sector_ego)

    # multiply sectoral share with overall load timeseries given as input
    sectoral_load = pd.DataFrame(index=timeindex)
    for sector in sectoral_share_of_load_timeseries:
        sectoral_load[sector] = \
            sectoral_share_of_load_timeseries[sector] * timeseries_load['load']
    # calculate rated time series per sector
    annual_consumption_per_sector_df = pd.DataFrame(
        annual_consumption_per_sector_ego, index=[0])
    sectoral_rated_load = (sectoral_load /
                           annual_consumption_per_sector_df.loc[0, :])
    return sectoral_rated_load


def get_sectoral_share_of_load_timeseries(
        edisgo_obj, timeindex, annual_consumption_per_sector):

    config_data = edisgo_obj.config
    rated_load_profiles = import_load_timeseries(config_data, 'demandlib')
    load_profiles = pd.DataFrame(index=rated_load_profiles.index)
    load_profiles_percent = pd.DataFrame(index=rated_load_profiles.index)
    for sector in annual_consumption_per_sector.keys():
        load_profiles[sector] = \
            rated_load_profiles[sector] * annual_consumption_per_sector[sector]
    load_profiles['cumulated'] = load_profiles.sum(axis=1)
    for sector in annual_consumption_per_sector.keys():
        load_profiles_percent[sector] = \
            load_profiles[sector] / load_profiles['cumulated']
    elia_logger.debug('Timeseries imported.')
    # return profile with right timeindex
    # Todo: overthink if this should be adjusted (full year vs. only part)
    return load_profiles_percent.iloc[0:len(timeindex)].set_index(
        timeindex)


def get_residential_heat_pump_timeseries(
        path,
        demand=22.4, # NEP C 2035
):
    timeindex = pd.date_range(
        '2011-01-01',
        periods=8760,
        freq='H',
    )

    timeseries_HP = pd.read_csv(
        path,
        nrows=len(timeindex),
    ).rename(
        columns={
            "HPs (NEP19-2030B)" : "HP",
        },
    )

    timeseries_HP.index = timeindex

    demand_old = 18.2 # NEP B 2030

    timeseries_HP.HP *= (demand / demand_old)

    return timeseries_HP