"""
This module provides tools to convert graph based representation of the network
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA network
container.
"""

import numpy as np
import pandas as pd
from math import sqrt
from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe
from networkx import connected_components
import collections


def to_pypsa(grid_object, timesteps, **kwargs):
    """
    Export edisgo object to PyPSA Network

    For details from a user perspective see API documentation of
    :meth:`~edisgo.EDisGo.analyze` of the API class
    :class:`~.edisgo.EDisGo`.

    Translating eDisGo's network topology to PyPSA representation is structured
    into translating the topology and adding time series for components of the
    network. In both cases translation of MV network only (`mode='mv'`,
    `mode='mvlv'`), LV network only(`mode='lv'`), MV and LV (`mode=None`)
    share some code. The code is organized as follows:

    * Medium-voltage only (`mode='mv'`): All medium-voltage network components
      are exported including the medium voltage side of LV station.
      Transformers are not exported in this mode. LV network load
      and generation is considered using :func:`append_lv_components`.
      Time series are collected and imported to PyPSA network.
    * Medium-voltage including transformers (`mode='mvlv'`). Works similar as
      the first mode, only attaching LV components to the LV side of the
      LVStation and therefore also adding the transformers to the PyPSA network.
    * Low-voltage only (`mode='lv'`): LV network topology including the MV-LV
      transformer is exported. The slack is defind at primary side of the MV-LV
      transformer.
    * Both level MV+LV (`mode=None`): The entire network topology is translated to
      PyPSA in order to perform a complete power flow analysis in both levels
      together. First, both network levels are translated seperately and then
      merged. Time series are obtained at once for both network levels.

    This PyPSA interface is aware of translation errors and performs so checks
    on integrity of data converted to PyPSA network representation

    * Sub-graphs/ Sub-networks: It is ensured the network has no islanded parts
    * Completeness of time series: It is ensured each component has a time
      series
    * Buses available: Each component (load, generator, line, transformer) is
      connected to a bus. The PyPSA representation is check for completeness of
      buses.
    * Duplicate labels in components DataFrames and components' time series
      DataFrames

    Parameters
    ----------
    grid_object: :class:`~.network.topology.Topology`
        eDisGo topology container
    mode : str
        Determines network levels that are translated to
        `PyPSA network representation
        <https://www.pypsa.org/doc/components.html#network>`_. Specify

        * None to export MV and LV network levels. None is the default.
        * 'mv' to export MV network level only. This includes cumulative load
          and generation from underlying LV network aggregated at respective LV
          station's primary side.
        * 'mvlv' to export MV network level only. This includes cumulative load
          and generation from underlying LV network aggregated at respective LV
          station's secondary side.
          #ToDo change name of this mode or use kwarg to define where to aggregate lv loads and generation
        * 'lv' to export specified LV network only.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or \
        :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps to export to pypsa representation
        and use in power flow analysis.

    Returns
    -------
    :pypsa:`pypsa.Network<network>`
        The `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_ container.

    """

    def _set_slack(grid):
        """
        Sets slack at given grid's station secondary side.

        It is assumed that bus of secondary side is always given in
        transformer's bus1.

        Parameters
        -----------
        grid : :class:`~.network.grids.Grid`
            Low or medium voltage grid to position slack in.

        Returns
        -------
        """
        slack_bus = grid.transformers_df.bus1.iloc[0]
        return pd.DataFrame(
            data={"bus": slack_bus, "control": "Slack"},
            index=["Generator_slack"],
        )

    mode = kwargs.get("mode", None)
    aggregate_loads = kwargs.get("aggregate_loads", None)
    aggregate_generators = kwargs.get("aggregate_generators", None)
    aggregate_storages = kwargs.get("aggregate_storages", None)
    aggregated_lv_components = {"Generator": {}, "Load": {}, "StorageUnit": {}}

    # check if timesteps is array-like, otherwise convert to list (necessary
    # to obtain a dataframe when using .loc in time series functions)
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.set_snapshots(timesteps)

    # define edisgo_obj, buses_df, slack_df and components for each use case
    if mode is None:

        edisgo_obj = grid_object
        buses_df = grid_object.topology.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(edisgo_obj.topology.mv_grid)

        components = {
            "Load": grid_object.topology.loads_df.loc[
                :, ["bus", "peak_load"]
            ].rename(columns={"peak_load": "p_set"}),
            "Generator": grid_object.topology.generators_df.loc[
                :, ["bus", "control", "p_nom"]
            ].append(slack_df),
            "StorageUnit": grid_object.topology.storage_units_df.loc[
                :, ["bus", "control"]
            ],
            "Line": grid_object.topology.lines_df.loc[
                :,
                ["bus0", "bus1", "x", "r", "s_nom", "num_parallel", "length"],
            ],
            "Transformer": grid_object.topology.transformers_df.loc[
                :, ["bus0", "bus1", "x_pu", "r_pu", "type", "s_nom"]
            ].rename(columns={"r_pu": "r", "x_pu": "x"}),
        }

    elif "mv" in mode:

        edisgo_obj = grid_object.edisgo_obj
        buses_df = grid_object.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(grid_object)

        # MV components
        mv_components = {
            "Load": grid_object.loads_df.loc[:, ["bus", "peak_load"]].rename(
                columns={"peak_load": "p_set"}
            ),
            "Generator": grid_object.generators_df.loc[
                :, ["bus", "control", "p_nom"]
            ].append(slack_df),
            "StorageUnit": grid_object.storage_units_df.loc[
                :, ["bus", "control"]
            ],
            "Line": grid_object.lines_df.loc[
                :,
                ["bus0", "bus1", "x", "r", "s_nom", "num_parallel", "length"],
            ],
        }
        mv_components["Generator"][
            "fluctuating"
        ] = grid_object.generators_df.type.isin(["solar", "wind"])

        if mode is "mv":
            mv_components["Transformer"] = pd.DataFrame()
        elif mode is "mvlv":
            # get all MV/LV transformers
            mv_components[
                "Transformer"
            ] = edisgo_obj.topology.transformers_df.loc[
                :, ["bus0", "bus1", "x_pu", "r_pu", "type", "s_nom"]
            ].rename(
                columns={"r_pu": "r", "x_pu": "x"}
            )
        else:
            raise ValueError("Provide proper mode for mv network export.")

        # LV components
        lv_components_to_aggregate = {
            "Load": "loads_df",
            "Generator": "generators_df",
            "StorageUnit": "storage_units_df",
        }
        lv_components = {
            key: pd.DataFrame() for key in lv_components_to_aggregate
        }

        for lv_grid in grid_object.lv_grids:
            if mode is "mv":
                # get primary side of station to append loads and generators to
                station_bus = grid_object.buses_df.loc[
                    lv_grid.transformers_df.bus0.unique()
                ]
            elif mode is "mvlv":
                # get secondary side of station to append loads and generators
                # to
                station_bus = lv_grid.buses_df.loc[
                    [lv_grid.transformers_df.bus1.unique()[0]]
                ]
                buses_df = buses_df.append(station_bus.loc[:, ["v_nom"]])
            # handle one gate component
            for comp, df in lv_components_to_aggregate.items():
                comps = getattr(lv_grid, df).copy()
                comps.bus = station_bus.index.values[0]
                aggregated_lv_components[comp].update(
                    _append_lv_components(
                        comp,
                        comps,
                        lv_components,
                        repr(lv_grid),
                        aggregate_loads=aggregate_loads,
                        aggregate_generators=aggregate_generators,
                        aggregate_storages=aggregate_storages,
                    )
                )

        # merge components
        components = collections.defaultdict(pd.DataFrame)
        for comps in (mv_components, lv_components):
            for key, value in comps.items():
                components[key] = components[key].append(value)

    elif mode is "lv":

        edisgo_obj = grid_object.edisgo_obj
        buses_df = grid_object.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(grid_object)

        components = {
            "Load": grid_object.loads_df.loc[:, ["bus", "peak_load"]].rename(
                columns={"peak_load": "p_set"}
            ),
            "Generator": grid_object.generators_df.loc[
                :, ["bus", "control", "p_nom"]
            ].append(slack_df),
            "StorageUnit": grid_object.storage_units_df.loc[
                :, ["bus", "control"]
            ],
            "Line": grid_object.lines_df.loc[
                :,
                ["bus0", "bus1", "x", "r", "s_nom", "num_parallel", "length"],
            ],
        }
    else:
        raise ValueError(
            "Provide proper mode or leave it empty to export "
            "entire network topology."
        )

    # import network topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(buses_df, "Bus")
    for k, comps in components.items():
        pypsa_network.import_components_from_dataframe(comps, k)

    # import time series to PyPSA network

    import_series_from_dataframe(
        pypsa_network,
        _buses_voltage_set_point(
            edisgo_obj,
            buses_df.index,
            slack_df.loc["Generator_slack", "bus"],
            timesteps,
        ),
        "Bus",
        "v_mag_pu_set",
    )

    if len(components["Generator"].index) > 0:
        if len(aggregated_lv_components["Generator"]) > 0:
            (
                generators_timeseries_active,
                generators_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_obj,
                timesteps,
                "generators",
                components["Generator"].index,
                aggregated_lv_components["Generator"],
            )
        else:
            generators_timeseries_active = edisgo_obj.timeseries.generators_active_power.loc[
                timesteps, components["Generator"].index
            ]
            generators_timeseries_reactive = edisgo_obj.timeseries.generators_reactive_power.loc[
                timesteps, components["Generator"].index
            ]

        import_series_from_dataframe(
            pypsa_network, generators_timeseries_active, "Generator", "p_set"
        )
        import_series_from_dataframe(
            pypsa_network, generators_timeseries_reactive, "Generator", "q_set"
        )
        # set slack time series
        slack_ts = pd.DataFrame(
            data=[0] * len(timesteps),
            columns=[slack_df.index[0]],
            index=timesteps,
        )
        import_series_from_dataframe(
            pypsa_network, slack_ts, "Generator", "p_set"
        )
        import_series_from_dataframe(
            pypsa_network, slack_ts, "Generator", "q_set"
        )

    if len(components["Load"].index) > 0:
        if len(aggregated_lv_components["Load"]) > 0:
            (
                loads_timeseries_active,
                loads_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_obj,
                timesteps,
                "loads",
                components["Load"].index,
                aggregated_lv_components["Load"],
            )
        else:
            loads_timeseries_active = edisgo_obj.timeseries.loads_active_power.loc[
                timesteps, components["Load"].index
            ]
            loads_timeseries_reactive = edisgo_obj.timeseries.loads_reactive_power.loc[
                timesteps, components["Load"].index
            ]
        import_series_from_dataframe(
            pypsa_network, loads_timeseries_active, "Load", "p_set"
        )
        import_series_from_dataframe(
            pypsa_network, loads_timeseries_reactive, "Load", "q_set"
        )

    if len(components["StorageUnit"].index) > 0:
        if len(aggregated_lv_components["StorageUnit"]) > 0:
            (
                storages_timeseries_active,
                storages_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_obj,
                timesteps,
                "storage_units",
                components["StorageUnit"].index,
                aggregated_lv_components["StorageUnit"],
            )
        else:
            storages_timeseries_active = edisgo_obj.timeseries.storage_units_active_power.loc[
                timesteps, components["StorageUnit"].index
            ]
            storages_timeseries_reactive = edisgo_obj.timeseries.storage_units_reactive_power.loc[
                timesteps, components["StorageUnit"].index
            ]
        import_series_from_dataframe(
            pypsa_network,
            storages_timeseries_active.apply(pd.to_numeric),
            "StorageUnit",
            "p_set",
        )
        import_series_from_dataframe(
            pypsa_network,
            storages_timeseries_reactive.apply(pd.to_numeric),
            "StorageUnit",
            "q_set",
        )

    _check_integrity_of_pypsa(pypsa_network)

    return pypsa_network


def _append_lv_components(
    comp,
    comps,
    lv_components,
    lv_grid_name,
    aggregate_loads=None,
    aggregate_generators=None,
    aggregate_storages=None,
):
    """
    Method to append lv components to component dictionary. Used when only
    exporting mv grid topology. All underlaying LV components of an LVGrid are
    then connected to one side of the LVStation. If required, the LV components
    can be aggregated in different modes. As an example, loads can be
    aggregated sector-wise or all loads can be aggregated into one
    representative load. The sum of p_nom or peak_load of all cumulated
    components is calculated.

    Parameters
    ----------
    comp: str
        indicator for component type, can be 'Load', 'Generator' or
        'StorageUnit'
    comps: `pandas.DataFrame<dataframe>`
        component dataframe of elements to be aggregated
    lv_components: dict
        dictionary of LV grid components, keys are the 'Load', 'Generator' and
        'StorageUnit'
    lv_grid_name: str
        representative of LV grid of which components are aggregated
    aggregate_loads: str
        mode for load aggregation, can be 'sectoral' aggregating the loads
        sector-wise or 'all' aggregating all loads into one. Defaults to None,
        not aggregating loads but appending them to the station one by one.
    aggregate_generators: str
        mode for generator aggregation, can be 'type' resulting in
        aggregated generator for each generator type, 'curtailable' aggregating
        'solar' and 'wind' generators into one and all other generators into
        another generator. Defaults to None, when no aggregation is undertaken
        and generators are addded one by one.
    aggregate_storages: str
        mode for storage unit aggregation. Can be 'all' where all storage units
        in the grid are replaced by one storage. Defaults to None, where no
        aggregation is conducted and storage units are added one by one.

    Returns
    -------
    dict
        dict of aggregated elements for timeseries creation. Keys are names
        of aggregated elements and entries is a list of the names of all
        components aggregated in that respective key component.
        An example could look the following way:
        {'LVGrid_1_loads':
            ['Load_agricultural_LVGrid_1_1', 'Load_retail_LVGrid_1_2']}
    """
    aggregated_elements = {}
    if len(comps) > 0:
        bus = comps.bus.unique()[0]
    else:
        return {}
    if comp is "Load":
        if aggregate_loads is None:
            comps_aggr = comps.loc[:, ["bus", "peak_load"]].rename(
                columns={"peak_load": "p_set"}
            )
        elif aggregate_loads == "sectoral":
            comps_aggr = (
                comps.groupby("sector")
                .sum()
                .rename(columns={"peak_load": "p_set"})
                .loc[:, ["p_set"]]
            )
            for sector in comps_aggr.index.values:
                aggregated_elements[lv_grid_name + "_" + sector] = comps[
                    comps.sector == sector
                ].index.values
            comps_aggr.index = lv_grid_name + "_" + comps_aggr.index
            comps_aggr["bus"] = bus
        elif aggregate_loads == "all":
            comps_aggr = pd.DataFrame(
                {"bus": [bus], "p_set": [sum(comps.peak_load)]},
                index=[lv_grid_name + "_loads"],
            )
            aggregated_elements[lv_grid_name + "_loads"] = comps.index.values
        else:
            raise ValueError("Aggregation type for loads invalid.")
        lv_components[comp] = lv_components[comp].append(comps_aggr)
    elif comp is "Generator":
        flucts = ["wind", "solar"]
        if aggregate_generators is None:
            comps_aggr = comps.loc[:, ["bus", "control", "p_nom"]]
            comps_aggr["fluctuating"] = comps.type.isin(flucts)
        elif aggregate_generators == "type":
            comps_aggr = (
                comps.groupby("type").sum().loc[:, ["bus", "control", "p_nom"]]
            )
            comps_aggr.bus = bus
            comps_aggr.control = "PQ"
            comps_aggr["fluctuating"] = comps_aggr.index.isin(flucts)
            for gen_type in comps_aggr.index.values:
                aggregated_elements[lv_grid_name + "_" + gen_type] = comps[
                    comps.type == gen_type
                ].index.values
            comps_aggr.index = lv_grid_name + "_" + comps_aggr.index
        elif aggregate_generators == "curtailable":
            comps_fluct = comps[comps.type.isin(flucts)]
            comps_disp = comps[~comps.index.isin(comps_fluct.index)]
            comps_aggr = pd.DataFrame(columns=["bus", "control", "p_nom"])
            if len(comps_fluct) > 0:
                comps_aggr = comps_aggr.append(
                    pd.DataFrame(
                        {
                            "bus": [bus],
                            "control": ["PQ"],
                            "p_nom": [sum(comps_fluct.p_nom)],
                            "fluctuating": [True],
                        },
                        index=[lv_grid_name + "_fluctuating"],
                    )
                )
                aggregated_elements[
                    lv_grid_name + "_fluctuating"
                ] = comps_fluct.index.values
            if len(comps_disp) > 0:
                comps_aggr = comps_aggr.append(
                    pd.DataFrame(
                        {
                            "bus": [bus],
                            "control": ["PQ"],
                            "p_nom": [sum(comps_disp.p_nom)],
                            "fluctuating": [False],
                        },
                        index=[lv_grid_name + "_dispatchable"],
                    )
                )
                aggregated_elements[
                    lv_grid_name + "_dispatchable"
                ] = comps_disp.index.values
        elif aggregate_generators == "all":
            comps_aggr = pd.DataFrame(
                {
                    "bus": [bus],
                    "control": ["PQ"],
                    "p_nom": [sum(comps.p_nom)],
                    "fluctuating": [
                        True
                        if (comps.type.isin(flucts)).all()
                        else False
                        if ~comps.type.isin(flucts).any()
                        else "Mixed"
                    ],
                },
                index=[lv_grid_name + "_generators"],
            )
            aggregated_elements[
                lv_grid_name + "_generators"
            ] = comps.index.values
        else:
            raise ValueError("Aggregation type for generators invalid.")
        lv_components[comp] = lv_components[comp].append(comps_aggr)
    elif comp is "StorageUnit":
        if aggregate_storages == None:
            comps_aggr = comps.loc[:, ["bus", "control"]]
        elif aggregate_storages == "all":
            comps_aggr = pd.DataFrame(
                {"bus": [bus], "control": ["PQ"]},
                index=[lv_grid_name + "_storages"],
            )
            aggregated_elements[
                lv_grid_name + "_storages"
            ] = comps.index.values
        else:
            raise ValueError("Aggregation type for storages invalid.")
        lv_components[comp] = lv_components[comp].append(comps_aggr)
    else:
        raise ValueError("Component type not defined.")

    return aggregated_elements


def _get_timeseries_with_aggregated_elements(
    edisgo_obj, timesteps, element_type, elements, aggr_dict
):
    """
    Creates timeseries for aggregated LV components by summing up the single
    timeseries and adding the respective entry to edisgo_obj.timeseries.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        the eDisGo network container
    timesteps: timesteps of format :pandas:`pandas.Timestamp<timestamp>`
        index timesteps for component's load or generation timeseries
    element_type: str
        type of element which was aggregated. Can be 'loads', 'generators' or
        'storage_units'
    elements: `pandas.DataFrame<dataframe>`
        component dataframe of all elements for which timeseries are added
    aggr_dict: dict
        dictionary containing aggregated elements as values and the
        representing new component as key. See :meth:`_append_lv_components`
        for structure of dictionary.

    Returns
    -------
    tuple of `pandas.DataFrame<dataframe>`
        active and reactive power timeseries for chosen elements. Dataframes
        with timesteps as index and name of elements as columns.
    """
    non_aggregated_elements = elements[~elements.isin(aggr_dict.keys())]
    # get timeseries for non aggregated generators
    elements_timeseries_active = getattr(
        edisgo_obj.timeseries, element_type + "_active_power"
    ).loc[timesteps, non_aggregated_elements]
    elements_timeseries_reactive = getattr(
        edisgo_obj.timeseries, element_type + "_reactive_power"
    ).loc[timesteps, non_aggregated_elements]
    # append timeseries for aggregated generators
    for aggr_gen in aggr_dict.keys():
        elements_timeseries_active[aggr_gen] = (
            getattr(edisgo_obj.timeseries, element_type + "_active_power")
            .loc[timesteps, aggr_dict[aggr_gen]]
            .sum(axis=1)
        )
        elements_timeseries_reactive[aggr_gen] = (
            getattr(edisgo_obj.timeseries, element_type + "_reactive_power")
            .loc[timesteps, aggr_dict[aggr_gen]]
            .sum(axis=1)
        )
    return elements_timeseries_active, elements_timeseries_reactive


def _buses_voltage_set_point(edisgo_obj, buses, slack_bus, timesteps):
    """
    Time series in PyPSA compatible format for bus instances

    Set all buses except for the slack bus to voltage of 1 p.u. (it is assumed
    this setting is entirely ignored during solving the power flow problem).
    The slack bus voltage is set based on a given HV/MV transformer offset and
    a control deviation, both defined in the config files. The control
    deviation is added to the offset in the reverse power flow case and
    subtracted from the offset in the heavy load flow case.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
    buses : list
        Buses names
    slack_bus : str

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    # set all buses to nominal voltage
    v_nom = pd.DataFrame(1, columns=buses, index=timesteps)

    # set slack bus to operational voltage (includes offset and control
    # deviation)
    control_deviation = edisgo_obj.config[
        "grid_expansion_allowed_voltage_deviations"
    ]["hv_mv_trafo_control_deviation"]
    if control_deviation != 0:
        control_deviation_ts = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: control_deviation
            if _ == "feedin_case"
            else -control_deviation
        ).loc[
            timesteps
        ]
    else:
        control_deviation_ts = pd.Series(0, index=timesteps)

    slack_voltage_pu = (
        control_deviation_ts
        + 1
        + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "hv_mv_trafo_offset"
        ]
    )

    v_nom.loc[timesteps, slack_bus] = slack_voltage_pu

    return v_nom


def _pypsa_bus_timeseries(network, buses, timesteps):
    """
    Todo: remove?
    Time series in PyPSA compatible format for bus instances

    Set all buses except for the slack bus to voltage of 1 pu (it is assumed
    this setting is entirely ignored during solving the power flow problem).
    This slack bus is set to an operational voltage which is typically greater
    than nominal voltage plus a control deviation.
    The control deviation is always added positively to the operational voltage.
    For example, the operational voltage (offset) is set to 1.025 pu plus the
    control deviation of 0.015 pu. This adds up to a set voltage of the slack
    bus of 1.04 pu.

    .. warning::

        Voltage settings for the slack bus defined by this function assume the
        feedin case (reverse power flow case) as the worst-case for the power
        system. Thus, the set point for the slack is always greater 1.


    Parameters
    ----------
    network : Topology
        The eDisGo topology topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
    buses : list
        Buses names

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    # get slack bus label
    slack_bus = "_".join(["Bus", network.mv_grid.station.__repr__(side="mv")])

    # set all buses (except slack bus) to nominal voltage
    v_set_dict = {bus: 1 for bus in buses if bus != slack_bus}

    # Set slack bus to operational voltage (includes offset and control
    # deviation
    control_deviation = network.config[
        "grid_expansion_allowed_voltage_deviations"
    ]["hv_mv_trafo_control_deviation"]
    if control_deviation != 0:
        control_deviation_ts = network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: control_deviation
            if _ == "feedin_case"
            else -control_deviation
        )
    else:
        control_deviation_ts = 0

    slack_voltage_pu = (
        control_deviation_ts
        + 1
        + network.config["grid_expansion_allowed_voltage_deviations"][
            "hv_mv_trafo_offset"
        ]
    )

    v_set_dict.update({slack_bus: slack_voltage_pu})

    # Convert to PyPSA compatible dataframe
    v_set_df = pd.DataFrame(v_set_dict, index=timesteps)

    return v_set_df


def _pypsa_generator_timeseries_aggregated_at_lv_station(network, timesteps):
    """
    Todo: remove?
    Aggregates generator time series per generator subtype and LV topology.

    Parameters
    ----------
    network : Topology
        The eDisGo topology topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.

    Returns
    -------
    tuple of :pandas:`pandas.DataFrame<dataframe>`
        Tuple of size two containing DataFrames that represent

            1. 'p_set' of aggregated Generation per subtype at each LV station
            2. 'q_set' of aggregated Generation per subtype at each LV station

    """

    generation_p = []
    generation_q = []

    for lv_grid in network.mv_grid.lv_grids:
        # Determine aggregated generation at LV stations
        generation = {}
        for gen in lv_grid.generators:
            # for type in gen.type:
            #     for subtype in gen.subtype:
            gen_name = "_".join(
                [
                    gen.type,
                    gen.subtype,
                    "aggregated",
                    "LV_grid",
                    str(lv_grid.id),
                ]
            )

            generation.setdefault(gen.type, {})
            generation[gen.type].setdefault(gen.subtype, {})
            generation[gen.type][gen.subtype].setdefault("timeseries_p", [])
            generation[gen.type][gen.subtype].setdefault("timeseries_q", [])
            generation[gen.type][gen.subtype]["timeseries_p"].append(
                gen.pypsa_timeseries("p")
                .rename(gen_name)
                .to_frame()
                .loc[timesteps]
            )
            generation[gen.type][gen.subtype]["timeseries_q"].append(
                gen.pypsa_timeseries("q")
                .rename(gen_name)
                .to_frame()
                .loc[timesteps]
            )

        for k_type, v_type in generation.items():
            for k_type, v_subtype in v_type.items():
                col_name = v_subtype["timeseries_p"][0].columns[0]
                generation_p.append(
                    pd.concat(v_subtype["timeseries_p"], axis=1)
                    .sum(axis=1)
                    .rename(col_name)
                    .to_frame()
                )
                generation_q.append(
                    pd.concat(v_subtype["timeseries_q"], axis=1)
                    .sum(axis=1)
                    .rename(col_name)
                    .to_frame()
                )

    return generation_p, generation_q


def _pypsa_load_timeseries_aggregated_at_lv_station(network, timesteps):
    """
    Todo: remove?
    Aggregates load time series per sector and LV topology.

    Parameters
    ----------
    network : Topology
        The eDisGo topology topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.

    Returns
    -------
    tuple of :pandas:`pandas.DataFrame<dataframe>`
        Tuple of size two containing DataFrames that represent

            1. 'p_set' of aggregated Load per sector at each LV station
            2. 'q_set' of aggregated Load per sector at each LV station

    """
    # ToDo: Load.pypsa_timeseries is not differentiated by sector so this
    # function will not work (either change here and in
    # add_aggregated_lv_components or in Load class)

    load_p = []
    load_q = []

    for lv_grid in network.mv_grid.lv_grids:
        # Determine aggregated load at LV stations
        load = {}
        for lo in lv_grid.graph.nodes_by_attribute("load"):
            for sector, val in lo.consumption.items():
                load.setdefault(sector, {})
                load[sector].setdefault("timeseries_p", [])
                load[sector].setdefault("timeseries_q", [])

                load[sector]["timeseries_p"].append(
                    lo.pypsa_timeseries("p")
                    .rename(repr(lo))
                    .to_frame()
                    .loc[timesteps]
                )
                load[sector]["timeseries_q"].append(
                    lo.pypsa_timeseries("q")
                    .rename(repr(lo))
                    .to_frame()
                    .loc[timesteps]
                )

        for sector, val in load.items():
            load_p.append(
                pd.concat(val["timeseries_p"], axis=1)
                .sum(axis=1)
                .rename("_".join(["Load", sector, repr(lv_grid)]))
                .to_frame()
            )
            load_q.append(
                pd.concat(val["timeseries_q"], axis=1)
                .sum(axis=1)
                .rename("_".join(["Load", sector, repr(lv_grid)]))
                .to_frame()
            )

    return load_p, load_q


def _check_topology(components):
    # Todo: remove?
    buses = components["Bus"].index.tolist()
    line_buses = (
        components["Line"]["bus0"].tolist()
        + components["Line"]["bus1"].tolist()
    )
    load_buses = components["Load"]["bus"].tolist()
    generator_buses = components["Generator"]["bus"].tolist()
    transformer_buses = (
        components["Transformer"]["bus0"].tolist()
        + components["Transformer"]["bus1"].tolist()
    )

    buses_to_check = (
        line_buses + load_buses + generator_buses + transformer_buses
    )

    missing_buses = []

    missing_buses.extend([_ for _ in buses_to_check if _ not in buses])

    if missing_buses:
        raise ValueError(
            "Buses {buses} are not defined.".format(buses=missing_buses)
        )

    # check if there are duplicate components and print them
    for k, comps in components.items():
        if len(list(comps.index.values)) != len(set(comps.index.values)):
            raise ValueError(
                "There are duplicates in the {comp} list: {dupl}".format(
                    comp=k,
                    dupl=[
                        item
                        for item, count in collections.Counter(
                            comps.index.values
                        ).items()
                        if count > 1
                    ],
                )
            )


def _check_integrity_of_pypsa(pypsa_network):
    """
    Checks whether the provided pypsa network is calculable. Isolated nodes,
    duplicate labels and completeness of buses and branch elements are checked.

    Parameters
    ----------
    pypsa_network: :pypsa:`pypsa.Network<network>`
        The `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_ container.
    """

    # check for sub-networks
    subgraphs = list(
        pypsa_network.graph().subgraph(c)
        for c in connected_components(pypsa_network.graph())
    )
    pypsa_network.determine_network_topology()

    if len(subgraphs) > 1 or len(pypsa_network.sub_networks) > 1:
        raise ValueError("The pypsa graph has isolated nodes or edges.")

    # check consistency of topology and time series data
    generators_ts_p_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t["p_set"].columns.tolist()
        )
    ]
    generators_ts_q_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t["q_set"].columns.tolist()
        )
    ]
    loads_ts_p_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t["p_set"].columns.tolist()
        )
    ]
    loads_ts_q_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t["q_set"].columns.tolist()
        )
    ]
    bus_v_set_missing = pypsa_network.buses.loc[
        ~pypsa_network.buses.index.isin(
            pypsa_network.buses_t["v_mag_pu_set"].columns.tolist()
        )
    ]

    # Comparison of generators excludes slack generators (have no time series)
    if not generators_ts_p_missing.empty and not all(
        generators_ts_p_missing["control"] == "Slack"
    ):
        raise ValueError(
            "Following generators have no `p_set` time series "
            "{generators}".format(generators=generators_ts_p_missing)
        )

    if not generators_ts_q_missing.empty and not all(
        generators_ts_q_missing["control"] == "Slack"
    ):
        raise ValueError(
            "Following generators have no `q_set` time series "
            "{generators}".format(generators=generators_ts_q_missing)
        )

    if not loads_ts_p_missing.empty:
        raise ValueError(
            "Following loads have no `p_set` time series "
            "{loads}".format(loads=loads_ts_p_missing)
        )

    if not loads_ts_q_missing.empty:
        raise ValueError(
            "Following loads have no `q_set` time series "
            "{loads}".format(loads=loads_ts_q_missing)
        )

    if not bus_v_set_missing.empty:
        raise ValueError(
            "Following loads have no `v_mag_pu_set` time series "
            "{buses}".format(buses=bus_v_set_missing)
        )

    # check for duplicate labels (of components)
    duplicated_labels = []
    if any(pypsa_network.buses.index.duplicated()):
        duplicated_labels.append(
            pypsa_network.buses.index[pypsa_network.buses.index.duplicated()]
        )
    if any(pypsa_network.generators.index.duplicated()):
        duplicated_labels.append(
            pypsa_network.generators.index[
                pypsa_network.generators.index.duplicated()
            ]
        )
    if any(pypsa_network.loads.index.duplicated()):
        duplicated_labels.append(
            pypsa_network.loads.index[pypsa_network.loads.index.duplicated()]
        )
    if any(pypsa_network.transformers.index.duplicated()):
        duplicated_labels.append(
            pypsa_network.transformers.index[
                pypsa_network.transformers.index.duplicated()
            ]
        )
    if any(pypsa_network.lines.index.duplicated()):
        duplicated_labels.append(
            pypsa_network.lines.index[pypsa_network.lines.index.duplicated()]
        )
    if duplicated_labels:
        raise ValueError(
            "{labels} have duplicate entry in "
            "one of the components dataframes".format(labels=duplicated_labels)
        )

    # duplicate p_sets and q_set
    duplicate_p_sets = []
    duplicate_q_sets = []
    if any(pypsa_network.loads_t["p_set"].columns.duplicated()):
        duplicate_p_sets.append(
            pypsa_network.loads_t["p_set"].columns[
                pypsa_network.loads_t["p_set"].columns.duplicated()
            ]
        )
    if any(pypsa_network.loads_t["q_set"].columns.duplicated()):
        duplicate_q_sets.append(
            pypsa_network.loads_t["q_set"].columns[
                pypsa_network.loads_t["q_set"].columns.duplicated()
            ]
        )

    if any(pypsa_network.generators_t["p_set"].columns.duplicated()):
        duplicate_p_sets.append(
            pypsa_network.generators_t["p_set"].columns[
                pypsa_network.generators_t["p_set"].columns.duplicated()
            ]
        )
    if any(pypsa_network.generators_t["q_set"].columns.duplicated()):
        duplicate_q_sets.append(
            pypsa_network.generators_t["q_set"].columns[
                pypsa_network.generators_t["q_set"].columns.duplicated()
            ]
        )

    if duplicate_p_sets:
        raise ValueError(
            "{labels} have duplicate entry in "
            "generators_t['p_set']"
            " or loads_t['p_set']".format(labels=duplicate_p_sets)
        )
    if duplicate_q_sets:
        raise ValueError(
            "{labels} have duplicate entry in "
            "generators_t['q_set']"
            " or loads_t['q_set']".format(labels=duplicate_q_sets)
        )

    # find duplicate v_mag_set entries
    duplicate_v_mag_set = []
    if any(pypsa_network.buses_t["v_mag_pu_set"].columns.duplicated()):
        duplicate_v_mag_set.append(
            pypsa_network.buses_t["v_mag_pu_set"].columns[
                pypsa_network.buses_t["v_mag_pu_set"].columns.duplicated()
            ]
        )

    if duplicate_v_mag_set:
        raise ValueError(
            "{labels} have duplicate entry in buses_t".format(
                labels=duplicate_v_mag_set
            )
        )


def process_pfa_results(edisgo, pypsa, timesteps):
    """
    Passing power flow results from PyPSA to
    :class:`~.network.results.Results`.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
    pypsa : :pypsa:`pypsa.Network<network>`
        The PyPSA `Network container
        <https://www.pypsa.org/doc/components.html#network>`_
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Time steps for which latest power flow analysis was conducted and
        for which to retrieve pypsa results.

    Notes
    -----
    P and Q are returned from the line ending/transformer side with highest
    apparent power S, exemplary written as

    .. math::
        S_{max} = max(\sqrt{P_0^2 + Q_0^2}, \sqrt{P_1^2 + Q_1^2}) \\
        P = P_0 P_1(S_{max}) \\
        Q = Q_0 Q_1(S_{max})

    See Also
    --------
    :class:`~.network.results.Results` to understand how results of power flow
    analysis are structured in eDisGo.

    """
    # get the absolute losses in the system (in MW and Mvar)
    # subtracting total generation (including slack) from total load
    grid_losses = {
        "p": (
            pypsa.generators_t["p"].sum(axis=1)
            - pypsa.loads_t["p"].sum(axis=1)
        ),
        "q": (
            pypsa.generators_t["q"].sum(axis=1)
            - pypsa.loads_t["q"].sum(axis=1)
        ),
    }
    edisgo.results.grid_losses = pd.DataFrame(grid_losses).loc[timesteps, :]

    # get slack results (HV/MV exchanges) in MW and Mvar
    grid_exchanges = {
        "p": (pypsa.generators_t["p"]["Generator_slack"]),
        "q": (pypsa.generators_t["q"]["Generator_slack"]),
    }
    edisgo.results.hv_mv_exchanges = pd.DataFrame(grid_exchanges).loc[
        timesteps, :
    ]

    # get P and Q of lines and transformers in MW and Mvar
    q0 = pd.concat(
        [np.abs(pypsa.lines_t["q0"]), np.abs(pypsa.transformers_t["q0"])],
        axis=1,
    ).loc[timesteps, :]
    q1 = pd.concat(
        [np.abs(pypsa.lines_t["q1"]), np.abs(pypsa.transformers_t["q1"])],
        axis=1,
    ).loc[timesteps, :]
    p0 = pd.concat(
        [np.abs(pypsa.lines_t["p0"]), np.abs(pypsa.transformers_t["p0"])],
        axis=1,
    ).loc[timesteps, :]
    p1 = pd.concat(
        [np.abs(pypsa.lines_t["p1"]), np.abs(pypsa.transformers_t["p1"])],
        axis=1,
    ).loc[timesteps, :]
    # determine apparent power at line endings/transformer sides
    s0 = np.hypot(p0, q0)
    s1 = np.hypot(p1, q1)
    # choose P and Q from line ending with max(s0,s1)
    edisgo.results.pfa_p = p0.where(s0 > s1, p1)
    edisgo.results.pfa_q = q0.where(s0 > s1, q1)

    # calculate line currents in kA
    lines_bus0 = pypsa.lines["bus0"].to_dict()
    bus0_v_mag_pu = (
        pypsa.buses_t["v_mag_pu"].T.loc[list(lines_bus0.values()), :].copy()
    )
    bus0_v_mag_pu.index = list(lines_bus0.keys())
    edisgo.results._i_res = np.hypot(
        pypsa.lines_t["p0"], pypsa.lines_t["q0"]
    ).truediv(pypsa.lines["v_nom"] * bus0_v_mag_pu.T, axis="columns") / sqrt(3)

    # get voltage results in kV
    edisgo.results._v_res = pypsa.buses_t["v_mag_pu"]
