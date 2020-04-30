import math
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import (
    _dijkstra as dijkstra_shortest_path_length,
)
import pandas as pd
import numpy as np

from edisgo.network.grids import LVGrid, MVGrid

import logging

logger = logging.getLogger("edisgo")


def reinforce_mv_lv_station_overloading(edisgo_obj, critical_stations):
    """
    Reinforce MV/LV substations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
        }

    """
    transformers_changes = _station_overloading(
        edisgo_obj, critical_stations, voltage_level="lv")

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to "
            "overloading issues.".format(
                str(len(transformers_changes["added"]))))

    return transformers_changes


def reinforce_hv_mv_station_overloading(edisgo_obj, critical_stations):
    """
    Reinforce HV/MV station due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded HV/MV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
        }

    """
    transformers_changes = _station_overloading(
        edisgo_obj, critical_stations, voltage_level="mv")

    if transformers_changes["added"]:
        logger.debug(
            "==> MV station has been reinforced due to overloading issues."
        )

    return transformers_changes


def _station_overloading(edisgo_obj, critical_stations, voltage_level):
    """
    Reinforce stations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.
    voltage_level : str
        Voltage level, over-loading is checked for. Possible options are
        "mv" or "lv".

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
        }

    """
    if voltage_level == "lv":
        try:
            standard_transformer = edisgo_obj.topology.equipment_data[
                "lv_transformers"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "mv_lv_transformer"
                ]
            ]
        except KeyError:
            raise KeyError(
                "Standard MV/LV transformer is not in equipment list.")
    elif voltage_level == "mv":
        try:
            standard_transformer = edisgo_obj.topology.equipment_data[
                "mv_transformers"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "hv_mv_transformer"
                ]
            ]
        except KeyError:
            raise KeyError(
                "Standard HV/MV transformer is not in equipment list.")
    else:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_level' in "
            "function _station_overloading. Try 'mv' or "
            "'lv'.".format(voltage_level)
        )

    transformers_changes = {"added": {}, "removed": {}}
    for grid_name in critical_stations.index:
        grid = edisgo_obj.topology._grids[grid_name]
        # list of maximum power of each transformer in the station
        s_max_per_trafo = grid.transformers_df.s_nom
        # missing capacity
        s_trafo_missing = critical_stations.at[grid_name, "s_missing"]

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            new_transformers = grid.transformers_df.loc[
                grid.transformers_df[s_max_per_trafo >= s_trafo_missing][
                    "s_nom"
                ].idxmin()
            ]
            name = new_transformers.name.split("_")
            name.insert(-1, "reinforced")
            name[-1] = len(grid.transformers_df) + 1
            new_transformers.name = "_".join([str(_) for _ in name])

            # add new transformer to list of added transformers
            transformers_changes["added"][grid_name] = [
                new_transformers.name
            ]
        else:
            # get any transformer to get attributes for new transformer from
            duplicated_transformer = grid.transformers_df.iloc[0]
            name = duplicated_transformer.name.split("_")
            name.insert(-1, "reinforced")
            duplicated_transformer.s_nom = standard_transformer.S_nom
            duplicated_transformer.type_info = standard_transformer.name
            if voltage_level == "lv":
                duplicated_transformer.r_pu = standard_transformer.r_pu
                duplicated_transformer.x_pu = standard_transformer.x_pu

            # set up as many new transformers as needed
            number_transformers = math.ceil(
                (s_trafo_missing + s_max_per_trafo.sum()) / standard_transformer.S_nom
            )
            new_transformers = pd.DataFrame()
            for i in range(number_transformers):
                name[-1] = i + 1
                duplicated_transformer.name = "_".join([str(_) for _ in name])
                new_transformers = new_transformers.append(
                    duplicated_transformer
                )

            # add new transformer to list of added transformers
            transformers_changes["added"][
                grid_name
            ] = new_transformers.index.values
            # add previous transformers to list of removed transformers
            transformers_changes["removed"][
                grid_name
            ] = grid.transformers_df.index.values
            # remove previous transformers from topology
            if voltage_level == "lv":
                edisgo_obj.topology.transformers_df.drop(
                    grid.transformers_df.index.values, inplace=True
                )
            else:
                edisgo_obj.topology.transformers_hvmv_df.drop(
                    grid.transformers_df.index.values, inplace=True
                )

        # add new transformers to topology
        if voltage_level == "lv":
            edisgo_obj.topology.transformers_df = (
                edisgo_obj.topology.transformers_df.append(
                    new_transformers
                )
            )
        else:
            edisgo_obj.topology.transformers_hvmv_df = (
                edisgo_obj.topology.transformers_hvmv_df.append(
                    new_transformers
                )
            )
    return transformers_changes


def reinforce_mv_lv_station_voltage_issues(edisgo_obj, critical_stations):
    """
    Reinforce MV/LV substations due to voltage issues.

    A parallel standard transformer is installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.LVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with station's voltage
        deviation from allowed lower or upper voltage limit as value.
        Index of the dataframe is the station with voltage issues.
        Columns are 'v_diff_max' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    :obj:`dict`
        Dictionary with added transformers in the form::

            {'added': {'Grid_1': ['transformer_reinforced_1',
                                  ...,
                                  'transformer_reinforced_x'],
                       'Grid_10': ['transformer_reinforced_10']
                       }
            }

    """

    # get parameters for standard transformer
    try:
        standard_transformer = edisgo_obj.topology.equipment_data[
            "lv_transformers"
        ].loc[
            edisgo_obj.config["grid_expansion_standard_equipment"][
                "mv_lv_transformer"
            ]
        ]
    except KeyError:
        raise KeyError("Standard MV/LV transformer is not in equipment list.")

    transformers_changes = {"added": {}}
    for grid_name in critical_stations.keys():
        grid = edisgo_obj.topology._grids[grid_name]
        # get any transformer to get attributes for new transformer from
        duplicated_transformer = grid.transformers_df.iloc[0]
        # change transformer parameters
        name = duplicated_transformer.name.split("_")
        name.insert(-1, "reinforced")
        name[-1] = len(grid.transformers_df) + 1
        duplicated_transformer.name = "_".join([str(_) for _ in name])
        duplicated_transformer.s_nom = standard_transformer.S_nom
        duplicated_transformer.r_pu = standard_transformer.r_pu
        duplicated_transformer.x_pu = standard_transformer.x_pu
        duplicated_transformer.type_info = standard_transformer.name
        # add new transformer to topology
        edisgo_obj.topology.transformers_df = edisgo_obj.topology.transformers_df.append(
            duplicated_transformer
        )
        transformers_changes["added"][grid_name] = [
            duplicated_transformer.name
        ]

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to voltage "
            "issues.".format(
                len(transformers_changes["added"])
            )
        )

    return transformers_changes


def reinforce_lines_voltage_issues(edisgo_obj, grid, crit_nodes):
    """
    Reinforce lines in MV and LV topology due to voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all nodes with voltage issues in the grid and
        their maximal deviations from allowed lower or upper voltage limits
        sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'v_diff_max' containing the maximum
        absolute voltage deviation as float and 'time_index' containing the
        corresponding time step the voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`. Index of the dataframe are the
        names of all buses with voltage issues.

    Returns
    -------
    Dictionary with name of lines as keys and the corresponding number of lines
    added as values.

    Notes
    -----
    Reinforce measures:

    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. Install parallel standard line

    In LV grids only lines outside buildings are reinforced; loads and
    generators in buildings cannot be directly connected to the MV/LV station.

    In MV grids lines can only be disconnected at LV stations because they
    have switch disconnectors needed to operate the lines as half rings (loads
    in MV would be suitable as well because they have a switch bay (Schaltfeld)
    but loads in dingo are only connected to MV busbar). If there is no
    suitable LV station the generator is directly connected to the MV busbar.
    There is no need for a switch disconnector in that case because generators
    don't need to be n-1 safe.

    """

    def change_line_to_standard_line(line_name):
        edisgo_obj.topology._lines_df.at[
            line_name, "type_info"
        ] = standard_line.name
        edisgo_obj.topology._lines_df.at[line_name, "num_parallel"] = 1
        edisgo_obj.topology._lines_df.at[line_name, "kind"] = "cable"
        edisgo_obj.topology._lines_df.at[line_name, "r"] = (
            standard_line.R_per_km * grid.lines_df.at[line_name, "length"]
        )
        edisgo_obj.topology._lines_df.at[line_name, "x"] = (
            standard_line.L_per_km
            * 2 * np.pi * 50
            / 1e3
            * grid.lines_df.at[line_name, "length"]
        )
        edisgo_obj.topology._lines_df.at[line_name, "s_nom"] = (
            np.sqrt(3) * standard_line.U_n * standard_line.I_max_th
        )
        lines_changes[line_name] = 1

    # load standard line data
    if isinstance(grid, LVGrid):
        try:
            standard_line = edisgo_obj.topology.equipment_data[
                "lv_cables"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "lv_line"
                ]
            ]
        except KeyError:
            raise KeyError("Chosen standard LV line is not in equipment list.")
    elif isinstance(grid, MVGrid):
        try:
            standard_line = edisgo_obj.topology.equipment_data[
                "mv_cables"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "mv_line"
                ]
            ]
            standard_line.U_n = grid.nominal_voltage
        except KeyError:
            raise KeyError("Chosen standard MV line is not in equipment list.")
    else:
        raise ValueError("Inserted grid is invalid.")

    # find path to each node in order to find node with voltage issues farthest
    # away from station in each feeder
    station_node = grid.transformers_df.bus1.iloc[0]
    graph = grid.graph
    paths = {}
    nodes_feeder = {}
    for node in crit_nodes.index:
        path = nx.shortest_path(graph, station_node, node)
        paths[node] = path
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logging.error(
                "Voltage issues at busbar in LV network {} should have "
                "been solved in previous steps.".format(grid)
            )
        nodes_feeder.setdefault(path[1], []).append(node)

    lines_changes = {}
    for repr_node in nodes_feeder.keys():

        # find node farthest away
        get_weight = lambda u, v, data: data["length"]
        path_length = 0
        for n in nodes_feeder[repr_node]:
            path_length_dict_tmp = dijkstra_shortest_path_length(
                graph, station_node, get_weight, target=n
            )
            if path_length_dict_tmp[n] > path_length:
                node = n
                path_length = path_length_dict_tmp[n]
                path_length_dict = path_length_dict_tmp
        path = paths[node]

        # find first node in path that exceeds 2/3 of the line length
        # from station to critical node farthest away from the station
        node_2_3 = next(
            j
            for j in path
            if path_length_dict[j] >= path_length_dict[node] * 2 / 3
        )

        # if LVGrid: check if node_2_3 is outside of a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_2_3].in_building)
                and grid.buses_df.loc[node_2_3].in_building
            ):
                node_2_3 = path[path.index(node_2_3) - 1]
                # break if node is station
                if node_2_3 is path[0]:
                    logger.error("Could not reinforce voltage issue.")
                    break

        # if MVGrid: check if node_2_3 is LV station and if not find
        # next LV station
        else:
            while (
                node_2_3
                not in edisgo_obj.topology.transformers_df.bus0.values
            ):
                try:
                    # try to find LVStation behind node_2_3
                    node_2_3 = path[path.index(node_2_3) + 1]
                except IndexError:
                    # if no LVStation between node_2_3 and node with
                    # voltage problem, connect node directly to
                    # MVStation
                    node_2_3 = node
                    break

        # if node_2_3 is a representative (meaning it is already
        # directly connected to the station), line cannot be
        # disconnected and must therefore be reinforced
        if node_2_3 in nodes_feeder.keys():
            crit_line_name = graph.get_edge_data(
                station_node, node_2_3
            )["branch_name"]
            crit_line = grid.lines_df.loc[crit_line_name]

            # if critical line is already a standard line install one
            # more parallel line
            if crit_line.type_info == standard_line.name:
                num_parallel_pre = grid.lines_df.at[
                    crit_line_name, "num_parallel"
                ]
                edisgo_obj.topology._lines_df.at[
                    crit_line_name, "num_parallel"
                ] += 1
                edisgo_obj.topology._lines_df.at[
                    crit_line_name, "r"
                ] = (
                    grid.lines_df.at[crit_line_name, "r"]
                    * num_parallel_pre
                    / (num_parallel_pre + 1)
                )
                edisgo_obj.topology._lines_df.at[
                    crit_line_name, "x"
                ] = (
                    grid.lines_df.at[crit_line_name, "x"]
                    * num_parallel_pre
                    / (num_parallel_pre + 1)
                )
                lines_changes[crit_line_name] = 1

            # if critical line is not yet a standard line replace old
            # line by a standard line
            else:
                # number of parallel standard lines could be calculated
                # following [2] p.103; for now number of parallel
                # standard lines is iterated
                change_line_to_standard_line(crit_line_name)

        # if node_2_3 is not a representative, disconnect line
        else:
            # get line between node_2_3 and predecessor node (that is
            # closer to the station)
            pred_node = path[path.index(node_2_3) - 1]
            crit_line_name = graph.get_edge_data(node_2_3, pred_node)[
                "branch_name"
            ]
            if grid.lines_df.at[crit_line_name, "bus0"] == pred_node:
                edisgo_obj.topology._lines_df.at[
                    crit_line_name, "bus0"
                ] = station_node
            elif grid.lines_df.at[crit_line_name, "bus1"] == pred_node:
                edisgo_obj.topology._lines_df.at[
                    crit_line_name, "bus1"
                ] = station_node
            else:
                raise ValueError(
                    "Bus not in line buses. " "Please check."
                )
            # change line length and type
            edisgo_obj.topology._lines_df.at[
                crit_line_name, "length"
            ] = path_length_dict[node_2_3]
            change_line_to_standard_line(crit_line_name)

    if not lines_changes:
        logger.debug(
            "==> {} branche(s) was/were reinforced due to voltage "
            "issues.".format(
                len(lines_changes)
            )
        )

    return lines_changes


def reinforce_branches_overloading(edisgo_obj, crit_lines):
    """
    Reinforce MV or LV topology due to overloading.
    
    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
    crit_lines : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the names of over-loaded lines. Columns are
        'max_rel_overload' containing the maximum relative over-loading as
        float and 'time_index' containing the corresponding time step the
        over-loading occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    Dictionary with name of lines and the number of Lines added.
        
    Notes
    -----
    Reinforce measures:

    1. Install parallel line of the same type as the existing line (Only if
       line is a cable, not an overhead line. Otherwise a standard equipment
       cable is installed right away.)
    2. Remove old line and install as many parallel standard lines as
       needed.

    """

    lines_changes = {}
    # reinforce mv lines
    lines_changes = reinforce_lines_overloaded_per_grid_level(
        edisgo_obj, "mv", crit_lines, lines_changes
    )
    # reinforce lv lines
    lines_changes = reinforce_lines_overloaded_per_grid_level(
        edisgo_obj, "lv", crit_lines, lines_changes
    )

    if not crit_lines.empty:
        logger.debug(
            "==> {} branche(s) was/were reinforced ".format(
                crit_lines.shape[0]
            )
            + "due to over-loading issues."
        )

    return lines_changes


def reinforce_lines_overloaded_per_grid_level(
    edisgo_obj, grid_level, crit_lines, lines_changes
):
    def reinforce_standard_lines(relevant_lines):
        # Todo: make sure, the other parameters are the same as well
        lines_standard = relevant_lines.loc[
            relevant_lines.type_info == standard_line.name
        ]
        number_parallel_lines = np.ceil(
            crit_lines.max_rel_overload[lines_standard.index]
            * lines_standard.num_parallel
        )
        number_parallel_lines_pre = edisgo_obj.topology.lines_df.loc[
            lines_standard.index, "num_parallel"
        ]
        edisgo_obj.topology.lines_df.loc[
            lines_standard.index, "num_parallel"
        ] = number_parallel_lines
        edisgo_obj.topology.lines_df.loc[lines_standard.index, "x"] = (
            edisgo_obj.topology.lines_df.loc[lines_standard.index, "x"]
            * number_parallel_lines_pre
            / number_parallel_lines
        )
        edisgo_obj.topology.lines_df.loc[lines_standard.index, "r"] = (
            edisgo_obj.topology.lines_df.loc[lines_standard.index, "r"]
            * number_parallel_lines_pre
            / number_parallel_lines
        )
        lines_changes.update(
            (number_parallel_lines - number_parallel_lines_pre).to_dict()
        )
        lines_default = relevant_lines.loc[
            ~relevant_lines.index.isin(lines_standard.index)
        ]
        return lines_default

    def reinforce_single_lines(lines_default):
        lines_single = (
            lines_default.loc[lines_default.num_parallel == 1]
            .loc[lines_default.kind == "cable"]
            .loc[crit_lines.max_rel_overload < 2]
        )
        edisgo_obj.topology.lines_df.loc[
            lines_single.index, "num_parallel"
        ] = 2
        edisgo_obj.topology.lines_df.loc[lines_single.index, "r"] = (
            edisgo_obj.topology.lines_df.loc[lines_single.index, "r"] / 2
        )
        edisgo_obj.topology.lines_df.loc[lines_single.index, "x"] = (
            edisgo_obj.topology.lines_df.loc[lines_single.index, "x"] / 2
        )
        lines_changes.update({_: 1 for _ in lines_single.index})
        lines_default = lines_default.loc[
            ~lines_default.index.isin(lines_single.index)
        ]
        return lines_default

    def reinforce_default_lines(lines_default):
        number_parallel_lines = np.ceil(
            lines_default.s_nom
            * crit_lines.loc[lines_default.index, "max_rel_overload"]
            / (math.sqrt(3) * standard_line.U_n * standard_line.I_max_th)
        )
        edisgo_obj.topology.lines_df.loc[
            lines_default.index, "type_info"
        ] = standard_line.name
        edisgo_obj.topology.lines_df.loc[lines_default.index, "s_nom"] = (
            math.sqrt(3) * standard_line.U_n * standard_line.I_max_th
        )
        edisgo_obj.topology.lines_df.loc[
            lines_default.index, "num_parallel"
        ] = number_parallel_lines
        edisgo_obj.topology.lines_df.loc[lines_default.index, "r"] = (
            standard_line.R_per_km
            * edisgo_obj.topology.lines_df.loc[lines_default.index, "length"]
            / edisgo_obj.topology.lines_df.loc[
                lines_default.index, "num_parallel"
            ]
        )
        omega = 2 * np.pi * edisgo_obj.config["network_parameters"]["freq"]
        edisgo_obj.topology.lines_df.loc[lines_default.index, "x"] = (
            standard_line.L_per_km
            * omega
            * 1e-3
            * edisgo_obj.topology.lines_df.loc[lines_default.index, "length"]
            / edisgo_obj.topology.lines_df.loc[
                lines_default.index, "num_parallel"
            ]
        )
        lines_changes.update(number_parallel_lines.to_dict())

    # load standard line data
    try:
        standard_line = edisgo_obj.topology.equipment_data[
            "{}_cables".format(grid_level)
        ].loc[
            edisgo_obj.config["grid_expansion_standard_equipment"][
                "{}_line".format(grid_level)
            ]
        ]
        # Todo: check voltage of standard line to distinguish between 10
        #  and 20 kV. Remove following part afterwards.
        if grid_level == "mv":
            standard_line.U_n = edisgo_obj.topology.mv_grid.nominal_voltage
    except KeyError:
        print(
            "Chosen standard {} line is not in equipment list.".format(
                grid_level
            )
        )
    # chose lines of right grid level
    relevant_lines = edisgo_obj.topology.lines_df.loc[
        crit_lines[crit_lines.grid_level == grid_level].index
    ]
    # handling of standard lines
    lines_default = reinforce_standard_lines(relevant_lines)
    # handling of cables where adding one cable is sufficient
    lines_default = reinforce_single_lines(lines_default)
    # default lines that haven't been handled so far
    # Todo: removed lines are not handled here unlike for trafos.
    #  Overthink and unify
    reinforce_default_lines(lines_default)

    return lines_changes
