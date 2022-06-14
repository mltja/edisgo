import networkx as nx
import pandas as pd
import edisgo.network as nw


def translate_df_to_graph(buses_df, lines_df, transformers_df=None):
    graph = nx.OrderedGraph()

    # add nodes
    buses = [(bus_name, {'pos': (x, y)}) for bus_name, x, y in zip(buses_df.index,
                                                                   buses_df.x,
                                                                   buses_df.y)]
    graph.add_nodes_from(buses)

    # add branches
    branches = []
    for line_name, bus0, bus1, length in zip(lines_df.index,
                                             lines_df.bus0,
                                             lines_df.bus1,
                                             lines_df.length):
        branches.append((bus0, bus1, {"branch_name": line_name, "length": length}))

    if transformers_df is not None:
        for trafo_name, bus0, bus1 in zip(transformers_df.index,
                                          transformers_df.bus0,
                                          transformers_df.bus1):
            branches.append((bus0, bus1, {"branch_name": trafo_name, "length": 0}))

    graph.add_edges_from(branches)

    return graph


def get_downstream_nodes_matrix(edisgo):
    """
    Method that returns matrix M with 0 and 1 entries describing the relation
    of buses within the network. If bus b is descendant of a (assuming the
    station is the root of the radial network) M[a,b] = 1, otherwise M[a,b] = 0.
    The matrix is later used to determine the power flow at the different buses
    by multiplying with the nodal power flow. S_sum = M * s, where s is the
    nodal power vector.

    Note: only works for radial networks.

    :param edisgo_obj:
    :return:
    """
    buses = edisgo.topology.buses_df.index.values
    print('Matrix for {} buses is extracted.'.format(len(buses)))
    tree = \
        nx.bfs_tree(edisgo.to_graph(), edisgo.topology.slack_df.bus.values[0])
    downstream_node_matrix = pd.DataFrame(columns=buses, index=buses)
    downstream_node_matrix.fillna(0, inplace=True)
    i = 0
    for bus in buses:
        ancestors = list(nx.ancestors(tree, bus))
        downstream_node_matrix.loc[ancestors, bus] = 1
        downstream_node_matrix.loc[bus, bus] = 1
        i += 1
        if (i % 10) == 0:
            print('{} % of the buses have been checked'.format(i/len(buses)*100))
    return downstream_node_matrix


def get_downstream_nodes_matrix_iterative(grid):
    """
    Method that returns matrix M with 0 and 1 entries describing the relation
    of buses within the network. If bus b is descendant of a (assuming the
    station is the root of the radial network) M[a,b] = 1, otherwise M[a,b] = 0.
    The matrix is later used to determine the power flow at the different buses
    by multiplying with the nodal power flow. S_sum = M * s, where s is the
    nodal power vector.

    Note: only works for radial networks.

    :param grid: either Topology, MVGrid or LVGrid
    :return:
    Todo: Check version with networkx successor
    """

    def recursive_downstream_node_matrix_filling(current_bus, current_feeder,
                                                 downstream_node_matrix,
                                                 grid,
                                                 visited_buses):
        current_feeder.append(current_bus)
        for neighbor in tree.successors(current_bus):
            if neighbor not in visited_buses and neighbor not in current_feeder:
                recursive_downstream_node_matrix_filling(
                    neighbor, current_feeder, downstream_node_matrix, grid,
                    visited_buses)
        # current_bus = current_feeder.pop()
        downstream_node_matrix.loc[current_feeder, current_bus] = 1
        visited_buses.append(current_bus)
        if len(visited_buses)%10==0:
            print('{} % of the buses have been checked'.format(
                len(visited_buses)/len(buses)*100))
        current_feeder.pop()

    buses = grid.buses_df.index.values
    if str(type(grid)) == str(nw.topology.Topology):
        graph = grid.to_graph()
        slack = grid.mv_grid.station.index[0]
    else:
        graph = grid.graph
        slack = grid.transformers_df.bus1.iloc[0]
    tree = \
        nx.bfs_tree(graph, slack)

    print('Matrix for {} buses is extracted.'.format(len(buses)))
    downstream_node_matrix = pd.DataFrame(columns=buses, index=buses)
    downstream_node_matrix.fillna(0, inplace=True)

    print('Starting iteration.')
    visited_buses = []
    current_feeder = []

    recursive_downstream_node_matrix_filling(slack, current_feeder,
                                             downstream_node_matrix, grid,
                                             visited_buses)

    return downstream_node_matrix


