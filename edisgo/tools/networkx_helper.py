import networkx as nx
import pandas as pd


def translate_df_to_graph(buses_df, lines_df, transformers_df=None):
    graph = nx.OrderedGraph()

    buses = buses_df.index
    # add nodes
    graph.add_nodes_from(buses)
    # add branches
    branches = []
    for line_name, line in lines_df.iterrows():
        branches.append(
            (
                line.bus0,
                line.bus1,
                {"branch_name": line_name, "length": line.length},
            )
        )
    if transformers_df is not None:
        for trafo_name, trafo in transformers_df.iterrows():
            branches.append(
                (
                    trafo.bus0,
                    trafo.bus1,
                    {"branch_name": trafo_name, "length": 0},
                )
            )
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
            print('{} % of the buses have been checked'.format(i/len(buses*100)))
    return downstream_node_matrix