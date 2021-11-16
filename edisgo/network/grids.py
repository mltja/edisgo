from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np

from edisgo.network.components import Generator, Load, Switch
from edisgo.tools.networkx_helper import translate_df_to_graph


class Grid(ABC):
    """
    Defines a basic grid in eDisGo.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    id : str or int, optional
        Identifier

    """

    def __init__(self, **kwargs):
        self._id = kwargs.get("id", None)
        if isinstance(self._id, float):
            self._id = int(self._id)
        self._edisgo_obj = kwargs.get("edisgo_obj", None)

        self._nominal_voltage = None

    @property
    def id(self):
        return self._id

    @property
    def edisgo_obj(self):
        return self._edisgo_obj

    @property
    def nominal_voltage(self):
        """
        Nominal voltage of network in kV.

        Parameters
        ----------
        nominal_voltage : float

        Returns
        -------
        float
            Nominal voltage of network in kV.

        """
        if self._nominal_voltage is None:
            self._nominal_voltage = self.buses_df.v_nom.max()
        return self._nominal_voltage

    @nominal_voltage.setter
    def nominal_voltage(self, nominal_voltage):
        self._nominal_voltage = nominal_voltage

    @property
    def graph(self):
        """
        Graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """
        return translate_df_to_graph(self.buses_df, self.lines_df)

    @property
    def station(self):
        """
        DataFrame with form of buses_df with only grid's station's secondary
        side bus information.

        """
        return (
            self.buses_df.loc[self.transformers_df.iloc[0].bus1].to_frame().T
        )

    @property
    def generators_df(self):
        """
        Connected generators within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all generators in topology. For more information on
            the dataframe see
            :attr:`~.network.topology.Topology.generators_df`.

        """
        return self.edisgo_obj.topology.generators_df[
            self.edisgo_obj.topology.generators_df.bus.isin(
                self.buses_df.index
            )
        ]

    @property
    def generators(self):
        """
        Connected generators within the network.

        Returns
        -------
        list(:class:`~.network.components.Generator`)
            List of generators within the network.

        """
        for gen in self.generators_df.index:
            yield Generator(id=gen, edisgo_obj=self.edisgo_obj)

    @property
    def loads_df(self):
        """
        Connected loads within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all loads in topology. For more information on the
            dataframe see :attr:`~.network.topology.Topology.loads_df`.

        """
        return self.edisgo_obj.topology.loads_df[
            self.edisgo_obj.topology.loads_df.bus.isin(self.buses_df.index)
        ]

    @property
    def loads(self):
        """
        Connected loads within the network.

        Returns
        -------
        list(:class:`~.network.components.Load`)
            List of loads within the network.

        """
        for l in self.loads_df.index:
            yield Load(id=l, edisgo_obj=self.edisgo_obj)

    @property
    def storage_units_df(self):
        """
        Connected storage units within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all storage units in topology. For more information
            on the dataframe see
            :attr:`~.network.topology.Topology.storage_units_df`.

        """
        return self.edisgo_obj.topology.storage_units_df[
            self.edisgo_obj.topology.storage_units_df.bus.isin(
                self.buses_df.index
            )
        ]

    @property
    def charging_points_df(self):
        """
        Connected charging points within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all charging points in topology. For more
            information on the dataframe see
            :attr:`~.network.topology.Topology.charging_points_df`.

        """
        return self.edisgo_obj.topology.charging_points_df[
            self.edisgo_obj.topology.charging_points_df.bus.isin(
                self.buses_df.index
            )
        ]

    @property
    def switch_disconnectors_df(self):
        """
        Switch disconnectors in network.

        Switch disconnectors are points where rings are split under normal
        operating conditions.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all switch disconnectors in network. For more
            information on the dataframe see
            :attr:`~.network.topology.Topology.switches_df`.

        """
        return self.edisgo_obj.topology.switches_df[
            self.edisgo_obj.topology.switches_df.bus_closed.isin(
                self.buses_df.index
            )
        ][
            self.edisgo_obj.topology.switches_df.type_info
            == "Switch Disconnector"
        ]

    @property
    def switch_disconnectors(self):
        """
        Switch disconnectors within the network.

        Returns
        -------
        list(:class:`~.network.components.Switch`)
            List of switch disconnectory within the network.

        """
        for s in self.switch_disconnectors_df.index:
            yield Switch(id=s, edisgo_obj=self.edisgo_obj)

    @property
    def lines_df(self):
        """
        Lines within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in topology. For more information on the
            dataframe see :attr:`~.network.topology.Topology.lines_df`.

        """
        return self.edisgo_obj.topology.lines_df[
            self.edisgo_obj.topology.lines_df.bus0.isin(self.buses_df.index)
            & self.edisgo_obj.topology.lines_df.bus1.isin(self.buses_df.index)
        ]

    @property
    @abstractmethod
    def buses_df(self):
        """
        Buses within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in topology. For more information on the
            dataframe see :attr:`~.network.topology.Topology.buses_df`.

        """

    @property
    def weather_cells(self):
        """
        Weather cells in network.

        Returns
        -------
        list(int)
            List of weather cell IDs in network.

        """
        return self.generators_df.weather_cell_id.dropna().unique()

    @property
    def peak_generation_capacity(self):
        """
        Cumulative peak generation capacity of generators in the network in MW.

        Returns
        -------
        float
            Cumulative peak generation capacity of generators in the network
            in MW.

        """
        return self.generators_df.p_nom.sum()

    @property
    def peak_generation_capacity_per_technology(self):
        """
        Cumulative peak generation capacity of generators in the network per
        technology type in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Cumulative peak generation capacity of generators in the network
            per technology type in MW.

        """
        return self.generators_df.groupby(["type"]).sum()["p_nom"]

    @property
    def peak_load(self):
        """
        Cumulative peak load of loads in the network in MW.

        Returns
        -------
        float
            Cumulative peak load of loads in the network in MW.

        """
        return self.loads_df.peak_load.sum()

    @property
    def peak_load_per_sector(self):
        """
        Cumulative peak load of loads in the network per sector in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Cumulative peak load of loads in the network per sector in MW.

        """
        return self.loads_df.groupby(["sector"]).sum()["peak_load"]

    def __repr__(self):
        return "_".join([self.__class__.__name__, str(self.id)])

    def convert_to_pu_system(self, s_base=1, t_base=1, convert_timeseries=True,
                             timeseries_inplace=False):
        """
        Method to convert grid to pu-system. Can be used to run optimisation with
        it.

        :param self:
        :param s_base: default 1MW
        :param t_base: default 1h
        :param convert_timeseries: boolean, determines whether timeseries data
            should also be converted
        :param timeseries_inplace: boolean, determines whether timeseries data is
            changed directly inside the edisgo-object. Otherwise timeseries are
            returned as DataFrame. Note: Be careful with this option and only use
            when the whole object is converted or you only need one grid.
        :return:
        """
        v_base = self.nominal_voltage
        z_base = np.square(v_base) / s_base
        self.base_power = s_base
        self.base_impedance = z_base
        self.base_time = t_base
        # convert all components and add pu_columns
        pu_cols = {'lines': ['r_pu', 'x_pu', 's_nom_pu'],
                   'generators': ['p_nom_pu'],
                   'loads': ['peak_load_pu'],
                   'storage_units': ['p_nom_pu', 'capacity_pu']}
        for comp, cols in pu_cols.items():
            new_cols = [col for col in cols if col not in
                        getattr(self.edisgo_obj.topology,
                                comp + '_df').columns]
            for col in new_cols:
                getattr(self.edisgo_obj.topology, comp + '_df')[
                    col] = np.NaN
        self.edisgo_obj.topology.lines_df.loc[
            self.lines_df.index, 'r_pu'] = \
            self.lines_df.r / self.base_impedance
        self.edisgo_obj.topology.lines_df.loc[
            self.lines_df.index, 'x_pu'] = \
            self.lines_df.x / self.base_impedance
        self.edisgo_obj.topology.lines_df.loc[
            self.lines_df.index, 's_nom_pu'] = \
            self.lines_df.s_nom / self.base_power
        # make sure that transformer is not hvmv transformer
        if self.transformers_df.index.values in \
                self.edisgo_obj.topology.transformers_df.index.values:
            self.edisgo_obj.topology.transformers_df.loc[
                self.transformers_df.index, 's_nom_pu'] = \
                self.transformers_df.s_nom / self.base_power
        self.edisgo_obj.topology.generators_df.loc[
            self.generators_df.index,
            'p_nom_pu'] = \
            self.generators_df.p_nom / self.base_power
        self.edisgo_obj.topology.loads_df.loc[
            self.loads_df.index, 'peak_load_pu'] = \
            self.loads_df.peak_load / self.base_power
        self.edisgo_obj.topology.storage_units_df.loc[
            self.storage_units_df.index,
            'p_nom_pu'] = \
            self.storage_units_df.p_nom / self.base_power
        if not self.edisgo_obj.topology.storage_units_df.empty:
            self.edisgo_obj.topology.storage_units_df.loc[
                self.storage_units_df.index,
                'capacity_pu'] = \
                self.storage_units_df.capacity / (
                        self.base_power * self.base_time)
        if hasattr(self, 'charging_points_df') and \
                not self.charging_points_df.empty:
            if not 'p_nom_pu' in self.charging_points_df.columns:
                self.edisgo_obj.topology.charging_points_df[
                    'p_nom_pu'] = np.NaN
            self.edisgo_obj.topology.charging_points_df.loc[
                self.charging_points_df.index, 'p_nom_pu'] = \
                self.charging_points_df.p_nom / self.base_power
        # convert timeseries
        if convert_timeseries:
            if not hasattr(self.edisgo_obj.timeseries,
                           'generators_active_power'):
                print(
                    'No data inside the timeseries object. Please provide '
                    'timeseries to convert to the pu-system. Process is '
                    'interrupted.')
                return
            timeseries = {}
            # pass if values would not change
            if self.base_power == 1:
                if timeseries_inplace:
                    return
            for component in ['generators', 'loads',
                              'storage_units', 'charging_points']:
                if hasattr(self.edisgo_obj.timeseries,
                           component + '_active_power'):
                    active_power = getattr(self.edisgo_obj.timeseries,
                                           component + '_active_power')
                    reactive_power = getattr(self.edisgo_obj.timeseries,
                                             component + '_reactive_power')
                    comp_names = getattr(self, component + '_df').index
                    active_power_pu = active_power[
                                          comp_names] / self.base_power
                    reactive_power_pu = reactive_power[comp_names]
                    if timeseries_inplace:
                        active_power[comp_names] = active_power_pu
                        reactive_power[comp_names] = reactive_power_pu
                    else:
                        timeseries[
                            component + '_active_power_pu'] = active_power_pu
                        timeseries[
                            component + '_reactive_power_pu'] = reactive_power_pu
            return timeseries


class MVGrid(Grid):
    """
    Defines a medium voltage network in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._lv_grids = kwargs.get("lv_grids", [])

    @property
    def lv_grids(self):
        """
        Underlying LV grids.

        Parameters
        ----------
        lv_grids : list(:class:`~.network.grids.LVGrid`)

        Returns
        -------
        list generator
            Generator object of underlying LV grids of type
            :class:`~.network.grids.LVGrid`.

        """
        for lv_grid in self._lv_grids:
            yield lv_grid

    @lv_grids.setter
    def lv_grids(self, lv_grids):
        self._lv_grids = lv_grids

    @property
    def buses_df(self):
        """
        Buses within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in topology. For more information on the
            dataframe see :attr:`~.network.topology.Topology.buses_df`.

        """
        return self.edisgo_obj.topology.buses_df.drop(
            self.edisgo_obj.topology.buses_df.lv_grid_id.dropna().index
        )

    @property
    def transformers_df(self):
        """
        Transformers to overlaying network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all transformers to overlaying network. For more
            information on the dataframe see
            :attr:`~.network.topology.Topology.transformers_df`.

        """
        return self.edisgo_obj.topology.transformers_hvmv_df

    def draw(self):
        """
        Draw MV network.

        """
        # ToDo call EDisGoReimport.plot_mv_grid_topology
        raise NotImplementedError


class LVGrid(Grid):
    """
    Defines a low voltage network in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def buses_df(self):
        """
        Buses within the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in topology. For more information on the
            dataframe see :attr:`~.network.topology.Topology.buses_df`.

        """
        return self.edisgo_obj.topology.buses_df.loc[
            self.edisgo_obj.topology.buses_df.lv_grid_id == self.id
        ]

    @property
    def transformers_df(self):
        """
        Transformers to overlaying network.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all transformers to overlaying network. For more
            information on the dataframe see
            :attr:`~.network.topology.Topology.transformers_df`.

        """
        return self.edisgo_obj.topology.transformers_df[
            self.edisgo_obj.topology.transformers_df.bus1.isin(
                self.buses_df.index
            )
        ]

    def draw(self,
             node_color="black", edge_color="black",
             colorbar=False, labels=False, filename=None):
        """
        Draw LV network.

        Currently, edge width is proportional to nominal apparent power of
        the line and node size is proportional to peak load of connected loads.

        Parameters
        -----------
        node_color : str or :pandas:`pandas.Series<Series>`
            Color of the nodes (buses) of the grid. If provided as string
            all nodes will have that color. If provided as series, the
            index of the series must contain all buses in the LV grid and the
            corresponding values must be float values, that will be translated
            to the node color using a colormap, currently set to "Blues".
            Default: "black".
        edge_color : str or :pandas:`pandas.Series<Series>`
            Color of the edges (lines) of the grid. If provided as string
            all edges will have that color. If provided as series, the
            index of the series must contain all lines in the LV grid and the
            corresponding values must be float values, that will be translated
            to the edge color using a colormap, currently set to "inferno_r".
            Default: "black".
        colorbar : bool
            If True, a colorbar is added to the plot for node and edge colors,
            in case these are sequences. Default: False.
        labels : bool
            If True, displays bus names. As bus names are quite long, this
            is currently not very pretty. Default: False.
        filename : str or None
            If a filename is provided, the plot is saved under that name but
            not displayed. If no filename is provided, the plot is only
            displayed. Default: None.

        """
        G = self.graph
        pos = graphviz_layout(G, prog="dot")

        # assign edge width + color and node size + color
        top = self.edisgo_obj.topology
        edge_width = [
            top.get_line_connecting_buses(u, v).s_nom.sum() * 10 for
            u, v in G.edges()]
        if isinstance(edge_color, pd.Series):
            edge_color = [
                edge_color.loc[
                    top.get_line_connecting_buses(u, v).index[0]]
                for u, v in G.edges()]
            edge_color_is_sequence = True
        else:
            edge_color_is_sequence = False

        node_size = [top.get_connected_components_from_bus(v)[
                        "loads"].peak_load.sum() * 50000 + 10 for v in G]
        if isinstance(node_color, pd.Series):
            node_color = [node_color.loc[v] for v in G]
            node_color_is_sequence = True
        else:
            node_color_is_sequence = False

        # draw edges and nodes of the graph
        fig, ax = plt.subplots(figsize=(12, 12))
        cm_edges = nx.draw_networkx_edges(
            G, pos,
            width=edge_width,
            edge_color=edge_color,
            edge_cmap=plt.cm.get_cmap("inferno_r")
        )
        cm_nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_color,
            cmap="Blues"
        )
        if colorbar:
            if edge_color_is_sequence:
                fig.colorbar(cm_edges, ax=ax)
            if node_color_is_sequence:
                fig.colorbar(cm_nodes, ax=ax)
        if labels:
            # ToDo find nicer way to display bus names
            label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
            nx.draw_networkx_labels(
                G, pos, font_size=8, bbox=label_options,
                horizontalalignment="right")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename,
                        dpi=150, bbox_inches='tight', pad_inches=0.1
                        )
            plt.close()
