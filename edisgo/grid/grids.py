from edisgo.grid.components import Generator, Load, Storage


class Grid:
    """
    Defines a basic grid in eDisGo

    Parameters
    -----------
    _id : str or int
        Identifier
    _network : :class:`~.grid.network.Network`
        Network container.

    # ToDo add annual_consumption property?

    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._network = kwargs.get('network', None)

        self._nominal_voltage = None

        # # ToDo Implement if necessary
        # self._transformers = None
        # self._station = None
        # ToDo maybe add lines_df and lines property if needed

    @property
    def id(self):
        return self._id

    @property
    def network(self):
        return self._network

    @property
    def nominal_voltage(self):
        """
        Nominal voltage of grid in V.

        Parameters
        ----------
        nominal_voltage : float

        Returns
        -------
        float
            Nominal voltage of grid in V.

        """
        if self._nominal_voltage is None:
            self.nominal_voltage = self.buses_df.v_nom.max()
        return self._nominal_voltage

    @nominal_voltage.setter
    def nominal_voltage(self, nominal_voltage):
        self._nominal_voltage = nominal_voltage

    @property
    def generators(self):
        """
        Connected generators within the grid.

        Returns
        -------
        list(:class:`~.grid.components.Generator`)
            List of generators within the grid.

        """
        for gen in self.generators_df.index:
            yield Generator(id=gen, network=self.network)

    @property
    def loads_df(self):
        """
        Connected loads within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all loads in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.loads_df`.

        """
        return self.network.loads_df[
            self.network.loads_df.bus.isin(self.buses_df.index)]

    @property
    def loads(self):
        """
        Connected loads within the grid.

        Returns
        -------
        list(:class:`~.grid.components.Load`)
            List of loads within the grid.

        """
        for l in self.loads_df.index:
            yield Load(id=l, network=self.network)

    @property
    def weather_cells(self):
        """
        Weather cells in grid.

        Returns
        -------
        list(int)
            List of weather cell IDs in grid.

        """
        return self.generators_df.weather_cell_id.dropna().unique()

    @property
    def peak_generation_capacity(self):
        """
        Cumulative peak generation capacity of generators in the grid in MW.

        Returns
        -------
        float
            Cumulative peak generation capacity of generators in the grid
            in MW.

        """
        return self.generators_df.p_nom.sum()

    @property
    def peak_generation_capacity_per_technology(self):
        """
        Cumulative peak generation capacity of generators in the grid per
        technology type in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Cumulative peak generation capacity of generators in the grid per
            technology type in MW.

        """
        return self.generators_df.groupby(['type']).sum()['p_nom']

    @property
    def peak_load(self):
        """
        Cumulative peak load of loads in the grid in MW.

        Returns
        -------
        float
            Cumulative peak load of loads in the grid in MW.

        """
        return self.loads_df.peak_load.sum()

    @property
    def peak_load_per_sector(self):
        """
        Cumulative peak load of loads in the grid per sector in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Cumulative peak load of loads in the grid per sector in MW.

        """
        return self.loads_df.groupby(['sector']).sum()['peak_load']

    def __repr__(self):
        return '_'.join([self.__class__.__name__, str(self._id)])

    def connect_generators(self, generators):
        """
        Connects generators to grid.

        Parameters
        ----------
        generators : :pandas:`pandas.DataFrame<dataframe>`
            Generators to be connected.

        """
        # ToDo: Should we implement this or move function from tools here?
        raise NotImplementedError


class MVGrid(Grid):
    """
    Defines a medium voltage grid in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._disconnecting_points = kwargs.get('disconnecting_points', None)
        self._lv_grids = kwargs.get('lv_grids', [])

    @property
    def lv_grids(self):
        """
        Underlying LV grids.

        Parameters
        ----------
        lv_grids : list(:class:`~.grid.grids.LVGrid`)

        Returns
        -------
        list generator
            Generator object of underlying LV grids of type
            :class:`~.grid.grids.LVGrid`.

        """
        for lv_grid in self._lv_grids:
            yield lv_grid

    @lv_grids.setter
    def lv_grids(self, lv_grids):
        self._lv_grids = lv_grids

    @property
    def generators_df(self):
        """
        Connected generators within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all generators in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.generators_df`.

        """
        return self.network.generators_df[
            self.network.generators_df.bus.isin(self.buses_df.index)].drop(
            labels=['Generator_slack'])

    @property
    def disconnecting_points(self):
        """
        Medium voltage disconnecting points.

        Medium voltage disconnecting points are points where MV rings are split
        under normal operation conditions.

        Parameters
        ----------
        disconnecting_points : list(:class:`~.grid.components.MVDisconnectingPoint`)

        Returns
        -------
        list(:class:`~.grid.components.MVDisconnectingPoint`)
            List of medium voltage disconnecting points.

        """
        return self._disconnecting_points

    @disconnecting_points.setter
    def disconnecting_points(self, disconnecting_points):
        self._disconnecting_points = disconnecting_points

    @property
    def buses_df(self):
        """
        Buses within the grid.

         Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.buses_df`.

        """
        return self.network.buses_df.drop(
            self.network.buses_df.lv_grid_id.dropna().index)

    def draw(self):
        """
        Draw MV grid.

        """
        # ToDo call EDisGoReimport.plot_mv_grid_topology
        raise NotImplementedError


class LVGrid(Grid):
    """
    Defines a low voltage grid in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def buses_df(self):
        """
        Buses within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.buses_df`.

        """
        return self.network.buses_df.loc[
            self.network.buses_df.lv_grid_id == self.id]

    @property
    def generators_df(self):
        """
        Connected generators within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all generators in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.generators_df`.

        """
        return self.network.generators_df[
            self.network.generators_df.bus.isin(self.buses_df.index)]

    def draw(self):
        """
        Draw LV grid.

        """
        # ToDo: implement networkx graph plot
        raise NotImplementedError
