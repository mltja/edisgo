import pandas as pd
import numpy as np
import logging

from edisgo.grid.tools import get_gen_info, \
    get_capacities_by_type, \
    get_capacities_by_type_and_weather_cell


def curtail_voltage(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_voltage'.

    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators. The voltage
    at the nodes are used as an input provide a feedin_factor that changes
    the curtailment by modifiying the multiplied feedin at the points where
    there are very high voltages.

    The lower voltage threshold is the node voltage below which no
    curtailment is assigned to the respective generator connected
    to the node. This assignment can be done by using the keyword
    argument 'voltage_threshold_lower'. By default, this voltage
    is set to 1.0 per unit. Lowering this voltage will increase
    the amount of curtailment to generators with higher node
    voltages.

    This method runs an edisgo_object.analyze internally to find out
    the voltage at the nodes if an ediso_object.analyze has not already
    performed and the results saved in edisgo_object.network.results.v_res()

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        : :class:`edisgo.network.CurtailmentControl` class. The feedin dataframe
        has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        is a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the :class:`edisgo.network.CurtailmentControl`
        class and available through the :meth:`edisgo.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.network.CurtailmentControl` for more details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the edisgo_objects'
        network. This is input through the edisgo_object. See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object in which this function was called through the respective
        :class:`edisgo.network.CurtailmentControl` instance.
    voltage_threshold_lower: :float
        the node voltage below which no curtailment would be assigned to the respective
        generator.
    """
    voltage_threshold_lower = kwargs.get('voltage_threshold_lower', 1.0)

    # get the results of a load flow
    # get the voltages at the nodes

    feedin_gens = feedin.columns.levels[0].values.copy()
    feedin_gen_reprs = feedin.columns.levels[1].values.copy()

    try:
        v_pu = edisgo_object.network.results.v_res()

    except AttributeError:
        # if the load flow hasn't been done,
        # do it!
        edisgo_object.analyze()
        # ToDo: Figure out the problem with putting feedin_gens inside of v_res()
        v_pu = edisgo_object.network.results.v_res()

    if not(v_pu.empty):
        # get only the specific feedin objects
        v_pu = v_pu.loc[:, (slice(None), feedin_gen_reprs)]

        # curtailment calculation by inducing a reduced or increased feedin
        # find out the difference from lower threshold
        feedin_factor = v_pu - voltage_threshold_lower + 1
        feedin_factor.columns = feedin_factor.columns.droplevel(0)  # drop the 'mv' 'lv' labels

        # multiply feedin_factor to feedin to modify the amount of curtailment
        modified_feedin = feedin_factor.multiply(feedin, level=1)

        # total_curtailment
        curtailment = modified_feedin.divide(modified_feedin.sum(axis=1), axis=0). \
            multiply(total_curtailment_ts, axis=0)

        # assign curtailment to individual generators
        assign_curtailment(curtailment, edisgo_object)
    else:
        message = "There is no resulting node voltages after the PFA calculation" +\
            " which correspond to the generators in the columns of the given feedin data"
        logging.warning(message)


def curtail_loading(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_loading'.
    This method has not been implemented yet.
    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        : :class:`edisgo.network.CurtailmentControl` class. The feedin dataframe
        has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        is a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the :class:`edisgo.network.CurtailmentControl`
        class and available through the :meth:`edisgo.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.network.CurtailmentControl` for more details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the edisgo_objects'
        network. This is input through the edisgo_object. See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object in which this function was called through the respective
        :class:`edisgo.network.CurtailmentControl` instance.
    """

    raise NotImplementedError

    return None


def curtail_droop(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_loading'.
    This method has not been implemented yet.
    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators and given a specific
    droop characteristic

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        : :class:`edisgo.network.CurtailmentControl` class. The feedin dataframe
        has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        is a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the :class:`edisgo.network.CurtailmentControl`
        class and available through the :meth:`edisgo.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.network.CurtailmentControl` for more details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the edisgo_objects'
        network. This is input through the edisgo_object. See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object in which this function was called through the respective
        :class:`edisgo.network.CurtailmentControl` instance.
    """

    raise NotImplementedError

    return None


def curtail_selected(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_selected'.
    This method has not been implemented yet.
    The curtailment that has to be met in each step is allocated
    to specific selected generators.

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        : :class:`edisgo.network.CurtailmentControl` class. The feedin dataframe
        has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        is a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the :class:`edisgo.network.CurtailmentControl`
        class and available through the :meth:`edisgo.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.network.CurtailmentControl` for more details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the edisgo_objects'
        network. This is input through the edisgo_object. See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object in which this function was called through the respective
        :class:`edisgo.network.CurtailmentControl` instance.
    """

    raise NotImplementedError

    return None


def curtail_all(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_all'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step. This is a simple curtailment method where
    the feedin is summed up and normalized, multiplied with `total_curtailment_ts`
    and assigned to each generator directly based on the columns in
    `total_curtailment_ts`.

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        : :class:`edisgo.network.CurtailmentControl` class. The feedin dataframe
        has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        is a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the :class:`edisgo.network.CurtailmentControl`
        class and available through the :meth:`edisgo.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.network.CurtailmentControl` for more details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the edisgo_objects'
        network. This is input through the edisgo_object. See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object in which this function was called through the respective
        :class:`edisgo.network.CurtailmentControl` instance.
    """
    # create a feedin factor of 1
    # make sure the nans are filled an
    feedin_factor = total_curtailment_ts.copy()
    feedin_factor = feedin_factor / feedin_factor
    feedin_factor.fillna(1.0, inplace=True)

    feedin.mul(feedin_factor, axis=0, level=1)

    # total_curtailment
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
        multiply(total_curtailment_ts, axis=0)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo_object)


def assign_curtailment(curtailment, edisgo_object):
    """
    Implements curtailment helper function to assign the curtailment time series
    to each and every individual generator and ensure that they get processed
    and included in the edisgo_object.timeseries.curtailment correctly

    Parameters
    ----------
    curtailment : : pandas:`pandas.DataFrame<dataframe>`
        final curtailment dataframe with generator objects as column
        labels and a DatetimeIndex as the index
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object created
    """
    # pre-process curtailment before assigning it to generatos
    # Drop columns where there were 0/0 divisions due to feedin being 0
    curtailment.dropna(axis=1, how='all', inplace=True)
    # fill the remaining nans if there are any with 0s
    curtailment.fillna(0, inplace=True)

    # drop extra column levels that were present in feedin
    for r in range(len(curtailment.columns.levels) - 1):
        curtailment.columns = curtailment.columns.droplevel(1)

    # assign curtailment to individual generators
    for gen in curtailment.columns:
        gen.curtailment = curtailment.loc[:, gen]

    if not edisgo_object.network.timeseries._curtailment:
        edisgo_object.network.timeseries._curtailment = list(curtailment.columns)
    else:
        edisgo_object.network.timeseries._curtailment.extend(list(curtailment.columns))
