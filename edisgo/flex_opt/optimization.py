import numpy as np
import pandas as pd
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
from edisgo.tools.tools import get_nodal_residual_load


def setup_model(edisgo, downstream_node_matrix, timesteps=None, optimize_storage=True,
                optimize_ev_charging=True, objective='curtailment', **kwargs):
    """
    Method to set up pyomo model for optimisation of storage procurement
    and/or ev charging with linear approximation of power flow from
    eDisGo-object.

    :param edisgo:
    :param downstream_node_matrix:
    :param timesteps:
    :param optimize_storage:
    :param optimize_ev_charging:
    :param objective: choose the objective that should be minimized, so far
            'curtailment' and 'peak_load' are implemented
    :param kwargs:
    :return:
    """
    model = pm.ConcreteModel()
    # check if correct value of objective is inserted
    if objective not in ['curtailment', 'peak_load']:
        raise ValueError('The objective you inserted is not implemented yet.')

    # Todo: Extract kwargs values from cfg?

    # DEFINE SETS AND FIX PARAMETERS
    print('Setup model: Defining sets and parameters.')
    model.bus_set = pm.Set(initialize=edisgo.topology.buses_df.index)
    model.slack_bus = pm.Set(initialize=edisgo.topology.slack_df.bus)
    if timesteps is not None:
        model.timeindex = timesteps
    else:
        model.timeindex = edisgo.timeseries.timeindex
    model.time_set = pm.RangeSet(0, len(timesteps)-1)
    print('First timestep: {}, last timestep: {}.'.format(model.time_set[1], model.time_set[-1]))
    model.time_zero = [model.time_set[1]]
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set[1], model.time_set[-1]])
    model.line_set = pm.Set(initialize=edisgo.topology.lines_df.index)
    if optimize_storage:
        model.storage_set = pm.Set(initialize=edisgo.topology.storage_units_df.index)
        optimized_storage_units = \
            kwargs.get('flexible_storage_units',
                       edisgo.topology.storage_units_df.index)
        model.optimized_storage_set = \
            pm.Set(initialize=optimized_storage_units)
        model.fixed_storage_set = model.storage_set - \
                                        model.optimized_storage_set
        model.fix_relative_soc = kwargs.get('fix_relative_soc', 0.5)
        inflexible_storage_units = list(model.fixed_storage_set.data())
    else:
        inflexible_storage_units = None
    if optimize_ev_charging:
        model.energy_band_charging_points = kwargs.get('energy_band_charging_points')
        model.mapping_cp = kwargs.get('mapping_cp')
        model.charging_points_set = \
            pm.Set(initialize=edisgo.topology.charging_points_df.index)
        model.flexible_charging_points_set = \
            pm.Set(initialize=model.mapping_cp.index)
        model.inflexible_charging_points_set = \
            model.charging_points_set - model.flexible_charging_points_set
        model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
        inflexible_charging_points = list(model.inflexible_charging_points_set.data())
    else:
        inflexible_charging_points = None
    model.residual_load = \
        get_residual_load_of_not_optimized_components(edisgo, model)
    model.grid = edisgo.topology
    model.downstream_nodes_matrix = downstream_node_matrix
    nodal_active_power, nodal_reactive_power = get_nodal_residual_load(
        edisgo, considered_storage=inflexible_storage_units,
        considered_charging_points=inflexible_charging_points)
    model.nodal_active_power = nodal_active_power.T
    model.nodal_reactive_power = nodal_reactive_power.T
    model.v_min = kwargs.get("v_min", 0.9)
    model.v_max = kwargs.get("v_max", 1.1)
    model.power_factor = kwargs.get("pf", 0.9)
    model.v_nom = edisgo.topology.buses_df.v_nom.iloc[0]

    if objective == 'peak_load':
        model.delta_min = kwargs.get('delta_min', 0.9)
        model.delta_max = kwargs.get('delta_max', 0.1)
        model.min_load_factor = pm.Var()
        model.max_load_factor = pm.Var()

    # DEFINE VARIABLES
    print('Setup model: Defining variables.')
    model.p_cum = pm.Var(model.line_set, model.time_set,
                     bounds=lambda m, l, t:
                     (-m.power_factor * m.grid.lines_df.loc[l, 's_nom'],
                      m.power_factor * m.grid.lines_df.loc[l, 's_nom']))
    model.q_cum = pm.Var(model.line_set, model.time_set) # Todo: remove? Not necessary at current configuration
    model.v = pm.Var(model.bus_set, model.time_set,
                 bounds=(
                 np.square(model.v_min * model.v_nom), np.square(model.v_max *
                                                                 model.v_nom)))
    model.curtailment_load = pm.Var(model.bus_set, model.time_set,
                                    bounds=(0, None))
    model.curtailment_feedin = pm.Var(model.bus_set, model.time_set,
                                      bounds=(0, None))
    model.curtailment_reactive_load = pm.Var(model.bus_set, model.time_set,
                                             bounds=(0, None))
    model.curtailment_reactive_feedin = pm.Var(model.bus_set, model.time_set,
                                               bounds=(0, None))
    if optimize_storage:
        model.soc = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   0, m.grid.storage_units_df.loc[b, 'capacity']))
        model.charging = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   -m.grid.storage_units_df.loc[b, 'p_nom'],
                   m.grid.storage_units_df.loc[b, 'p_nom']))
    if optimize_ev_charging:
        model.charging_ev = \
            pm.Var(model.flexible_charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (0, m.energy_band_charging_points.loc[
                       m.timeindex[t], '_'.join(['power', str(m.mapping_cp.loc[b, 'ags']),
                                    str(m.mapping_cp.loc[b, 'cp_idx'])])]))
        model.energy_level_ev = \
            pm.Var(model.flexible_charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (m.energy_band_charging_points.loc[
                        m.timeindex[t], '_'.join(['lower', str(m.mapping_cp.loc[b, 'ags']),
                                     str(m.mapping_cp.loc[b, 'cp_idx'])])],
                    m.energy_band_charging_points.loc[
                        m.timeindex[t], '_'.join(['upper', str(m.mapping_cp.loc[b, 'ags']),
                                     str(m.mapping_cp.loc[b, 'cp_idx'])])]))

    # DEFINE CONSTRAINTS
    print('Setup model: Setting constraints.')
    model.ActivePower = pm.Constraint(model.line_set, model.time_set,
                                      rule=active_power)
    model.ReactivePower = pm.Constraint(model.line_set, model.time_set,
                                    rule=reactive_power)
    model.SlackVoltage = pm.Constraint(model.slack_bus, model.time_set,
                                       rule=slack_voltage)
    model.VoltageDrop = pm.Constraint(model.line_set, model.time_set,
                                      rule=voltage_drop)
    if optimize_storage:
        model.BatteryCharging = pm.Constraint(model.storage_set,
                                              model.time_non_zero, rule=soc)
        model.FixedSOC = pm.Constraint(model.storage_set,
                                       model.times_fixed_soc, rule=fix_soc)
    if optimize_ev_charging:
        model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                         model.time_non_zero, rule=charging_ev)
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                          rule=initial_energy_level)

    # DEFINE OBJECTIVE
    print('Setup model: Setting objective.')
    if objective == 'peak_load':
        model.LoadFactorMin = pm.Constraint(model.time_set, rule=load_factor_min)
        model.LoadFactorMax = pm.Constraint(model.time_set, rule=load_factor_max)
        model.objective = pm.Objective(rule=minimize_max_residual_load,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'curtailment':
        model.objective = pm.Objective(rule=minimize_curtailment,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    else:
        raise Exception('Unknown objective.')

    if kwargs.get('print_model', False):
        model.pprint()
    print('Successfully set up optimisation model.')
    return model


def update_model(model, timeindex, energy_band_charging_points, **kwargs):
    """
    Method to update model parameter where necessary if rolling horizon
    optimization is chosen.
    :param model:
    :param timeindex:
    :return:
    """
    model.del_component(model.timeindex)
    model.timeindex = timeindex
    model.energy_band_charging_points = energy_band_charging_points
    if kwargs.get('print_model', False):
        model.pprint()
    return model


def optimize(model, solver, save_dir=None, load_solutions=True):
    """
    Method to run the optimization and extract the results.

    :param model: pyomo.environ.ConcreteModel
    :param solver: str
                    Solver type, e.g. 'glpk', 'gurobi', 'ipopt'
    :param save_dir: str
                    directory to which results are saved, default None will
                    no saving of the results
    :return:
    """
    opt = pm.SolverFactory(solver)

    # Optimize
    results = opt.solve(model, tee=True, load_solutions=load_solutions)

    # Extract results
    x_charge = pd.DataFrame()
    soc = pd.DataFrame()
    x_charge_ev = pd.DataFrame()
    energy_level_cp = pd.DataFrame()
    curtailment_load = pd.DataFrame()
    curtailment_feedin = pd.DataFrame()
    curtailment_reactive_load = pd.DataFrame()
    curtailment_reactive_feedin = pd.DataFrame()
    p_line = pd.DataFrame()
    q_line = pd.DataFrame()
    v_bus = pd.DataFrame()

    if (results.solver.status == SolverStatus.ok) and \
            (
                    results.solver.termination_condition == TerminationCondition.optimal):
        print('Model Solved to Optimality')
        for time in model.time_set:
            timeindex = model.timeindex[time]
            if hasattr(model, 'storage_set'):
                for bus in model.storage_set:
                    x_charge.loc[timeindex, bus] = model.charging[bus, time].value
                    soc.loc[timeindex, bus] = model.soc[bus, time].value
            if hasattr(model, 'charging_points_set'):
                for cp in model.charging_points_set:
                    x_charge_ev.loc[timeindex, cp] = model.charging_ev[cp, time].value
                    energy_level_cp.loc[timeindex, cp] = \
                        model.energy_level_ev[cp, time].value
            for bus in model.bus_set:
                curtailment_feedin.loc[timeindex, bus] = \
                    model.curtailment_feedin[bus, time].value
                curtailment_load.loc[timeindex, bus] = \
                    model.curtailment_load[bus, time].value
                curtailment_reactive_feedin.loc[timeindex, bus] = \
                    model.curtailment_reactive_feedin[bus, time].value
                curtailment_reactive_load.loc[timeindex, bus] = \
                    model.curtailment_reactive_load[bus, time].value
                v_bus.loc[timeindex, bus] = np.sqrt(model.v[bus, time].value)
            for line in model.line_set:
                p_line.loc[timeindex, line] = model.p_cum[line, time].value
                q_line.loc[timeindex, line] = model.q_cum[line, time].value
        if save_dir:
            x_charge.to_csv(save_dir+'/x_charge_storage.csv')
            soc.to_csv(save_dir+'/soc_storage.csv')
            x_charge_ev.to_csv(save_dir+'/x_charge_ev.csv')
            energy_level_cp.to_csv(save_dir+'/energy_level_ev.csv')
            curtailment_feedin.to_csv(save_dir + '/curtailment_feedin.csv')
            curtailment_load.to_csv(save_dir + '/curtailment_load.csv')
            curtailment_reactive_feedin.to_csv(save_dir + '/curtailment_reactive_feedin.csv')
            curtailment_reactive_load.to_csv(save_dir + '/curtailment_reactive_load.csv')

            v_bus.to_csv(save_dir + '/voltage_buses.csv')
            p_line.to_csv(save_dir + '/active_power_lines.csv')
            q_line.to_csv(save_dir + '/reactive_power_lines.csv')
        return x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
               curtailment_load, curtailment_reactive_feedin, curtailment_reactive_load, \
               v_bus, p_line, q_line
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('Model is infeasible')
        return
        # Do something when model in infeasible
    else:
        print('Solver Status: ', results.solver.status)
        return


def get_residual_load_of_not_optimized_components(edisgo, model):
    """
    Method to get residual load of fixed components.

    :param edisgo:
    :param model:
    :return:
    """
    if hasattr(model, 'fixed_storage_set'):
        relevant_storage_units = model.fixed_storage_set
    else:
        relevant_storage_units = edisgo.topology.storage_units_df.index

    if hasattr(model, 'inflexible_charging_points_set'):
        relevant_charging_points = model.inflexible_charging_points_set
    else:
        relevant_charging_points = edisgo.topology.charging_points_df.index

    if edisgo.timeseries.charging_points_active_power.empty:
        return (
                edisgo.timeseries.generators_active_power.sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power.sum(axis=1)
        ).loc[edisgo.timeseries.timeindex]
    else:
        return (
                edisgo.timeseries.generators_active_power.sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power.sum(axis=1)
                - edisgo.timeseries.charging_points_active_power[
                    relevant_charging_points].sum(axis=1)
        ).loc[edisgo.timeseries.timeindex]


def minimize_max_residual_load(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return -model.delta_min * model.min_load_factor + \
           model.delta_max * model.max_load_factor + \
           sum(model.curtailment_load[bus, time] +
               model.curtailment_feedin[bus, time] +
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)


def minimize_curtailment(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return sum(model.curtailment_load[bus, time] +
               model.curtailment_feedin[bus, time] +
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)


def active_power(model, line, time):
    '''
    Constraint for active power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    bus0 = model.grid.lines_df.loc[line, 'bus0']
    bus1 = model.grid.lines_df.loc[line, 'bus1']
    relevant_buses_bus0 = \
        model.downstream_nodes_matrix.loc[bus0][
            model.downstream_nodes_matrix.loc[bus0] == 1].index.values
    relevant_buses_bus1 = \
        model.downstream_nodes_matrix.loc[bus1][
            model.downstream_nodes_matrix.loc[bus1] == 1].index.values
    relevant_buses = list(set(relevant_buses_bus0).intersection(
        relevant_buses_bus1))
    if hasattr(model, 'storage_set'):
        relevant_storage_units = \
            model.grid.storage_units_df.loc[
                model.grid.storage_units_df.index.isin(
                    model.optimized_storage_set) &
                model.grid.storage_units_df.bus.isin(relevant_buses)].index.values
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = \
            model.grid.charging_points_df.loc[
                model.grid.charging_points_df.index.isin(
                    model.flexible_charging_points_set) &
                model.grid.charging_points_df.bus.isin(relevant_buses)].index.values
    load_flow_on_line = \
        model.nodal_active_power.loc[relevant_buses, timeindex].sum()
    return model.p_cum[line, time] == load_flow_on_line + \
        sum(model.charging[storage, time]
            for storage in relevant_storage_units) - \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + \
        sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in relevant_buses)


def reactive_power(model, line, time):
    '''
    Constraint for reactive power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    bus0 = model.grid.lines_df.loc[line, 'bus0']
    bus1 = model.grid.lines_df.loc[line, 'bus1']
    relevant_buses_bus0 = \
        model.downstream_nodes_matrix.loc[bus0][
            model.downstream_nodes_matrix.loc[bus0] == 1].index.values
    relevant_buses_bus1 = \
        model.downstream_nodes_matrix.loc[bus1][
            model.downstream_nodes_matrix.loc[bus1] == 1].index.values
    relevant_buses = list(set(relevant_buses_bus0).intersection(
        relevant_buses_bus1))
    load_flow_on_line = \
        model.nodal_reactive_power.loc[relevant_buses, timeindex].sum()
    return model.q_cum[line, time] == load_flow_on_line + \
        sum(model.curtailment_reactive_load[bus, time] -
            model.curtailment_reactive_feedin[bus, time] for bus in relevant_buses)


def soc(model, storage, time):
    '''
    Constraint for battery charging
    :param model:
    :param storage:
    :param time:
    :return:
    '''
    return model.soc[storage, time] == model.soc[storage, time - 1] - \
           model.grid.storage_units_df.loc[storage, 'efficiency_store'] * \
           model.charging[storage, time - 1]


def fix_soc(model, bus, time):
    '''
    Constraint with which state of charge at beginning and end of charging
    period is fixed at certain value
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    return model.soc[bus, time] == model.fix_relative_soc * \
           model.grid.storage_units_df.loc[bus, 'capacity']


def charging_ev(model, charging_point, time):
    """
    Constraint for charging of EV that has to ly between the lower and upper
    energy band.

    :param model:
    :param charging_point:
    :param time:
    :return:
    """
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_ev[charging_point, time - 1] + \
           model.charging_efficiency * \
           model.charging_ev[charging_point, time - 1]


def initial_energy_level(model, charging_point, time):
    '''
    Constraint for initial value pf energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    # Todo: Change to determined level from previous period
    return model.energy_level_ev[charging_point, time] == 0


def load_factor_min(model, time):
    '''
    Constraint that describes the load factor.
    :param model:
    :param time:
    :return:
    '''
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    timeindex = model.timeindex[time]
    return model.min_load_factor <= model.residual_load.loc[timeindex] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units) - \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points)  + \
        sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in model.bus_set)


def load_factor_max(model, time):
    '''
    Constraint that describes the load factor.
    :param model:
    :param time:
    :return:
    '''
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    timeindex = model.timeindex[time]
    return model.max_load_factor >= model.residual_load.loc[timeindex] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units)- \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + \
        sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in model.bus_set)


def slack_voltage(model, bus, time):
    """
    Constraint that fixes voltage to nominal voltage
    :param model:
    :param bus:
    :param time:
    :return:
    """
    return model.v[bus, time] == np.square(model.v_nom)


def voltage_drop(model, line, time):
    """
    Constraint that describes the voltage drop over one line
    :param model:
    :param line:
    :param time:
    :return:
    """
    bus0 = model.grid.lines_df.loc[line, 'bus0']
    bus1 = model.grid.lines_df.loc[line, 'bus1']
    if model.downstream_nodes_matrix.loc[bus0, bus1] == 1:
        upstream_bus = bus0
        downstream_bus = bus1
    elif model.downstream_nodes_matrix.loc[bus1, bus0] == 1:
        upstream_bus = bus1
        downstream_bus = bus0
    else:
        raise Exception('Something went wrong. Bus0 and bus1 of line {} are '
                        'not connected in downstream_nodes_matrix.'.format(line))
    return model.v[downstream_bus, time] == model.v[upstream_bus, time] + \
        2 * (model.p_cum[line, time]*model.grid.lines_df.loc[line, 'r'] +
             model.q_cum[line, time]*model.grid.lines_df.loc[line, 'x'])


def check_mapping(mapping_cp, edisgo, energy_bands):
    """
    Method to make sure the mapping is valid. Checks the existence of the
    indices in the edisgo object and the existence of entries in the e
    nergy_bands dataframe.

    :param mapping_cp:
    :param edisgo:
    :param energy_bands:
    :return:
    """
    non_existing_cp = mapping_cp.index[~mapping_cp.index.isin(
        edisgo.topology.charging_points_df.index)]
    if len(non_existing_cp) > 0:
        raise Warning('The following charging points of the mapping are not '
                      'existent in the passed edisgo object: {}.'.format(
            non_existing_cp))
    cp_identifier = ['_'.join(['power', str(mapping_cp.loc[cp, 'ags']), str(mapping_cp.loc[cp, 'cp_idx'])]) for cp in mapping_cp.index]
    non_existing_energy_bands = [identifier for identifier in cp_identifier if identifier not in (energy_bands.columns)]
    if len(non_existing_energy_bands):
        raise Warning('The following identifiers do not have respective '
                      'entries inside the energy bands dataframe: {}. This '
                      'might cause problems in the optimization process.'
                      .format(non_existing_energy_bands))