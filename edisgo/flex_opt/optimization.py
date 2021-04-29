import numpy as np
import pandas as pd
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
from edisgo.tools.tools import get_nodal_residual_load
from edisgo.tools.networkx_helper import get_downstream_nodes_matrix


def setup_model(edisgo, timesteps=None, optimize_storage=True,
                optimize_ev_charging=True, **kwargs):
    """
    Method to set up pyomo model for optimisation of storage procurement
    and/or ev charging with linear approximation of power flow from
    eDisGo-object.

    :param edisgo:
    :return:
    """
    model = pm.ConcreteModel()

    # Todo: Extract kwargs values from cfg?

    # DEFINE SETS AND FIX PARAMETERS
    model.bus_set = pm.Set(initialize=edisgo.topology.buses_df.index)
    model.slack_bus = pm.Set(initialize=edisgo.topology.slack_df.bus)
    if timesteps is not None:
        model.time_set = pm.Set(initialize=timesteps)
    else:
        model.time_set = pm.Set(initialize=edisgo.timeseries.timeindex)
    model.time_zero = [model.time_set[1]]
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set[1], model.time_set[-1]])
    model.line_set = pm.Set(initialize=edisgo.topology.lines_df.index)
    if optimize_storage:
        model.storage_set = pm.Set(initialize=edisgo.topology.storage_units_df.index)
        model.fix_relative_soc = kwargs.get('fix_relative_soc', 0.5)
    if optimize_ev_charging:
        # Todo: have two sets with flexible and inflexible charging points
        model.charging_points_set = \
            pm.Set(initialize=edisgo.topology.charging_points_df.index)
        model.energy_band_charging_points = kwargs.get('energy_band_charging_points')
        model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
    model.residual_load = edisgo.timeseries.residual_load
    model.grid = edisgo.topology
    model.delta_min = kwargs.get('delta_min', 0.9)
    model.delta_max = kwargs.get('delta_max', 0.1)
    model.downstream_nodes_matrix = get_downstream_nodes_matrix(edisgo)
    nodal_active_power, nodal_reactive_power = get_nodal_residual_load(edisgo)
    model.nodal_active_power = nodal_active_power.T
    model.nodal_reactive_power = nodal_reactive_power.T
    model.v_min = kwargs.get("v_min", 0.9)
    model.v_max = kwargs.get("v_max", 1.1)
    model.power_factor = kwargs.get("pf", 0.9)
    model.v_nom = edisgo.topology.buses_df.v_nom.iloc[0]

    # DEFINE VARIABLES
    model.min_load_factor = pm.Var()
    model.max_load_factor = pm.Var()
    model.p_cum = pm.Var(model.line_set, model.time_set,
                     bounds=lambda m, l, t:
                     (-m.power_factor * m.grid.lines_df.loc[l, 's_nom'],
                      m.power_factor * m.grid.lines_df.loc[l, 's_nom']))
    model.q_cum = pm.Var(model.line_set, model.time_set)
    model.v = pm.Var(model.bus_set, model.time_set,
                 bounds=(
                 np.square(model.v_min * model.v_nom), np.square(model.v_max *
                                                                 model.v_nom)))
    if optimize_storage:
        model.soc = \
            pm.Var(model.storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   0, m.grid.storage_units_df.loc[b, 'capacity']))
        model.charging = \
            pm.Var(model.storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   -m.grid.storage_units_df.loc[b, 'p_nom'],
                   m.grid.storage_units_df.loc[b, 'p_nom']))
    if optimize_ev_charging:
        model.charging_ev = \
            pm.Var(model.charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (0, m.energy_band_charging_points.loc[t, 'power_' + b]))
        model.energy_level_ev = \
            pm.Var(model.charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (m.energy_band_charging_points.loc[t, 'lower_' + b],
                    m.energy_band_charging_points.loc[t, 'upper_' + b]))

    # DEFINE CONSTRAINTS
    model.LoadFactorMin = pm.Constraint(model.time_set, rule=load_factor_min)
    model.LoadFactorMax = pm.Constraint(model.time_set, rule=load_factor_max)
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
        model.EVCharging = pm.Constraint(model.charging_points_set,
                                         model.time_non_zero, rule=charging_ev)
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.charging_points_set, model.time_zero,
                          rule=initial_energy_level)

    # DEFINE OBJECTIVE
    model.objective = pm.Objective(rule=minimize_max_residual_load,
                               sense=pm.minimize,
                               doc='Define objective function')

    if kwargs.get('print_model', False):
        model.pprint()
    return model


def optimize(model, solver, save_dir=None):
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
    results = opt.solve(model, tee=True)

    # Extract results
    x_charge = pd.DataFrame()
    soc = pd.DataFrame()
    x_charge_ev = pd.DataFrame()
    energy_band_cp = pd.DataFrame()

    if (results.solver.status == SolverStatus.ok) and \
            (
                    results.solver.termination_condition == TerminationCondition.optimal):
        print('Model Solved to Optimality')
        for time in model.time_set:
            if hasattr(model, 'storage_set'):
                for bus in model.storage_set:
                    x_charge.loc[time, bus] = model.charging[bus, time].value
                    soc.loc[time, bus] = model.soc[bus, time].value
            if hasattr(model, 'charging_points_set'):
                for cp in model.charging_points_set:
                    x_charge_ev.loc[time, cp] = model.charging_ev[cp, time].value
                    energy_band_cp.loc[time, cp] = \
                        model.energy_level_ev[cp, time].value

        res_load_before_storage = model.residual_load.loc[model.time_set]
        res_after_storage = model.residual_load.loc[model.time_set] + x_charge[
            'Storage 1']
        res_load_ev_and_storage = \
            model.residual_load.loc[model.time_set] + x_charge['Storage 1'] + \
            x_charge_ev['ChargingPoint']
        if save_dir:
            x_charge.to_csv(save_dir+'/x_charge_storage.csv')
            soc.to_csv(save_dir+'/soc_storage.csv')
            x_charge_ev.to_csv(save_dir+'/x_charge_ev.csv')
            energy_band_cp.to_csv(save_dir+'/energy_level_ev.csv')
        return x_charge, soc, x_charge_ev, energy_band_cp
    # Do something when the solution in optimal and feasible
    elif (
            results.solver.termination_condition == TerminationCondition.infeasible):
        print('Model is infeasible')
        return
        # Do something when model in infeasible
    else:
        print('Solver Status: ', results.solver.status)
        return


def minimize_max_residual_load(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return -model.delta_min * model.min_load_factor + \
           model.delta_max * model.max_load_factor


def active_power(model, line, time):
    '''
    Constraint for active power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
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
    relevant_storage_units = \
        model.grid.storage_units_df.loc[
            model.grid.storage_units_df.bus.isin(relevant_buses)].index.values
    relevant_charging_points = \
        model.grid.charging_points_df.loc[
            model.grid.charging_points_df.bus.isin(relevant_buses)].index.values
    load_flow_on_line = \
        model.nodal_active_power.loc[relevant_buses, time].sum()
    return model.p_cum[line, time] == load_flow_on_line + \
        sum(model.charging[storage, time]
            for storage in relevant_storage_units) - \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def reactive_power(model, line, time):
    '''
    Constraint for reactive power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
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
        model.nodal_reactive_power.loc[relevant_buses, time].sum()
    return model.q_cum[line, time] == load_flow_on_line


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
    return model.min_load_factor <= model.residual_load.loc[time] + \
        sum(model.charging[storage, time] for storage in model.storage_set) - \
        sum(model.charging_ev[cp, time] for cp in model.charging_points_set)


def load_factor_max(model, time):
    '''
    Constraint that describes the load factor.
    :param model:
    :param time:
    :return:
    '''
    return model.max_load_factor >= model.residual_load.loc[time] + \
        sum(model.charging[storage, time] for storage in model.storage_set)- \
        sum(model.charging_ev[cp, time] for cp in model.charging_points_set)


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