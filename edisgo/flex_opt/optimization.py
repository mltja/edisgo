import numpy as np
import pandas as pd
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
from edisgo.tools.tools import get_nodal_residual_load, calculate_impedance_for_parallel_components
from copy import deepcopy
import itertools


def setup_model(edisgo, downstream_node_matrix, timesteps=None, optimize_storage=True,
                optimize_ev_charging=True, objective='curtailment', pu=True, **kwargs):
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
    edisgo_object, grid_object, slack = setup_grid_object(edisgo)
    # check if correct value of objective is inserted
    if objective not in ['curtailment', 'peak_load', 'minimize_energy_level',
                         'residual_load', 'maximize_energy_level']:
        raise ValueError('The objective you inserted is not implemented yet.')
    # check if multiple voltage levels are present
    if len(grid_object.buses_df.v_nom.unique()) > 1 and not pu:
        print('More than one voltage level included. Please make sure to '
              'adapt all impedance values to one reference system.')

    # Todo: Extract kwargs values from cfg?

    # DEFINE SETS AND FIX PARAMETERS
    print('Setup model: Defining sets and parameters.')
    model.bus_set = pm.Set(initialize=grid_object.buses_df.index)
    model.slack_bus = pm.Set(initialize=slack)
    if timesteps is not None:
        model.timeindex = timesteps
    else:
        model.timeindex = edisgo_object.timeseries.timeindex
    model.time_increment = pd.infer_freq(model.timeindex)
    if not any(char.isdigit() for char in model.time_increment):
        model.time_increment = '1' + model.time_increment
    model.time_set = pm.RangeSet(0, len(model.timeindex)-1)
    model.time_zero = [model.time_set[1]]
    model.time_end = [model.time_set[-1]]
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set[1],
                                               model.time_set[-1]])
    if optimize_storage:
        model.storage_set = \
            pm.Set(initialize=grid_object.storage_units_df.index)
        optimized_storage_units = \
            kwargs.get('flexible_storage_units',
                       grid_object.storage_units_df.index)
        model.optimized_storage_set = \
            pm.Set(initialize=optimized_storage_units)
        model.fixed_storage_set = model.storage_set - \
                                  model.optimized_storage_set
        model.fix_relative_soc = kwargs.get('fix_relative_soc', 0.5)
        inflexible_storage_units = list(model.fixed_storage_set.data())
    else:
        inflexible_storage_units = None
    if optimize_ev_charging:
        model.energy_band_charging_points = \
            kwargs.get('energy_band_charging_points')
        model.mapping_cp = kwargs.get('mapping_cp')
        model.charging_points_set = \
            pm.Set(initialize=grid_object.charging_points_df.index)
        model.flexible_charging_points_set = \
            pm.Set(initialize=model.mapping_cp.index)
        model.inflexible_charging_points_set = \
            model.charging_points_set - model.flexible_charging_points_set
        model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
        inflexible_charging_points = \
            list(model.inflexible_charging_points_set.data())
    else:
        inflexible_charging_points = None
    model.v_min = kwargs.get("v_min", 0.9)
    model.v_max = kwargs.get("v_max", 1.1)
    model.power_factor = kwargs.get("pf", 0.9)
    # handle pu conversion
    if pu:
        model.v_nom = 1.0
        s_base = kwargs.get("s_base", 1)
        grid_object.convert_to_pu_system(s_base, timeseries_inplace=True)
        model.pars = {'r': 'r_pu', 'x': 'x_pu', 's_nom': 's_nom_pu',
                      'p_nom': 'p_nom_pu', 'peak_load': 'peak_load_pu',
                      'capacity': 'capacity_pu'}
    else:
        model.v_nom = grid_object.buses_df.v_nom.iloc[0]
        model.pars = {'r': 'r', 'x': 'x', 's_nom': 's_nom',
                      'p_nom': 'p_nom', 'peak_load': 'peak_load',
                      'capacity': 'capacity'}
        grid_object.transformers_df['r'] = grid_object.transformers_df[
                                               'r_pu'] * np.square(
            model.v_nom) / grid_object.transformers_df.s_nom
        grid_object.transformers_df['x'] = grid_object.transformers_df[
                                               'x_pu'] * np.square(
            model.v_nom) / grid_object.transformers_df.s_nom
    model.residual_load = get_residual_load_of_not_optimized_components(
        grid_object, edisgo_object, model)
    model.grid = grid_object
    model.downstream_nodes_matrix = downstream_node_matrix
    nodal_active_power, nodal_reactive_power, nodal_active_load, nodal_reactive_load, \
           nodal_active_generation, nodal_reactive_generation, nodal_active_charging_points, \
           nodal_reactive_charging_points, nodal_active_storage, nodal_reactive_storage = get_nodal_residual_load(
        grid_object, edisgo_object, considered_storage=inflexible_storage_units,
        considered_charging_points=inflexible_charging_points)
    model.nodal_active_power = nodal_active_power.T
    model.nodal_reactive_power = nodal_reactive_power.T
    model.nodal_active_load = nodal_active_load.T + nodal_active_charging_points.T
    model.nodal_reactive_load = nodal_reactive_load.T
    model.nodal_active_feedin = nodal_active_generation.T
    model.nodal_reactive_feedin = nodal_reactive_generation.T
    model.v_slack = kwargs.get('v_slack', model.v_nom)
    model.branches = concat_parallel_branch_elements(grid_object)
    model.underlying_branch_elements, model.power_factors = get_underlying_elements(model)

    model.branch_set = pm.Set(initialize=model.branches.index)

    if objective == 'peak_load':
        model.delta_min = kwargs.get('delta_min', 0.9)
        model.delta_max = kwargs.get('delta_max', 0.1)
        model.min_load_factor = pm.Var()
        model.max_load_factor = pm.Var()
    elif objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.grid_power_flexible = pm.Var(model.time_set)

    # add n-1 security
    # adapt i_lines_allowed for radial feeders
    buses_in_cycles = list(
        set(itertools.chain.from_iterable(edisgo_object.topology.rings)))

    # Find lines in cycles
    lines_in_cycles = list(
        grid_object.lines_df.loc[grid_object.lines_df[[
            'bus0', 'bus1']].isin(buses_in_cycles).all(
            axis=1)].index.values)

    model.branches_load_factors = pd.DataFrame(index=model.time_set,
                                               columns=model.branch_set)
    model.branches_load_factors.loc[:, :] = 1
    tmp_residual_load = edisgo_object.timeseries.residual_load.loc[timesteps]
    indices = pd.DataFrame(index=timesteps, columns=['index'])
    indices['index'] = [i for i in range(len(timesteps))]
    model.branches_load_factors.loc[
        indices.loc[tmp_residual_load.loc[timesteps] < 0].values.T[0],
        lines_in_cycles
    ] = kwargs.get('load_factor_rings', 1.0) #0.5

    # DEFINE VARIABLES
    print('Setup model: Defining variables.')
    model.p_cum = pm.Var(model.branch_set, model.time_set)
    model.slack_p_cum_pos = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.slack_p_cum_neg = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.q_cum = pm.Var(model.branch_set, model.time_set)
    model.v = pm.Var(model.bus_set, model.time_set)
    model.slack_v_pos = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    model.slack_v_neg = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    # if not objective == 'minimize_energy_level' and \
    #         not objective == 'maximize_energy_level':
    model.curtailment_load = pm.Var(model.bus_set, model.time_set,
                                    bounds=(0, None))
    model.curtailment_feedin = pm.Var(model.bus_set, model.time_set,
                                      bounds=lambda m, b, t:
                                      (0, m.nodal_active_feedin.loc[b, model.timeindex[t]]))
    model.curtailment_reactive_load = pm.Var(model.bus_set, model.time_set,
                                             bounds=lambda m, b, t:
                                             (0, m.nodal_reactive_load.loc[b, model.timeindex[t]]))
    model.curtailment_reactive_feedin = pm.Var(model.bus_set, model.time_set,
                                               bounds=lambda m, b, t:
                                               (0, abs(m.nodal_reactive_feedin.loc[b, model.timeindex[t]])))#Todo: change back
    if optimize_storage:
        model.soc = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   0, m.grid.storage_units_df.loc[b, model.pars['capacity']]))
        model.charging = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   -m.grid.storage_units_df.loc[b, model.pars['p_nom']],
                   m.grid.storage_units_df.loc[b, model.pars['p_nom']]))
    if optimize_ev_charging:
        if pu:
            print('Optimisation in pu-system. Make sure the inserted energy '
                  'bands are also converted to the same pu-system.')
        model.charging_ev = \
            pm.Var(model.flexible_charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (0, m.energy_band_charging_points.loc[
                       m.timeindex[t], '_'.join(
                           ['power', str(m.mapping_cp.loc[b, 'ags']),
                            str(m.mapping_cp.loc[b, 'cp_idx']),
                            m.mapping_cp.loc[b, 'use_case']])]))
        if not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
            model.energy_level_ev = \
                pm.Var(model.flexible_charging_points_set, model.time_set,
                       bounds=lambda m, b, t:
                       (m.energy_band_charging_points.loc[
                            m.timeindex[t], '_'.join(
                                ['lower', str(m.mapping_cp.loc[b, 'ags']),
                                 str(m.mapping_cp.loc[b, 'cp_idx']),
                                 m.mapping_cp.loc[b, 'use_case']])],
                        m.energy_band_charging_points.loc[
                            m.timeindex[t], '_'.join(
                                ['upper', str(m.mapping_cp.loc[b, 'ags']),
                                 str(m.mapping_cp.loc[b, 'cp_idx']),
                                 m.mapping_cp.loc[b, 'use_case']])]))

    # DEFINE CONSTRAINTS
    print('Setup model: Setting constraints.')
    model.ActivePower = pm.Constraint(model.branch_set, model.time_set,
                                      rule=active_power)
    model.UpperActive = pm.Constraint(model.branch_set, model.time_set,
                                      rule=upper_active_power)
    model.LowerActive = pm.Constraint(model.branch_set, model.time_set,
                                      rule=lower_active_power)
    model.ReactivePower = pm.Constraint(model.branch_set, model.time_set,
                                        rule=reactive_power)
    model.SlackVoltage = pm.Constraint(model.slack_bus, model.time_set,
                                       rule=slack_voltage)
    model.VoltageDrop = pm.Constraint(model.branch_set, model.time_set,
                                      rule=voltage_drop)
    model.UpperVoltage = pm.Constraint(model.bus_set, model.time_set,
                                       rule=upper_voltage)
    model.LowerVoltage = pm.Constraint(model.bus_set, model.time_set,
                                       rule=lower_voltage)
    model.UpperCurtLoad = pm.Constraint(model.bus_set, model.time_set,
                                        rule=upper_bound_curtailment_load)
    if optimize_storage:
        model.BatteryCharging = pm.Constraint(model.storage_set,
                                              model.time_non_zero, rule=soc)
        model.FixedSOC = pm.Constraint(model.storage_set,
                                       model.times_fixed_soc, rule=fix_soc)
    if optimize_ev_charging and not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
        model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                         model.time_non_zero, rule=charging_ev)
        # set initial energy level
        model.energy_level_start = kwargs.get('energy_level_start', None)
        if model.energy_level_start is not None:
            model.slack_initial_energy_pos = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
            model.slack_initial_energy_neg = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
            model.InitialEVEnergyLevel = \
                pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                                  rule=initial_energy_level)
        else:
            model.InitialEVEnergyLevel = \
                pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                              rule=fixed_energy_level)
        # set final energy level and if necessary charging power
        model.energy_level_end = kwargs.get('energy_level_end', None)
        if model.energy_level_end is not None:
            if type(model.energy_level_end) == bool:
                model.FinalEVEnergyLevel = \
                    pm.Constraint(model.flexible_charging_points_set, model.time_end,
                                  rule=fixed_energy_level)
            else:
                model.energy_level_beginning = kwargs.get('energy_level_beginning',
                                                          None)
                if model.energy_level_beginning is None:
                    model.energy_level_beginning = pd.Series(index=model.flexible_charging_points_set,
                                                             data=[0]*len(model.flexible_charging_points_set))
                model.FinalEVEnergyLevel = \
                    pm.Constraint(model.flexible_charging_points_set, model.time_end,
                                  rule=final_energy_level)
                model.FinalEVChargingPower = \
                    pm.Constraint(model.flexible_charging_points_set, model.time_end,
                                  rule=final_charging_power)
        # set initial charging power
        model.charging_initial = kwargs.get('charging_start', None)
        if model.charging_initial is not None:
            model.slack_initial_charging_pos = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
            model.slack_initial_charging_neg = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
            model.InitialEVChargingPower = \
                pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                              rule=initial_charging_power)

    if objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.AggrGrid = pm.Constraint(model.time_set, rule=aggregated_power)

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
    elif objective == 'minimize_energy_level':
        model.objective = pm.Objective(rule=minimize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'maximize_energy_level':
        model.objective = pm.Objective(rule=maximize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'residual_load':
        model.grid_residual_load = pm.Var(model.time_set)
        model.GridResidualLoad = pm.Constraint(model.time_set,
                                               rule=grid_residual_load)
        model.objective = pm.Objective(rule=minimize_residual_load,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    else:
        raise Exception('Unknown objective.')

    if kwargs.get('print_model', False):
        model.pprint()
    print('Successfully set up optimisation model.')
    return model


def concat_parallel_branch_elements(grid_object):
    """
    Method to merge parallel lines and transformers into one element, respectively.

    Parameters
    ----------
    grid_object

    Returns
    -------

    """
    lines = fuse_parallel_branches(grid_object.lines_df)
    trafos = grid_object.transformers_df.loc[
        grid_object.transformers_df.bus0.isin(grid_object.buses_df.index)].loc[
        grid_object.transformers_df.bus1.isin(grid_object.buses_df.index)]
    transformers = fuse_parallel_branches(trafos)
    return pd.concat([lines, transformers], sort=False)


def fuse_parallel_branches(branches):
    branches_tmp = branches[['bus0', 'bus1']]
    parallel_branches = pd.DataFrame(columns=branches.columns)
    if branches_tmp.duplicated().any():
        duplicated_branches = branches_tmp.loc[branches_tmp.duplicated(keep=False)]
        duplicated_branches['visited'] = False
        branches_tmp.drop(duplicated_branches.index, inplace=True)
        for name, buses in duplicated_branches.iterrows():
            if duplicated_branches.loc[name, 'visited']:
                continue
            else:
                parallel_branches_tmp = duplicated_branches.loc[(duplicated_branches == buses).all(axis=1)]
                duplicated_branches.loc[parallel_branches_tmp.index, 'visited'] = True
                name_par = '_'.join(str.split(name, '_')[:-1])
                parallel_branches.loc[name_par] = branches.loc[name]
                parallel_branches.loc[name_par, ['r', 'x', 's_nom']] = calculate_impedance_for_parallel_components(
                    branches.loc[parallel_branches_tmp.index, ['r', 'x', 's_nom']],
                    pu=False)
    fused_branches = pd.concat([branches.loc[branches_tmp.index], parallel_branches], sort=False)
    return fused_branches


def setup_model_wo_bands(edisgo, downstream_node_matrix, timesteps=None, optimize_storage=True,
                optimize_ev_charging=True, objective='curtailment', pu=True, **kwargs):
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
    edisgo_object, grid_object, slack = setup_grid_object(edisgo)
    # check if correct value of objective is inserted
    if objective not in ['curtailment', 'peak_load', 'minimize_energy_level',
                         'maximize_energy_level']:
        raise ValueError('The objective you inserted is not implemented yet.')
    # check if multiple voltage levels are present
    if len(grid_object.buses_df.v_nom.unique()) > 1 and not pu:
        print('More than one voltage level included. Please make sure to '
              'adapt all impedance values to one reference system.')

    # Todo: Extract kwargs values from cfg?

    # DEFINE SETS AND FIX PARAMETERS
    print('Setup model: Defining sets and parameters.')
    model.bus_set = pm.Set(initialize=grid_object.buses_df.index)
    model.slack_bus = pm.Set(initialize=slack)
    if timesteps is not None:
        model.timeindex = timesteps
    else:
        model.timeindex = edisgo_object.timeseries.timeindex
    model.time_increment = pd.infer_freq(model.timeindex)
    if not any(char.isdigit() for char in model.time_increment):
        model.time_increment = '1' + model.time_increment
    model.time_set = pm.RangeSet(0, len(model.timeindex)-1)
    model.time_zero = [model.time_set[1]]
    # fix end level
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.times_fixed_soc = pm.Set(initialize=[model.time_set[1],
                                               model.time_set[-1]])
    if optimize_storage:
        model.storage_set = \
            pm.Set(initialize=grid_object.storage_units_df.index)
        optimized_storage_units = \
            kwargs.get('flexible_storage_units',
                       grid_object.storage_units_df.index)
        model.optimized_storage_set = \
            pm.Set(initialize=optimized_storage_units)
        model.fixed_storage_set = model.storage_set - \
                                  model.optimized_storage_set
        model.fix_relative_soc = kwargs.get('fix_relative_soc', 0.5)
        inflexible_storage_units = list(model.fixed_storage_set.data())
    else:
        inflexible_storage_units = None
    if optimize_ev_charging:
        standing_times = kwargs.get('standing_times')
        model.charging_events = standing_times.loc[standing_times.chargingdemand>0]
        model.charging_events_set = pm.Set(initialize=standing_times.index)
        model.mapping_cp = kwargs.get('mapping_cp')
        model.charging_points_set = \
            pm.Set(initialize=grid_object.charging_points_df.index)
        model.flexible_charging_points_set = \
            pm.Set(initialize=model.mapping_cp.index)
        model.inflexible_charging_points_set = \
            model.charging_points_set - model.flexible_charging_points_set
        model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
        inflexible_charging_points = \
            list(model.inflexible_charging_points_set.data())
    else:
        inflexible_charging_points = None
    model.v_min = kwargs.get("v_min", 0.9)
    model.v_max = kwargs.get("v_max", 1.1)
    model.power_factor = kwargs.get("pf", 0.9)
    # handle pu conversion
    if pu:
        model.v_nom = 1.0
        s_base = kwargs.get("s_base", 1)
        grid_object.convert_to_pu_system(s_base, timeseries_inplace=True)
        model.pars = {'r': 'r_pu', 'x': 'x_pu', 's_nom': 's_nom_pu',
                      'p_nom': 'p_nom_pu', 'peak_load': 'peak_load_pu',
                      'capacity': 'capacity_pu'}
    else:
        model.v_nom = grid_object.buses_df.v_nom.iloc[0]
        model.pars = {'r': 'r', 'x': 'x', 's_nom': 's_nom',
                      'p_nom': 'p_nom', 'peak_load': 'peak_load',
                      'capacity': 'capacity'}
        grid_object.transformers_df['r'] = grid_object.transformers_df[
                                               'r_pu'] * np.square(
            model.v_nom) / grid_object.transformers_df.s_nom
        grid_object.transformers_df['x'] = grid_object.transformers_df[
                                               'x_pu'] * np.square(
            model.v_nom) / grid_object.transformers_df.s_nom
    model.residual_load = get_residual_load_of_not_optimized_components(
        grid_object, edisgo_object, model)
    model.grid = grid_object
    model.downstream_nodes_matrix = downstream_node_matrix
    nodal_active_power, nodal_reactive_power, nodal_active_load, nodal_reactive_load, \
           nodal_active_generation, nodal_reactive_generation = get_nodal_residual_load(
        grid_object, edisgo_object, considered_storage=inflexible_storage_units,
        considered_charging_points=inflexible_charging_points)
    model.nodal_active_power = nodal_active_power.T
    model.nodal_reactive_power = nodal_reactive_power.T
    model.nodal_active_power_load = nodal_active_load.T
    model.nodal_reactive_power_load = nodal_reactive_load.T
    model.nodal_active_power_generation = nodal_active_generation.T
    model.nodal_reactive_power_generation = nodal_reactive_generation.T
    model.v_slack = kwargs.get('v_slack', model.v_nom)
    trafos = grid_object.transformers_df.loc[
        grid_object.transformers_df.bus0.isin(grid_object.buses_df.index)].loc[
        grid_object.transformers_df.bus1.isin(grid_object.buses_df.index)]
    model.branches = pd.concat([grid_object.lines_df, trafos])

    model.branch_set = pm.Set(initialize=model.branches.index)

    if objective == 'peak_load':
        model.delta_min = kwargs.get('delta_min', 0.9)
        model.delta_max = kwargs.get('delta_max', 0.1)
        model.min_load_factor = pm.Var()
        model.max_load_factor = pm.Var()
    elif objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.grid_power_flexible = pm.Var(model.time_set)

    # add n-1 security
    # adapt i_lines_allowed for radial feeders
    buses_in_cycles = list(
        set(itertools.chain.from_iterable(edisgo_object.topology.rings)))

    # Find lines in cycles
    lines_in_cycles = list(
        grid_object.lines_df.loc[grid_object.lines_df[[
            'bus0', 'bus1']].isin(buses_in_cycles).all(
            axis=1)].index.values)

    model.branches_load_factors = pd.DataFrame(index=model.time_set,
                                               columns=model.branch_set)
    model.branches_load_factors.loc[:, :] = 1
    tmp_residual_load = edisgo_object.timeseries.residual_load.loc[timesteps]
    indices = pd.DataFrame(index=timesteps, columns=['index'])
    indices['index'] = [i for i in range(len(timesteps))]
    model.branches_load_factors.loc[
        indices.loc[tmp_residual_load.loc[timesteps] < 0].values.T[0],
        lines_in_cycles
    ] = kwargs.get('load_factor_rings', 1.0) #0.5

    # DEFINE VARIABLES
    print('Setup model: Defining variables.')
    model.p_cum = pm.Var(model.branch_set, model.time_set,
                     bounds=lambda m, l, t:
                     (-m.power_factor * m.branches.loc[l, model.pars['s_nom']] *
                      m.branches_load_factors.loc[t, l],
                      m.power_factor * m.branches.loc[l, model.pars['s_nom']] *
                      m.branches_load_factors.loc[t, l]))
    model.v = pm.Var(
        model.bus_set, model.time_set, bounds=(
            np.square(model.v_min * model.v_nom), np.square(model.v_max *
                                                            model.v_nom)))
    # if not objective == 'minimize_energy_level' and \
    #         not objective == 'maximize_energy_level':
    model.curtailment_load = pm.Var(model.bus_set, model.time_set,
                                    bounds=(0, None))
    model.curtailment_feedin = pm.Var(model.bus_set, model.time_set,
                                      bounds=(0, None))
    if optimize_storage:
        model.soc = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   0, m.grid.storage_units_df.loc[b, model.pars['capacity']]))
        model.charging = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   -m.grid.storage_units_df.loc[b, model.pars['p_nom']],
                   m.grid.storage_units_df.loc[b, model.pars['p_nom']]))
    if optimize_ev_charging:
        if pu:
            print('Optimisation in pu-system. Make sure the inserted energy '
                  'bands are also converted to the same pu-system.')
        model.charging_ev = \
            pm.Var(model.flexible_charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (0, m.energy_band_charging_points.loc[
                       m.timeindex[t], '_'.join(['power', str(m.mapping_cp.loc[b, 'ags']),
                                    str(m.mapping_cp.loc[b, 'cp_idx'])])]))
        if not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
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
    model.ActivePower = pm.Constraint(model.branch_set, model.time_set,
                                      rule=active_power)
    model.SlackVoltage = pm.Constraint(model.slack_bus, model.time_set,
                                       rule=slack_voltage)
    model.VoltageDrop = pm.Constraint(model.branch_set, model.time_set,
                                      rule=voltage_drop)
    if optimize_storage:
        model.BatteryCharging = pm.Constraint(model.storage_set,
                                              model.time_non_zero, rule=soc)
        model.FixedSOC = pm.Constraint(model.storage_set,
                                       model.times_fixed_soc, rule=fix_soc)
    if optimize_ev_charging and not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
        model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                         model.time_non_zero, rule=charging_ev)
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.flexible_charging_points_set, model.times_fixed_soc,
                          rule=fixed_energy_level)
    if objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.AggrGrid = pm.Constraint(model.time_set, rule=aggregated_power)

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
    elif objective == 'minimize_energy_level':
        model.objective = pm.Objective(rule=minimize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'maximize_energy_level':
        model.objective = pm.Objective(rule=maximize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    else:
        raise Exception('Unknown objective.')

    if kwargs.get('print_model', False):
        model.pprint()
    print('Successfully set up optimisation model.')
    return model


def setup_grid_object(edisgo):
    if hasattr(edisgo, 'topology'):
        grid_object = deepcopy(edisgo.topology)
        edisgo_object = deepcopy(edisgo)
        slack = grid_object.slack_df.bus
    else:
        grid_object = deepcopy(edisgo)
        edisgo_object = deepcopy(edisgo.edisgo_obj)
        slack = [grid_object.transformers_df.bus1.iloc[
                     0]]  # Todo: careful with MV grid, does not work with that right?
    return edisgo_object, grid_object, slack


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


def setup_model_bands(aggregated_bands, power_bands, mode, **kwargs):
    model = pm.ConcreteModel()
    print('Setup Model: Defining Parameters and Sets')
    model.timeindex = aggregated_bands.index
    model.time_increment = pd.infer_freq(model.timeindex)
    model.time_set = pm.RangeSet(0, len(model.timeindex) - 1)
    model.time_non_zero = model.time_set - [model.time_set[1]]
    model.energy_bands = aggregated_bands
    model.power_bands = power_bands
    model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
    print('Setup Model: Defining Variables')
    model.charging_power = \
        pm.Var(model.time_set, bounds=lambda m, t:
        (m.power_bands.loc[m.timeindex[t], 'lower'],
         m.power_bands.loc[m.timeindex[t], 'upper']))
    model.energy_level = \
        pm.Var(model.time_set, bounds=lambda m, t:
        (m.energy_bands.loc[m.timeindex[t], 'lower'],
         m.energy_bands.loc[m.timeindex[t], 'upper']))
    print('Setup Model: Defining Constraints')
    model.charging = pm.Constraint(model.time_non_zero, rule=charging_flex)
    print('Setup Model: Defining Objective')
    if mode == 'minimize':
        model.objective = pm.Objective(rule=maximize_energy_band,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif mode == 'maximize':
        model.objective = pm.Objective(rule=maximize_energy_band,
                                       sense=pm.maximize,
                                       doc='Define objective function')
    else:
        raise ValueError('Only mode "minimize" and "maximize" are defined '
                         'so far.')
    if kwargs.get('print_model', False):
        model.pprint()
    return model


def optimize(model, solver, save_dir=None, load_solutions=True, mode=None):
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
    #opt.options['preprocessing presolve'] = 0

    # Optimize
    results = opt.solve(model, tee=True, #options={"threads":4},
                        load_solutions=load_solutions)

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
    slack_charging = pd.DataFrame(columns=['slack'])
    slack_energy = pd.DataFrame(columns=['slack'])
    slack_v_pos = pd.DataFrame()
    slack_v_neg = pd.DataFrame()
    slack_p_cum_pos = pd.DataFrame()
    slack_p_cum_neg = pd.DataFrame()
    # Todo: extract slacks for voltage and loading limits
    if mode == 'energy_band':
        p_aggr = pd.DataFrame()

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
            if hasattr(model, 'flexible_charging_points_set'):
                for cp in model.flexible_charging_points_set:
                    x_charge_ev.loc[timeindex, cp] = model.charging_ev[cp, time].value
                    if hasattr(model, 'energy_level_ev'):
                        energy_level_cp.loc[timeindex, cp] = \
                            model.energy_level_ev[cp, time].value
                    if hasattr(model, 'slack_initial_charging_pos'):
                        slack_charging.loc[cp, 'slack'] = model.slack_initial_charging_pos[cp].value + model.slack_initial_charging_neg[cp].value
                    if hasattr(model, 'slack_initial_energy_pos'):
                        slack_energy.loc[cp, 'slack'] = model.slack_initial_energy_pos[cp].value + model.slack_initial_energy_neg[cp].value
            for bus in model.bus_set:
                curtailment_feedin.loc[timeindex, bus] = \
                    model.curtailment_feedin[bus, time].value
                curtailment_load.loc[timeindex, bus] = \
                    model.curtailment_load[bus, time].value
                curtailment_reactive_feedin.loc[timeindex, bus] = \
                    model.curtailment_reactive_feedin[bus, time].value
                curtailment_reactive_load.loc[timeindex, bus] = \
                    model.curtailment_reactive_load[bus, time].value
                slack_v_pos.loc[timeindex, bus]  = model.slack_v_pos[bus, time].value
                slack_v_neg.loc[timeindex, bus] = model.slack_v_neg[bus, time].value
                try:
                    v_bus.loc[timeindex, bus] = np.sqrt(model.v[bus, time].value)
                except:
                    print('Error for bus {} at time {}'.format(bus, time))
            for line in model.branch_set:
                p_line.loc[timeindex, line] = model.p_cum[line, time].value
                q_line.loc[timeindex, line] = model.q_cum[line, time].value
                slack_p_cum_pos.loc[timeindex, line] = model.slack_p_cum_pos[line, time].value
                slack_p_cum_neg.loc[timeindex, line] = model.slack_p_cum_neg[line, time].value
            if mode == 'energy_band':
                p_aggr.loc[timeindex, repr(model.grid)] = model.grid_power_flexible[time].value
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
        if not mode=='energy_band':
            return x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
                   curtailment_load,  curtailment_reactive_feedin, curtailment_reactive_load, \
                   v_bus, p_line, q_line, slack_charging, slack_energy, slack_v_pos, slack_v_neg, \
                   slack_p_cum_pos, slack_p_cum_neg
        else:
            return x_charge, soc, x_charge_ev, energy_level_cp, curtailment_feedin, \
                   curtailment_load,  curtailment_reactive_feedin, curtailment_reactive_load, \
                   v_bus, p_line, q_line, p_aggr
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('Model is infeasible')
        return
        # Do something when model in infeasible
    else:
        print('Solver Status: ', results.solver.status)
        return


def optimize_bands(model, solver, mode):
    """

    :param model:
    :param solver:
    :param mode: either "minimize" or "maximize"
    :return:
    """
    opt = pm.SolverFactory(solver)
    # opt.options['preprocessing presolve'] = 0

    # Optimize
    results = opt.solve(model, tee=True)

    # Extract results
    energy_level = pd.DataFrame()
    charging = pd.DataFrame()
    if mode == "minimize":
        label = 'lower'
    elif mode == "maximize":
        label = 'upper'
    else:
        raise ValueError('Only modes "minimize" and "maximize" are '
                         'implemented so far.')

    if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
        print('Model Solved to Optimality')
        for time in model.time_set:
            timeindex = model.timeindex[time]
            energy_level.loc[timeindex, label] = model.energy_level[time].value
            charging.loc[timeindex, label] = model.charging_power[time].value
        return energy_level, charging
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('Model is infeasible')
        return
        # Do something when model in infeasible
    else:
        print('Solver Status: ', results.solver.status)
        return


def get_residual_load_of_not_optimized_components(grid, edisgo, model):
    """
    Method to get residual load of fixed components.

    :param edisgo:
    :param model:
    :return:
    """
    relevant_generators = grid.generators_df.index
    relevant_loads = grid.loads_df.index
    if hasattr(model, 'fixed_storage_set'):
        relevant_storage_units = model.fixed_storage_set
    else:
        relevant_storage_units = grid.storage_units_df.index

    if hasattr(model, 'inflexible_charging_points_set'):
        relevant_charging_points = model.inflexible_charging_points_set
    else:
        relevant_charging_points = grid.charging_points_df.index

    if edisgo.timeseries.charging_points_active_power.empty:
        return (
                edisgo.timeseries.generators_active_power[
                    relevant_generators].sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power[relevant_loads
                ].sum(axis=1)
        ).loc[edisgo.timeseries.timeindex]
    else:
        return (
                edisgo.timeseries.generators_active_power[
                    relevant_generators].sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power[relevant_loads
                ].sum(axis=1)
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
               model.curtailment_feedin[bus, time]+
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)


def minimize_residual_load_depr(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return sum([1e-5*np.square([model.residual_load.loc[model.timeindex[time]] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units) - \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points)
        # + sum(model.curtailment_load[bus, time] -
        #     model.curtailment_feedin[bus, time] for bus in model.bus_set)
                         ]) for time in model.time_set])+ \
        sum(model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time]+
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
            for bus in model.bus_set for time in model.time_set)


def minimize_residual_load(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    if hasattr(model, 'slack_initial_charging_pos'):
        slack_charging = sum(model.slack_initial_charging_pos[cp]+model.slack_initial_charging_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_charging = 0
    if hasattr(model, 'slack_initial_energy_pos'):
        slack_energy = sum(model.slack_initial_energy_pos[cp]+model.slack_initial_energy_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_energy = 0
    return 1e-5*sum(model.grid_residual_load[time]**2 for time in model.time_set) + \
        sum(1e-2*(model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time]+
           model.curtailment_reactive_load[bus, time] +
           model.curtailment_reactive_feedin[bus, time]) +
            1000* (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
        for bus in model.bus_set for time in model.time_set) + 1000*(slack_charging + slack_energy) + \
        1000*sum(model.slack_p_cum_pos[branch, time] + model.slack_p_cum_pos[branch, time]
                 for branch in model.branch_set for time in model.time_set)


def grid_residual_load(model, time):
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.grid_residual_load[time] == \
    model.residual_load.loc[model.timeindex[time]] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units) - \
    sum(model.charging_ev[cp, time] for cp in relevant_charging_points)  #+ \
    # sum(model.curtailment_load[bus, time] -
    #     model.curtailment_feedin[bus, time] for bus in model.bus_set)


def minimize_curtailment(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return sum(model.curtailment_load[bus, time] +
               model.curtailment_feedin[bus, time]+
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)


def minimize_energy_level(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return sum(model.curtailment_load[bus, time] +
               model.curtailment_feedin[bus, time]+
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)*1e6 + \
           sum(model.grid_power_flexible[time] for
               time in model.time_set)


def maximize_energy_level(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    return sum(model.curtailment_load[bus, time] +
               model.curtailment_feedin[bus, time]+
               model.curtailment_reactive_load[bus, time] +
               model.curtailment_reactive_feedin[bus, time]
               for bus in model.bus_set
               for time in model.time_set)*1e6 - \
           sum(model.grid_power_flexible[time] for
               time in model.time_set)


def maximize_energy_band(model):
    return sum(model.energy_level[t] for t in model.time_set)


def minimize_energy_band(model):
    return sum(model.energy_level[t] for t in model.time_set)


def active_power(model, branch, time):
    '''
    Constraint for active power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    relevant_buses = model.underlying_branch_elements.loc[branch, 'buses']
    relevant_storage_units = model.underlying_branch_elements.loc[branch, 'flexible_storage']
    relevant_charging_points = model.underlying_branch_elements.loc[branch, 'flexible_ev']
    load_flow_on_line = \
        model.nodal_active_power.loc[relevant_buses, timeindex].sum()
    return model.p_cum[branch, time] == load_flow_on_line + \
           sum(model.charging[storage, time]
            for storage in relevant_storage_units) - \
           sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + \
           sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in relevant_buses)


def upper_active_power(model, branch, time):
    return model.p_cum[branch, time] <= model.power_factors.loc[branch, model.timeindex[time]] * \
           model.branches.loc[branch, model.pars['s_nom']] + model.slack_p_cum_pos[branch, time]


def lower_active_power(model, branch, time):
    return model.p_cum[branch, time] >= - model.power_factors.loc[branch, model.timeindex[time]] * \
           model.branches.loc[branch, model.pars['s_nom']] - model.slack_p_cum_neg[branch, time]


def reactive_power(model, branch, time):
    '''
    Constraint for reactive power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    relevant_buses = model.underlying_branch_elements.loc[branch, 'buses']
    load_flow_on_line = \
        model.nodal_reactive_power.loc[relevant_buses, timeindex].sum()
    return model.q_cum[branch, time] == load_flow_on_line + \
           sum(model.curtailment_reactive_load[bus, time] +
            model.curtailment_reactive_feedin[bus, time] for bus in relevant_buses)


def upper_bound_curtailment_load(model, bus, time):
    '''
    Constraint for upper bound of curtailment load
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    relevant_buses = [bus]
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
    else:
        relevant_charging_points = []
    return model.curtailment_load[bus, time] <= model.nodal_active_load.loc[bus, timeindex] \
           -sum(model.charging[storage, time]
            for storage in relevant_storage_units) + \
           sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def ratio_active_reactive_power_load(model, bus, time):
    """
    Constraint connecting active and reactive power
    Todo: change curtailment to EV and other load -> load factor stays constant
    Parameters
    ----------
    model
    bus
    time

    Returns
    -------

    """
    relevant_buses = [bus]
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
    else:
        relevant_charging_points = []
    timeindex = model.timeindex[time]
    return model.curtailment_reactive_load[bus, time] == model.nodal_reactive_power.loc[bus, timeindex]*\
        model.curtailment_load[bus, time] / \
           (-model.nodal_active_power.loc[bus, timeindex] -
            sum(model.charging[storage, time] for storage in relevant_storage_units) + \
            sum(model.charging_ev[cp, time] for cp in relevant_charging_points))


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
           model.charging[storage, time - 1] *\
           (pd.to_timedelta(model.time_increment)/pd.to_timedelta('1h'))


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
           model.grid.storage_units_df.loc[bus, model.pars['capacity']]


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
           model.charging_ev[charging_point, time - 1]*\
           (pd.to_timedelta(model.time_increment)/pd.to_timedelta('1h'))


def charging_flex(model, time):
    """
    Constraint for charging of EV that has to lie between the lower and upper
    energy band.

    :param model:
    :param charging_point:
    :param time:
    :return:
    """
    return model.energy_level[time] == \
           model.energy_level[time - 1] + \
           model.charging_efficiency * \
           model.charging_power[time - 1]*\
           (pd.to_timedelta(model.time_increment)/pd.to_timedelta('1h'))


def fixed_energy_level(model, charging_point, time):
    '''
    Constraint for initial value of energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    timeindex = model.timeindex[time]
    cp = '_'.join([str(model.mapping_cp.loc[charging_point, 'ags']),
                   str(model.mapping_cp.loc[charging_point, 'cp_idx']),
                   model.mapping_cp.loc[charging_point, 'use_case']])
    initial_lower_band = \
        model.energy_band_charging_points.loc[timeindex, 'lower_'+cp]
    initial_upper_band = \
        model.energy_band_charging_points.loc[timeindex, 'upper_'+cp]
    return model.energy_level_ev[charging_point, time] == \
           (initial_lower_band+initial_upper_band)/2


def final_energy_level(model, charging_point, time):
    '''
    Constraint for initial value of energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_beginning[charging_point] + model.energy_level_end[charging_point]


def initial_energy_level(model, charging_point, time):
    '''
    Constraint for initial value of energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_start[charging_point] + model.slack_initial_energy_pos[charging_point] - \
           model.slack_initial_energy_neg[charging_point]


def initial_charging_power(model, charging_point, time):
    '''
    Constraint for initial value of charging power
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.charging_ev[charging_point, time] == \
           model.charging_initial[charging_point] + model.slack_initial_charging_pos[charging_point] - \
           model.slack_initial_charging_neg[charging_point]


def final_charging_power(model, charging_point, time):
    '''
    Constraint for final value of charging power, setting it to 0
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.charging_ev[charging_point, time] == 0


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
    timeindex = model.timeindex[time]
    if isinstance(model.v_slack, pd.Series):
        return model.v[bus, time] == np.square(model.v_slack[timeindex] *
                                               model.v_nom)
    else:
        return model.v[bus, time] == np.square(model.v_slack)


def voltage_drop(model, branch, time):
    """
    Constraint that describes the voltage drop over one line
    :param model:
    :param branch:
    :param time:
    :return:
    """
    bus0 = model.branches.loc[branch, 'bus0']
    bus1 = model.branches.loc[branch, 'bus1']
    if model.downstream_nodes_matrix.loc[bus0, bus1] == 1:
        upstream_bus = bus0
        downstream_bus = bus1
    elif model.downstream_nodes_matrix.loc[bus1, bus0] == 1:
        upstream_bus = bus1
        downstream_bus = bus0
    else:
        raise Exception('Something went wrong. Bus0 and bus1 of line {} are '
                        'not connected in downstream_nodes_matrix.'.format(branch))
    return model.v[downstream_bus, time] == model.v[upstream_bus, time] + \
        2 * (model.p_cum[branch, time] * model.branches.loc[branch, model.pars['r']] +
             model.q_cum[branch, time] *  model.branches.loc[branch, model.pars['x']])


def upper_voltage(model, bus, time):
    return model.v[bus, time] <= np.square(model.v_max * model.v_nom) + model.slack_v_pos[bus, time]


def lower_voltage(model, bus, time):
    return model.v[bus, time] >= np.square(model.v_min * model.v_nom) - model.slack_v_neg[bus, time]


def voltage_drop_deprecated(model, branch, time):
    """
    Constraint that describes the voltage drop over one line
    :param model:
    :param branch:
    :param time:
    :return:
    """
    bus0 = model.branches.loc[branch, 'bus0']
    bus1 = model.branches.loc[branch, 'bus1']
    if model.downstream_nodes_matrix.loc[bus0, bus1] == 1:
        upstream_bus = bus0
        downstream_bus = bus1
    elif model.downstream_nodes_matrix.loc[bus1, bus0] == 1:
        upstream_bus = bus1
        downstream_bus = bus0
    else:
        raise Exception('Something went wrong. Bus0 and bus1 of line {} are '
                        'not connected in downstream_nodes_matrix.'.format(branch))
    q_cum = get_q_line(model, branch, time)
    return model.v[downstream_bus, time] == model.v[upstream_bus, time] + \
        2 * (model.p_cum[branch, time] * model.branches.loc[branch, model.pars['r']] +
             q_cum * model.branches.loc[branch, model.pars['x']])


def get_q_line(model, branch, time, get_results=False):
    """
    Method to extract reactive power flow on line.

    :param model:
    :param branch:
    :param time:
    :return:
    """
    timeindex = model.timeindex[time]
    bus0 = model.branches.loc[branch, 'bus0']
    bus1 = model.branches.loc[branch, 'bus1']
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
    tan_phi = np.sqrt(1 - np.square(model.power_factor))
    if get_results:
        return (load_flow_on_line + sum(
            model.curtailment_load[bus, time].value * tan_phi -
            model.curtailment_feedin[bus, time].value * tan_phi for bus in
            relevant_buses))
    else:
        return (load_flow_on_line + sum(
                 model.curtailment_load[bus, time]*tan_phi -
                 model.curtailment_feedin[bus, time]*tan_phi for bus in
                 relevant_buses))


def aggregated_power(model, time):
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.grid_power_flexible[time] == \
           -sum(model.charging[storage, time] for storage in relevant_storage_units)+ \
           sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def check_mapping(mapping_cp, grid, energy_bands):
    """
    Method to make sure the mapping is valid. Checks the existence of the
    indices in the edisgo object and the existence of entries in the e
    nergy_bands dataframe.

    :param mapping_cp:
    :param grid:
    :param energy_bands:
    :return:
    """
    non_existing_cp = mapping_cp.index[~mapping_cp.index.isin(
        grid.charging_points_df.index)]
    if len(non_existing_cp) > 0:
        raise Warning('The following charging points of the mapping are not '
                      'existent in the passed edisgo object: {}.'.format(
            non_existing_cp))
    cp_identifier = ['_'.join(['power', str(mapping_cp.loc[cp, 'ags']),
                               str(mapping_cp.loc[cp, 'cp_idx']),
                               mapping_cp.loc[cp, 'use_case']])
                     for cp in mapping_cp.index]
    non_existing_energy_bands = \
        [identifier for identifier in cp_identifier if
         identifier not in (energy_bands.columns)]
    if len(non_existing_energy_bands):
        raise Warning('The following identifiers do not have respective '
                      'entries inside the energy bands dataframe: {}. This '
                      'might cause problems in the optimization process.'
                      .format(non_existing_energy_bands))


def import_flexibility_bands(dir, grid_id, use_cases):
    flexibility_bands = pd.DataFrame()
    for use_case in use_cases:
        flexibility_bands_tmp = \
            pd.read_csv(dir+'/ev_flexibility_bands_{}_{}.csv'.format(grid_id, use_case),
                        index_col=0, dtype=np.float32)
        rename_dict = {col: col + '_{}'.format(use_case) for col in
                       flexibility_bands_tmp.columns}
        flexibility_bands_tmp.rename(columns=rename_dict, inplace=True)
        flexibility_bands = pd.concat([flexibility_bands, flexibility_bands_tmp],
                                      axis=1)
    # remove numeric problems
    flexibility_bands.loc[:,
    flexibility_bands.columns[flexibility_bands.columns.str.contains('power')]] = \
        (flexibility_bands[flexibility_bands.columns[
            flexibility_bands.columns.str.contains('power')]] + 1e-6).values
    flexibility_bands.loc[:,
    flexibility_bands.columns[flexibility_bands.columns.str.contains('upper')]] = \
        (flexibility_bands[flexibility_bands.columns[
            flexibility_bands.columns.str.contains('upper')]] + 1e-5).values
    flexibility_bands.loc[:,
    flexibility_bands.columns[flexibility_bands.columns.str.contains('lower')]] = \
        (flexibility_bands[flexibility_bands.columns[
            flexibility_bands.columns.str.contains('lower')]] - 1e-6).values
    return flexibility_bands


def get_underlying_elements(model):
    def _get_underlying_elements(downstream_elements, power_factors, model, branch):
        bus0 = model.branches.loc[branch, 'bus0']
        bus1 = model.branches.loc[branch, 'bus1']
        s_nom = model.branches.loc[branch, model.pars['s_nom']]
        relevant_buses_bus0 = \
            model.downstream_nodes_matrix.loc[bus0][
                model.downstream_nodes_matrix.loc[bus0] == 1].index.values
        relevant_buses_bus1 = \
            model.downstream_nodes_matrix.loc[bus1][
                model.downstream_nodes_matrix.loc[bus1] == 1].index.values
        relevant_buses = list(set(relevant_buses_bus0).intersection(
            relevant_buses_bus1))
        downstream_elements.loc[branch, 'buses'] = relevant_buses
        if (model.nodal_reactive_power.loc[relevant_buses].sum().divide(s_nom).apply(abs) > 1).any():
            print('Careful: Reactive power already exceeding line capacity for branch {}.'.format(branch))
        power_factors.loc[branch] = (1-
             model.nodal_reactive_power.loc[relevant_buses].sum().divide(s_nom).apply(np.square)).apply(np.sqrt)
        downstream_elements.loc[branch, 'generators'] = model.grid.generators_df.loc[model.grid.generators_df.bus.isin(
            relevant_buses)].index.values
        downstream_elements.loc[branch, 'loads'] = model.grid.loads_df.loc[model.grid.loads_df.bus.isin(
            relevant_buses)].index.values
        if hasattr(model, 'storage_set'):
            downstream_elements.loc[branch, 'flexible_storage'] = \
                model.grid.storage_units_df.loc[
                    model.grid.storage_units_df.index.isin(
                        model.optimized_storage_set) &
                    model.grid.storage_units_df.bus.isin(relevant_buses)].index.values
        else:
            downstream_elements.loc[branch, 'flexible_storage'] = []
        if hasattr(model, 'charging_points_set'):
            downstream_elements.loc[branch, 'flexible_ev'] = \
                model.grid.charging_points_df.loc[
                    model.grid.charging_points_df.index.isin(
                        model.flexible_charging_points_set) &
                    model.grid.charging_points_df.bus.isin(relevant_buses)].index.values
        else:
            downstream_elements.loc[branch, 'flexible_ev'] = []
        return downstream_elements, power_factors

    downstream_elements = pd.DataFrame(index=model.branches.index,
                                       columns=['buses', 'generators', 'loads', 'flexible_storage', 'flexible_ev', 'pf'])
    power_factors = pd.DataFrame(index=model.branches.index, columns=model.nodal_active_power.columns)
    for branch in downstream_elements.index:
        downstream_elements, power_factors = _get_underlying_elements(downstream_elements, power_factors, model, branch)
    power_factors = power_factors.fillna(0.9)
    #power_factors.where(power_factors >= 0.9, 0.9, inplace=True)
    return downstream_elements, power_factors