# Test to see if optimisation problem with flexibility bands is feasible
import numpy as np
import pandas as pd
import pyomo.environ as pm
import edisgo.flex_opt.optimization as opt


def objective(model):
    return sum([np.square(sum([model.charging_ev[cp, t] for cp in model.flexible_charging_points_set]))
               for t in model.time_set])


# Setting variable parameters
root_dir = r'U:\Software'
save_dir ='results/tests'
grid_ids = [1690, 1811, 2534, 176, 177]
use_cases = ['home', 'work']
solver = 'gurobi'

for grid_id in grid_ids:
    print('Evaluating grid {}'.format(grid_id))
    flexibility_bands = opt.import_flexibility_bands('grid_data', grid_id, use_cases)
    #flexibility_bands = flexibility_bands.iloc[0:10]
    charging_points = list(set(['_'.join(cp[1:]) for cp in flexibility_bands.columns.str.split('_')]))

    # Create model and add fix parameters
    print('Setting up model.')
    model = pm.ConcreteModel()
    model.timeindex = pd.date_range(start='2029-01-01 7:00', periods=len(flexibility_bands), freq='15min')
    model.time_increment = pd.infer_freq(model.timeindex)
    model.charging_efficiency = 0.9
    model.energy_band_charging_points = flexibility_bands.set_index(model.timeindex)
    # Setting sets
    print('Setup model: Defining sets.')
    model.flexible_charging_points_set = pm.Set(initialize=charging_points)
    model.time_set = pm.RangeSet(0, len(model.timeindex)-1)
    model.time_zero = [model.time_set[1]] + [model.time_set[-1]]
    model.time_non_zero = model.time_set - [model.time_set[1]]
    # Defining parameters
    print('Setup model: Defining parameters.')
    model.energy_level_ev = pm.Var(model.flexible_charging_points_set, model.time_set,
                                   bounds=lambda m, b, t:
                                   (m.energy_band_charging_points.loc[model.timeindex[t], '_'.join(['lower', b])],
                                    m.energy_band_charging_points.loc[model.timeindex[t], '_'.join(['upper', b])]))
    model.charging_ev = \
                pm.Var(model.flexible_charging_points_set, model.time_set,
                       bounds=lambda m, b, t:
                       (0, m.energy_band_charging_points.loc[
                           model.timeindex[t], '_'.join(['power', b])]))
    # Setting constraints
    'Setup model: Defining constraints.'
    model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                             model.time_non_zero, rule=opt.charging_ev)
    model.InitialEVEnergyLevel = \
                pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                              rule=opt.fixed_energy_level)
    # Adding objective
    'Setup model: Defining objective.'
    model.objective = pm.Objective(rule=objective, sense=pm.minimize,
                                           doc='Define objective function')
    print('Successfully set up optimisation model.')
    #model.pprint()
    # Optimize
    slv = pm.SolverFactory(solver)
    try:
        results = slv.solve(model, tee=True)
        # Load solutions
        x_charge_ev = pd.DataFrame()
        energy_level_cp = pd.DataFrame()
        for time in model.time_set:
            timeindex = model.timeindex[time]
            for cp in model.flexible_charging_points_set:
                x_charge_ev.loc[timeindex, cp] = model.charging_ev[cp, time].value
                energy_level_cp.loc[timeindex, cp] = model.energy_level_ev[cp, time].value
        x_charge_ev.to_csv(save_dir + '/x_charge_ev_{}.csv'.format(grid_id))
        energy_level_cp.to_csv(save_dir + '/energy_level_ev_{}.csv'.format(grid_id))
    except Exception as inst:
        print('Problem with grid: {}'.format(grid_id))
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,


print('SUCCESS')