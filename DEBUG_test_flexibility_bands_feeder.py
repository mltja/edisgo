# Test to see if optimisation problem with flexibility bands is feasible
import numpy as np
import pandas as pd
import pyomo.environ as pm
import geopandas as gpd
import edisgo.flex_opt.optimization as opt


def objective(model):
    return sum([np.square(sum([model.charging_ev[cp, t] for cp in model.flexible_charging_points_set]))
               for t in model.time_set])


# Setting variable parameters
grid_id = 1690
feeder_id = 0
stop_iter = 2
root_dir = r'U:\Software'
save_dir ='results/tests'
mapping_dir = root_dir + r'\simbev_nep_2035_results\eDisGo_charging_time_series\{}'.format(grid_id)
edisgo_dir = root_dir + r'\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)
use_cases = ['home', 'work']
solver = 'gurobi'

charging_points = pd.read_csv(edisgo_dir+r'\topology\charging_points.csv', index_col=0)

print('Evaluating grid {}, feeder {}.'.format(grid_id, feeder_id))
flexibility_bands = opt.import_flexibility_bands('grid_data', grid_id, use_cases)

mapping_home = \
        gpd.read_file(mapping_dir + '/cp_data_home_within_grid_{}.geojson'.
                      format(grid_id)).set_index('edisgo_id')
mapping_work = \
    gpd.read_file(mapping_dir + '/cp_data_work_within_grid_{}.geojson'.
                  format(grid_id)).set_index('edisgo_id')
mapping_home['use_case'] = 'home'
mapping_work['use_case'] = 'work'
mapping = pd.concat([mapping_work, mapping_home],
                    sort=False)  # , mapping_hpc, mapping_public
print('Mapping imported.')

# extract data for feeder
mapping = mapping.loc[mapping.index.isin(charging_points.index)]
cp_identifier = ['_'.join([str(mapping.loc[cp, 'ags']),
                           str(mapping.loc[cp, 'cp_idx']),
                           mapping.loc[cp, 'use_case']])
                 for cp in mapping.index]
flex_band_identifier = []
for cp in cp_identifier:
    flex_band_identifier.append('lower_'+cp)
    flex_band_identifier.append('upper_'+cp)
    flex_band_identifier.append('power_'+cp)
flexibility_bands = flexibility_bands[flex_band_identifier]
#flexibility_bands = flexibility_bands.iloc[0:10]
charging_points = list(set(['_'.join(cp[1:]) for cp in flexibility_bands.columns.str.split('_')]))

rename_dict = {cp[0]: '{}_{}_{}'.format(cp[1].ags, cp[1].cp_idx, cp[1].use_case) for cp in mapping.iterrows()}

timesteps_per_iteration = 24*4
iterations_per_era = 7
charging_start = pd.read_csv('results/tests/charging_start_{}_{}_{}.csv'.format(grid_id, feeder_id, stop_iter),
                             index_col=0, header=None).rename(rename_dict)[1]
energy_level_start = pd.read_csv('results/tests/energy_level_start_{}_{}_{}.csv'.format(grid_id, feeder_id, stop_iter),
                                 index_col=0, header=None).rename(rename_dict)[1]
overlap_interations = 48
energy_level = {}
charging_ev = {}
timeindex = pd.date_range(start='2029-01-01 7:00', periods=len(flexibility_bands), freq='15min')
for iteration in range(stop_iter,
        int(len(timeindex) / timesteps_per_iteration)):
    print('Starting optimisation for iteration {}.'.format(iteration))
    start_time = (iteration * timesteps_per_iteration) % 672
    if iteration % iterations_per_era != iterations_per_era - 1:
        timesteps = timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration + overlap_interations]
        flexibility_bands_week = flexibility_bands.iloc[
                                 start_time:start_time + timesteps_per_iteration + overlap_interations].set_index(
            timesteps)
    else:
        timesteps = timeindex[
                    iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
        flexibility_bands_week = flexibility_bands.iloc[start_time:start_time + timesteps_per_iteration].set_index(
            timesteps)
    for cp_tmp in energy_level_start.index:
        if energy_level_start[cp_tmp] > flexibility_bands_week.iloc[0]['upper_' + cp_tmp]:
            print('charging point {} violates upper bound.'.format(cp_tmp))
        if energy_level_start[cp_tmp] < flexibility_bands_week.iloc[0]['lower_' + cp_tmp]:
            print('charging point {} violates lower bound.'.format(cp_tmp))
    # Create model and add fix parameters
    print('Setting up model.')
    model = pm.ConcreteModel()
    model.timeindex = timesteps
    model.time_increment = pd.infer_freq(model.timeindex)
    model.charging_efficiency = 0.9
    model.energy_band_charging_points = flexibility_bands_week
    # Setting sets
    print('Setup model: Defining sets.')
    model.flexible_charging_points_set = pm.Set(initialize=charging_points)
    model.time_set = pm.RangeSet(0, len(model.timeindex)-1)
    model.time_zero = [model.time_set[1]]
    model.time_end = [model.time_set[-1]]
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
    model.charging_initial = charging_start
    if charging_start is not None:
        model.InitialEVChargingPower = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                          rule=opt.initial_charging_power)
    # Setting constraints
    'Setup model: Defining constraints.'
    model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                             model.time_non_zero, rule=opt.charging_ev)
    model.energy_level_start = energy_level_start
    if model.energy_level_start is not None:
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                          rule=opt.initial_energy_level)
    else:
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                          rule=opt.fixed_energy_level)

    model.FinalEVEnergyLevel = \
        pm.Constraint(model.flexible_charging_points_set, model.time_end,
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
            timeindex_tmp = model.timeindex[time]
            for cp in model.flexible_charging_points_set:
                x_charge_ev.loc[timeindex_tmp, cp] = model.charging_ev[cp, time].value
                energy_level_cp.loc[timeindex_tmp, cp] = model.energy_level_ev[cp, time].value
        x_charge_ev.to_csv(save_dir + '/x_charge_ev_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
        energy_level_cp.to_csv(save_dir + '/energy_level_ev_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
        charging_ev[iteration] = x_charge_ev
        energy_level[iteration] = energy_level_cp
        if iteration % iterations_per_era != iterations_per_era - 1:
            charging_start = charging_ev[iteration].iloc[-overlap_interations]
            energy_level_start = energy_level[iteration].iloc[-overlap_interations]
        else:
            charging_start = None
            energy_level_start = None
    except Exception as inst:
        print('Problem with grid: {}, feeder {}'.format(grid_id, feeder_id))
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,


print('SUCCESS')