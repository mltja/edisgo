from edisgo.edisgo import import_edisgo_from_files
import pandas as pd
from results_helper_functions import relative_load, voltage_diff
from edisgo.tools.tools import extract_feeders_nx
from run_extract_optimized_edisgo_object import combine_results_for_grid

grid_id = 176
feeder_id = 6
edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\feeder\{}'.format(grid_id, feeder_id)


def add_curtailment_linopt(edisgo_dir, feeders, res_dir):
    edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    curtailment_ev = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_ev")
    curtailment_load = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_load")
    curtailment_feedin = combine_results_for_grid(feeders, grid_id, res_dir, "curtailment_feedin")

    #Todo: calculate reactive power

    curtailment_load = pd.read_csv(edisgo_dir+r'\curtailment_load_optimised.csv', index_col=0, parse_dates=True)
    curtailment_feedin = pd.read_csv(edisgo_dir+r'\curtailment_feedin_optimised.csv', index_col=0, parse_dates=True)
    curtailment_reactive_load = pd.read_csv(edisgo_dir+r'\curtailment_load_reactive_optimised.csv', index_col=0, parse_dates=True)
    curtailment_reactive_feedin = pd.read_csv(edisgo_dir+r'\curtailment_feedin_reactive_optimised.csv', index_col=0, parse_dates=True)

    edisgo_obj.timeseries.mode = 'manual'
    for node in curtailment_load:
        if curtailment_load[node].sum().sum()>0:
            edisgo_obj.add_component('Generator', ts_active_power=curtailment_load[node],
                                     ts_reactive_power=curtailment_reactive_load[node], bus=node,
                                     generator_id=node, p_nom=curtailment_load[node].max(),
                                     generator_type='load_curtailment')
            print('Generator added for curtailment at bus {}'.format(node))
        if curtailment_feedin[node].sum().sum()>0:
            edisgo_obj.add_component('Load', ts_active_power=curtailment_feedin[node],
                                     ts_reactive_power=curtailment_reactive_feedin[node], bus=node,
                                     load_id=node, peak_load=curtailment_feedin[node].max(),
                                     annual_consumption=curtailment_feedin[node].sum(),
                                     sector='feedin_curtailment')
            print('Load added for curtailment at bus {}'.format(node))
    return edisgo_obj


def add_curtailment_julia(edisgo_dir, curtailment_dir, factor=10):
    edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

    timeindex = edisgo_obj.timeseries.timeindex[672:]
    edisgo_obj.timeseries._timeindex = timeindex
    curtailment_load = pd.read_csv(curtailment_dir + r'\curtailment_load.csv').set_index(timeindex)
    curtailment_feedin = pd.read_csv(curtailment_dir + r'\curtailment_feedin.csv').set_index(timeindex)
    curtailment_reactive_load = pd.read_csv(curtailment_dir + r'\curtailment_reactive_load.csv').set_index(timeindex)
    curtailment_reactive_feedin = pd.read_csv(curtailment_dir + r'\curtailment_reactive_feedin.csv').set_index(timeindex)

    edisgo_obj.timeseries.mode = 'manual'
    for load in curtailment_load:
        name_load = "_".join(load.split("_")[1:])
        node = edisgo_obj.topology.loads_df.loc[name_load, "bus"]
        if curtailment_load[load].sum().sum() > 0:
            edisgo_obj.add_component('Generator', ts_active_power=curtailment_load[load]*factor,
                                     ts_reactive_power=curtailment_reactive_load["Q"+load[1:]]*factor, bus=node,
                                     generator_id=name_load, p_nom=curtailment_load[load].max()*factor,
                                     generator_type='load_curtailment')
            print('Generator added for curtailment of load {}'.format(name_load))
    for gen in curtailment_feedin:
        name_gen = "_".join(gen.split("_")[1:])
        node = edisgo_obj.topology.generators_df.loc[name_gen, "bus"]
        if curtailment_feedin[gen].sum().sum() > 0:
            edisgo_obj.add_component('Load', ts_active_power=curtailment_feedin[gen]*factor,
                                     ts_reactive_power=curtailment_reactive_feedin["Q"+gen[1:]]*factor, bus=node,
                                     load_id=name_gen, peak_load=curtailment_feedin[gen].max()*factor,
                                     annual_consumption=curtailment_feedin[gen].sum()*factor,
                                     sector='feedin_curtailment')
            print('Load added for curtailment of generator {}'.format(name_gen))

    edisgo_obj.timeseries.residual_load.plot()
    return edisgo_obj


if __name__ == "__main__":
    edisgo_obj = add_curtailment_julia(
        edisgo_dir,
        r'\\s4d-fs.d.ethz.ch\itet-home$\aheider\Desktop\Code Conor\Networks\Feeder6\results')
    edisgo_obj.analyze(timesteps=edisgo_obj.timeseries.timeindex[10:15])
    rel_load = relative_load(edisgo_obj)

# edisgo_obj.analyze()
# edisgo_obj.save(r'U:\Software\curtailment\results\177')
# edisgo_obj = import_edisgo_from_files(r'U:\Software\curtailment\results\177', import_timeseries=True)
# edisgo_obj.analyze()
# # feeders = extract_feeders_nx(edisgo_obj)
# # feeders[0].save(r'U:\Software\curtailment\results\177')
#
# rel_load = relative_load(edisgo_obj)
# v_diff = voltage_diff(edisgo_obj)
# #
# rel_load.to_csv(r'U:\Software\curtailment\results\rel_load.csv')
# v_diff.to_csv(r'U:\Software\curtailment\results\v_diff.csv')

print('SUCCESS')
