from edisgo.edisgo import import_edisgo_from_files
import pandas as pd
from results_helper_functions import relative_load, voltage_diff
from edisgo.tools.tools import extract_feeders_nx

grid_id = 177
edisgo_dir = r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\optimised'.format(grid_id)

edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)

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


edisgo_obj.analyze()
edisgo_obj.save(r'U:\Software\curtailment\results\177')
# feeders = extract_feeders_nx(edisgo_obj)
# feeders[0].save(r'U:\Software\curtailment\results\177')

rel_load = relative_load(edisgo_obj)
v_diff = voltage_diff(edisgo_obj)
#
# rel_load.to_csv(r'U:\Software\curtailment\results\rel_load.csv')
# v_diff.to_csv(r'U:\Software\curtailment\results\v_diff.csv')

print('SUCCESS')
