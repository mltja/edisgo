from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.components import Switch
import multiprocessing as mp


def remove_1m_end_line(edisgo, line):
    # Check for end buses
    if len(edisgo.topology.get_connected_lines_from_bus(line.bus1))==1:
        end_bus = 'bus1'
        neighbor_bus = 'bus0'
    elif len(edisgo.topology.get_connected_lines_from_bus(line.bus0))==1:
        end_bus = 'bus0'
        neighbor_bus = 'bus1'
    else:
        end_bus = None
        neighbor_bus = None
        print('No end bus found. Implement method.')
        return
    # Move connected elements of end bus to the other bus
    connected_elements = edisgo.topology.get_connected_components_from_bus(line[end_bus])
    # move elements to neighboring bus
    rename_dict = {line[end_bus]: line[neighbor_bus]}
    for Type, components in connected_elements.items():
        if not components.empty and Type != 'lines':
            setattr(edisgo.topology, Type.lower() + '_df',
                        getattr(edisgo.topology, Type.lower() + '_df').replace(
                            rename_dict))
    # remove line
    edisgo.topology.remove_line(line.name)
    print('{} removed.'.format(line.name))


# For removal on 1m lines
def remove_1m_lines_from_edisgo(grid_id):
    # try:
    print('Removing 1m lines for grid {}'.format(grid_id))
    edisgo = import_edisgo_from_files(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id))

    # close switches such that lines with connected switches are not removed
    switches = [Switch(id=_, topology=edisgo.topology)
                for _ in edisgo.topology.switches_df.index]
    switch_status = {}
    for switch in switches:
        switch_status[switch] = switch.state
        switch.close()
    # get all lines and remove end 1m lines
    lines = edisgo.topology.lines_df.loc[edisgo.topology.lines_df.length == 0.001]
    for name, line in lines.iterrows():
        remove_1m_end_line(edisgo, line)
    # set switches back to original state
    for switch in switches:
        if switch_status[switch] == 'open':
            switch.open()
    edisgo.topology.to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced\topology'.format(grid_id))
    # except Exception as e:
    #     print('Problem in grid {}.'.format(grid_id))
    #     print(e)


if __name__ == '__main__':
    # pool = mp.Pool(3)
    #
    # grid_ids = [1056, 1811, 2534]
    # pool.map(remove_1m_lines_from_edisgo, grid_ids)
    # pool.close()
    remove_1m_lines_from_edisgo(2534)
    print('SUCCESS')