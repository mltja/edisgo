from edisgo.edisgo import import_edisgo_from_files
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
    rename_dict = {line[end_bus]:line[neighbor_bus]}
    for Type, components in connected_elements.items():
        if not components.empty and Type != 'Line':
            if Type == 'Switch':
                edisgo.topology.switches_df = edisgo.topology.switches_df.replace(rename_dict)
            else:
                setattr(edisgo.topology, Type.lower() + 's_df',
                        getattr(edisgo.topology, Type.lower() + 's_df').replace(
                            rename_dict))
    # remove line
    edisgo.topology.remove_line(line.name)
    print('{} removed.'.format(line.name))


# For removal on 1m lines
def remove_1m_lines_from_edisgo(grid_id):
    try:
        print('Removing 1m lines for grid {}'.format(grid_id))
        edisgo = import_edisgo_from_files(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced'.format(grid_id))

        lines = edisgo.topology.lines_df.loc[edisgo.topology.lines_df.length == 0.001]
        for name, line in lines.iterrows():
            remove_1m_end_line(edisgo, line)
        edisgo.topology.to_csv(r'U:\Software\eDisGo_object_files\simbev_nep_2035_results\{}\reduced\topology'.format(grid_id))
    except Exception as e:
        print('Problem in grid {}.'.format(grid_id))
        print(e)


if __name__ == '__main__':
    pool = mp.Pool(3)

    grid_ids = [1056, 1811, 2534]
    pool.map(remove_1m_lines_from_edisgo, grid_ids)
    pool.close()
    print('SUCCESS')