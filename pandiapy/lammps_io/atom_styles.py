# full list in LAMMPS here: https://docs.lammps.org/read_data.html
full = ['id', 'molecule', 'atom_type', 'q', 'x', 'y', 'z']
charge = ['id', 'atom_type', 'q', 'x', 'y', 'z']
atomic = ['id', 'atom_type', 'x', 'y', 'z']
angle = ['id', 'molecule', 'atom_type', 'x', 'y', 'z']
molecular = ['id', 'molecule', 'atom_type', 'x', 'y', 'z'] # same as angle


# dictionary for type assignments in read data
atom_styles = {'full'      : full,
               'charge'    : charge,
               'atomic'    : atomic,
               'angle'     : angle,
               'molecular' : molecular}