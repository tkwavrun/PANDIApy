from pandiapy.microstate import Microstate
from pandiapy.topology import Atom, Bond, Angle, Dihedral, Improper
from pandiapy.periodic_table import PERIODIC_TABLE
from pandiapy.lammps_io.atom_styles import atom_styles

from pandiapy.math import string2digit

def read_lmp_datafile(path: str, startfrom: str = 'Masses') -> Microstate:
    file_contents = {}
    mol = Microstate()

    with open(path, "r") as f:
        f.readline()  # skip header line

        _parse_box_dimensions(f, mol)
        _parse_sections_by_header(f, file_contents, startfrom)

    _populate_atoms(mol, file_contents)
    _populate_bonds(mol, file_contents)
    _populate_angles(mol, file_contents)
    _populate_dihedrals(mol, file_contents)
    _populate_impropers(mol, file_contents)
    # _populate_forcefield(mol, file_contents)
    return mol


## UTIL FUNCTIONS
def _strip_comments(line: str) -> tuple[str, str]:
    if len(line.split("#")) > 1:
        return line.split("#")[0].strip(), line.split("#")[1].strip()
    else:
        return line.split("#")[0].strip(), ""


def _parse_box_dimensions(f, mol: Microstate) -> None:
    for line in f:
        if 'xlo' not in line or 'xhi' not in line:
            continue

        xlo, xhi = map(string2digit, line.split()[:2])
        ylo, yhi = map(string2digit, f.readline().split()[:2])
        zlo, zhi = map(string2digit, f.readline().split()[:2])

        mol.box.xlo, mol.box.xhi = xlo, xhi
        mol.box.ylo, mol.box.yhi = ylo, yhi
        mol.box.zlo, mol.box.zhi = zlo, zhi

        triclinic_params = f.readline().split()
        if triclinic_params:
            mol.box.xy, mol.box.xz, mol.box.yz = map(string2digit, triclinic_params[:3])
            mol.box.triclinic = True
        else:
            mol.box.triclinic = False

        break


# POPULATORS
def _parse_sections_by_header(f, file_contents: dict, startfrom: str) -> None:
    for line in f:
        header, comment = _strip_comments(line)
        if header == startfrom:
            file_contents[header] = {"comment": comment}
            break

    # Read datafile sections that contain headers, add them to the dictionary
    reading_header = True
    for line in f:
        if not line.strip():
            reading_header = not reading_header
        elif reading_header:
            header, comment = _strip_comments(line)
            file_contents[header] = {"comment": comment}
        else:
            params = line.split()  # all items in line, seperated by whitespace
            property_id = int(params[0])  # atom ID, bond_type ID, etc
            file_contents[header][property_id] = params[1:]

def _populate_atoms(mol: Microstate, file_contents: dict) -> None:
    # map each atom type ID to its element using mass lookup
    type_to_element = {
        type_id: PERIODIC_TABLE.mass2element(float(data[0]))
        for type_id, data in file_contents['Masses'].items()
        if isinstance(type_id, int)
    }

    # Determine which attributes each atom row contains based on the LAMMPS atom style (i.e. full, charge)
    atom_style = file_contents['Atoms']['comment']
    atom_params = atom_styles[atom_style]

    # Populate each atom using the column names defined by the atom style
    for atom_id, values in file_contents['Atoms'].items():
        if not isinstance(atom_id, int):
            continue
        atom = Atom(**{param: string2digit(v) for param, v in zip(atom_params, [atom_id, *values])})
        atom.element = type_to_element[atom.atom_type]
        mol.add_atom(atom)

def _populate_bonds(mol: Microstate, file_contents: dict) -> None:
    for bond_id, values in file_contents['Bonds'].items():
        if not isinstance(bond_id, int):
            continue
        bond_type, *atom_ids = values
        bond = Bond(
            id=int(bond_id),
            bond_type=int(bond_type),
            atom1=mol.atoms[int(atom_ids[0])],
            atom2=mol.atoms[int(atom_ids[1])]
        )
        mol.add_bond(bond)

def _populate_angles(mol: Microstate, file_contents: dict) -> None:
    for angle_id, values in file_contents['Angles'].items():
        if not isinstance(angle_id, int):
            continue
        angle_type, *atom_ids = values
        mol.add_angle(Angle(
            id=int(angle_id),
            angle_type=int(angle_type),
            atom1=mol.atoms[int(atom_ids[0])],
            atom2=mol.atoms[int(atom_ids[1])],
            atom3=mol.atoms[int(atom_ids[2])]
        ))


def _populate_dihedrals(mol: Microstate, file_contents: dict) -> None:
    for dihedral_id, values in file_contents['Dihedrals'].items():
        if not isinstance(dihedral_id, int):
            continue
        dihedral_type, *atom_ids = values
        mol.add_dihedral(Dihedral(
            id=int(dihedral_id),
            dihedral_type=int(dihedral_type),
            atom1=mol.atoms[int(atom_ids[0])],
            atom2=mol.atoms[int(atom_ids[1])],
            atom3=mol.atoms[int(atom_ids[2])],
            atom4=mol.atoms[int(atom_ids[3])]
        ))


def _populate_impropers(mol: Microstate, file_contents: dict) -> None:
    for improper_id, values in file_contents['Impropers'].items():
        if not isinstance(improper_id, int):
            continue
        improper_type, *atom_ids = values
        mol.add_improper(Improper(
            id=int(improper_id),
            improper_type=int(improper_type),
            atom1=mol.atoms[int(atom_ids[0])],
            atom2=mol.atoms[int(atom_ids[1])],
            atom3=mol.atoms[int(atom_ids[2])],
            atom4=mol.atoms[int(atom_ids[3])]
        ))

def _populate_forcefield(mol: Microstate, file_contents: dict) -> None:
    from pandiapy.forcefield.pcff import PCFF, CoeffList
    from pandiapy.forcefield._headers import _LABEL_SECTIONS, _COEFF_SECTIONS

    ff_sections = (set(_COEFF_SECTIONS) | set(_LABEL_SECTIONS)) & file_contents.keys()
    if not ff_sections:
        mol.forcefield = None
        return

    ff = PCFF()

    for type_id, data in file_contents['Masses'].items():
        if not isinstance(type_id, int):
            continue
        ff.masses[type_id] = float(data[0])

    for section_name, ff_attr in _LABEL_SECTIONS.items():
        if section_name not in file_contents:
            continue
        label_dict = getattr(ff, ff_attr)
        for type_id, data in file_contents[section_name].items():
            if not isinstance(type_id, int):
                continue
            label_dict[type_id] = data[0]

    for section_name, ff_attr in _COEFF_SECTIONS.items():
        if section_name not in file_contents:
            continue
        section = file_contents[section_name]
        section_style = section.get('comment', '')
        coeff_dict = getattr(ff, ff_attr)
        is_hybrid = section_style == 'hybrid'

        for type_id, data in section.items():
            if not isinstance(type_id, int):
                continue
            tokens_str, comment = _strip_comments(' '.join(data))
            tokens = tokens_str.split()
            if is_hybrid:
                style, coeffs = tokens[0], tuple(float(v) for v in tokens[1:])
            else:
                style, coeffs = section_style, tuple(float(v) for v in tokens)
            coeff_dict[type_id] = CoeffList(coeffs=coeffs, style=style, comment=comment)

    mol.forcefield = ff


if __name__ == "__main__":
    mol = read_lmp_datafile('../PMMA_20k_r2.data')
    print(mol)