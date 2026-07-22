from pandiapy.microstate import Microstate
from pandiapy.topology import Atom, Bond, Angle, Dihedral, Improper
from pandiapy.lammps_io.atom_styles import atom_styles
from pandiapy.periodic_table import PERIODIC_TABLE
from pandiapy.forcefield.pcff import PCFF

## TODO: this is mostly AI coded, so type labels are not working


def write_lmp_datafile(mol: Microstate, path: str, atom_style: str = 'full', comment: str = 'LAMMPS data file') -> None:
    """
    Write a MolSystem to a LAMMPS data file.

    If mol.forcefield is populated, type labels and coefficient sections
    are written. Falls back to deriving type counts from mol if not.

    Parameters
    ----------
    mol        : populated MolSystem object
    path       : output file path
    atom_style : LAMMPS atom style (must be a key in atom_styles.py)
    comment    : header comment line written at the top of the file
    """
    if atom_style not in atom_styles:
        raise ValueError(f"Unknown atom style '{atom_style}'. Choose from: {list(atom_styles)}")

    ff = mol.forcefield if isinstance(mol.forcefield, PCFF) else None

    with open(path, 'w') as f:
        _write_header(f, comment)
        _write_counts(f, mol, ff)
        _write_box(f, mol)
        if ff:
            _write_type_labels(f, ff)
        _write_masses(f, mol, ff)
        if ff:
            _write_forcefield_coeffs(f, ff)
        _write_atoms(f, mol, atom_style)
        if mol.bonds:     _write_bonds(f, mol)
        if mol.angles:    _write_angles(f, mol)
        if mol.dihedrals: _write_dihedrals(f, mol)
        if mol.impropers: _write_impropers(f, mol)



# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------

def _write_header(f, comment: str) -> None:
    f.write(f"{comment}\n\n")


def _write_counts(f, mol: Microstate, ff: PCFF | None) -> None:
    # Use forcefield type counts if available, otherwise derive from mol

    # f.write(f"{len(mol.atoms)}    atoms\n")
    # n_atom_types = len({a.atom_type for a in mol.atoms.values()})
    # f.write(f"{n_atom_types}    atom types\n")
    # f.write("\n")
    #
    # return None

    # TODO: replace derived counts with forcefield object lookups when available
    n_atom_types     = ff.n_atom_types()     if ff else len({a.atom_type for a in mol.atoms.values()})
    n_bond_types     = ff.n_bond_types()     if ff else len({b.bond_type for b in mol.bonds.values()})
    n_angle_types    = ff.n_angle_types()    if ff else len({a.angle_type for a in mol.angles.values()})
    n_dihedral_types = ff.n_dihedral_types() if ff else len({d.dihedral_type for d in mol.dihedrals.values()})
    n_improper_types = ff.n_improper_types() if ff else len({i.improper_type for i in mol.impropers.values()})

    f.write(f"{len(mol.atoms)}    atoms\n")
    if mol.bonds:     f.write(f"{len(mol.bonds)}    bonds\n")
    if mol.angles:    f.write(f"{len(mol.angles)}    angles\n")
    if mol.dihedrals: f.write(f"{len(mol.dihedrals)}    dihedrals\n")
    if mol.impropers: f.write(f"{len(mol.impropers)}    impropers\n")
    f.write("\n")

    f.write(f"{n_atom_types}    atom types\n")
    if mol.bonds:     f.write(f"{n_bond_types}    bond types\n")
    if mol.angles:    f.write(f"{n_angle_types}    angle types\n")
    if mol.dihedrals: f.write(f"{n_dihedral_types}    dihedral types\n")
    if mol.impropers: f.write(f"{n_improper_types}    improper types\n")
    f.write("\n")


def _write_box(f, mol: Microstate) -> None:
    f.write(f"{mol.box.xlo} {mol.box.xhi}    xlo xhi\n")
    f.write(f"{mol.box.ylo} {mol.box.yhi}    ylo yhi\n")
    f.write(f"{mol.box.zlo} {mol.box.zhi}    zlo zhi\n")
    if mol.box.triclinic:
        f.write(f"{mol.box.xy} {mol.box.xz} {mol.box.yz}    xy xz yz\n")
    f.write("\n")


def _write_type_labels(f, ff: PCFF) -> None:
    """Write LAMMPS Type Labels sections for all populated label dicts."""
    label_sections = {
        'Atom Type Labels'     : ff.atom_type_labels,
        'Bond Type Labels'     : ff.bond_type_labels,
        'Angle Type Labels'    : ff.angle_type_labels,
        'Dihedral Type Labels' : ff.dihedral_type_labels,
        'Improper Type Labels' : ff.improper_type_labels,
    }

    for section_name, labels in label_sections.items():
        if not labels:
            continue
        f.write(f"{section_name}\n\n")
        for type_id, label in sorted(labels.items()):
            f.write(f"{type_id}    {label}\n")
        f.write("\n")


def _write_masses(f, mol: Microstate, ff: PCFF | None) -> None:
    """Write one mass entry per unique atom type.

    Uses forcefield masses if available, otherwise looks up via PeriodicTable.
    """
    if ff and ff.masses:
        masses = ff.masses
    else:
        masses = {}
        for atom in mol.atoms.values():
            if atom.atom_type not in masses:
                element = atom.element if isinstance(atom.element, str) else atom.element.symbol
                masses[atom.atom_type] = PERIODIC_TABLE.element2mass(element)


    f.write("Masses\n\n")
    for atom_type, mass in sorted(masses.items()):
        f.write(f"{atom_type}    {mass:>5f}\n")
    f.write("\n")


def _write_atoms(f, mol: Microstate, atom_style: str) -> None:
    """Write the Atoms section using column order defined by atom_style."""
    params = atom_styles[atom_style]
    f.write(f"Atoms  # {atom_style}\n\n")
    for atom in mol.atoms.values():
        row = [_get_atom_field(atom, p) for p in params]
        f.write("    ".join(str(v) for v in row) + "\n")
    f.write("\n")


def _write_bonds(f, mol: Microstate) -> None:
    f.write("Bonds\n\n")
    for bond in mol.bonds.values():
        f.write(f"{bond.id}    {bond.bond_type}    {bond.atom1.id}    {bond.atom2.id}\n")
    f.write("\n")


def _write_angles(f, mol: Microstate) -> None:
    f.write("Angles\n\n")
    for angle in mol.angles.values():
        f.write(f"{angle.id}    {angle.angle_type}    {angle.atom1.id}    {angle.atom2.id}    {angle.atom3.id}\n")
    f.write("\n")


def _write_dihedrals(f, mol: Microstate) -> None:
    f.write("Dihedrals\n\n")
    for dihedral in mol.dihedrals.values():
        f.write(f"{dihedral.id}    {dihedral.dihedral_type}    "
                f"{dihedral.atom1.id}    {dihedral.atom2.id}    "
                f"{dihedral.atom3.id}    {dihedral.atom4.id}\n")
    f.write("\n")


def _write_impropers(f, mol: Microstate) -> None:
    f.write("Impropers\n\n")
    for improper in mol.impropers.values():
        f.write(f"{improper.id}    {improper.improper_type}    "
                f"{improper.atom1.id}    {improper.atom2.id}    "
                f"{improper.atom3.id}    {improper.atom4.id}\n")
    f.write("\n")


def _write_forcefield_coeffs(f, ff: PCFF) -> None:
    """Write all non-empty coefficient sections from the forcefield."""
    for section_name, coeff_dict in ff.all_coeff_sections().items():
        if not coeff_dict:
            continue

        # Determine if any entry has a non-uniform style (hybrid)
        styles = {entry.style for entry in coeff_dict.values()}
        is_hybrid = len(styles) > 1
        section_style = 'hybrid' if is_hybrid else next(iter(styles))

        f.write(f"{section_name}  # {section_style}\n\n")
        for type_id, entry in sorted(coeff_dict.items()):
            coeffs_str = "    ".join(f"{v:>10.5f}\t" for v in entry.coeffs)
            style_prefix = f"{entry.style}    " if is_hybrid else ""
            comment_str  = f"\t # {entry.comment}" if entry.comment else ""
            f.write(f"{type_id}\t {style_prefix}{coeffs_str}{comment_str}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_atom_field(atom: Atom, field: str):
    """Retrieve a field value from an Atom for writing.
    Raises for critical missing fields; defaults optional fields to zero.
    """
    _CRITICAL_FIELDS = {'id', 'atom_type', 'x', 'y', 'z'}
    _OPTIONAL_FIELD_DEFAULT = 0

    value = getattr(atom, field, None)
    if value is None:
        if field in _CRITICAL_FIELDS:
            raise ValueError(f"Atom {atom.id} is missing required field '{field}'")
        return _OPTIONAL_FIELD_DEFAULT
    return value


if __name__ == "__main__":
    from pandiapy.lammps_io.read_lmp_data import read_lmp_datafile

    mol = read_lmp_datafile('../PMMA_20k_r2.data')

    write_lmp_datafile(mol, '../test.data')