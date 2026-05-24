"""
I call it a MolSystem because system is ambiguous, but it is statistically a system

bonds, angles, dihedrals, and impropers will all have their own classes

forcefield will be tied to a MolSystem (I think this is the same as mooonpy)

"""
from dataclasses import dataclass, field
from typing import overload, Optional, Union

from pandiapy.topology import Atom, Bond, Angle, Dihedral, Improper, Box
from pandiapy.periodic_table import Element


@dataclass
class Microstate:
    box: Box = field(default_factory=Box)
    atoms: dict[int, Atom] = field(default_factory=dict)
    bonds: dict[int, Bond] = field(default_factory=dict)
    angles: dict[int, Angle] = field(default_factory=dict)
    dihedrals: dict[int, Dihedral] = field(default_factory=dict)
    impropers: dict[int, Improper] = field(default_factory=dict)

    # counters for auto-assigning new BADI IDs
    _next_atom_id: int = field(default=1, init=False, repr=False)
    _next_bond_id: int = field(default=1, init=False, repr=False)
    _next_angle_id: int = field(default=1, init=False, repr=False)
    _next_dihedral_id: int = field(default=1, init=False, repr=False)
    _next_improper_id: int = field(default=1, init=False, repr=False)
    # forcefield: Union[pcff.PCFF, None] = field(default_factory=None) # wildcard, based on selected FF
    ## TODO: refactor how this handles forcefields

    def _add(self, abadi_dict: dict, obj, counter: str) -> None:
        # Generic helper method to add a new atom,BADI to existing the system
        # "abadi" is atom, bond, angle, dihedral, improper... could use a better name
        if obj.id is None:
            obj.id = getattr(self, counter)
        abadi_dict[obj.id] = obj
        setattr(self, counter, max(getattr(self, counter), obj.id + 1))

    ### ATOMS ###
    # provide multiple options to define an atom
    @overload
    def add_atom(self, atom: Atom) -> None: ...
    @overload
    def add_atom(self, x: float, y: float, z: float,
                 element: Element, atom_type: str) -> None: ...
    def add_atom(self, atom: Atom | None = None, **kwargs) -> None:
        self._add(self.atoms, atom or Atom(**kwargs), '_next_atom_id')

    ### BONDS ###
    # provide multiple ways to define a bond
    @overload
    def add_bond(self, bond: Bond) -> None: ...
    @overload
    def add_bond(self, atom1: Atom, atom2: Atom,
                 bond_type: int = 1, type_name: str | None = None,
                 bond_order: float | None = None) -> None: ...
    def add_bond(self, bond: Bond | None = None, **kwargs) -> None:
        self._add(self.bonds, bond or Bond(**kwargs), '_next_bond_id')

    ### ANGLES ###
    @overload
    def add_angle(self, angle: Angle) -> None: ...
    @overload
    def add_angle(self, atom1: Atom, atom2: Atom, atom3: Atom,
                  angle_type: int, type_name: str) -> None: ...
    def add_angle(self, angle: Angle | None = None, **kwargs) -> None:
        self._add(self.angles, angle or Angle(**kwargs), '_next_angle_id')

    ### DIHEDRALS ###
    @overload
    def add_dihedral(self, dihedral: Dihedral) -> None: ...
    @overload
    def add_dihedral(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom,
                     dihedral_type: int, type_name: str) -> None: ...
    def add_dihedral(self, dihedral: Dihedral | None = None, **kwargs) -> None:
        self._add(self.dihedrals, dihedral or Dihedral(**kwargs), '_next_dihedral_id')

    ### IMPROPERS ###
    @overload
    def add_improper(self, improper: Improper) -> None: ...
    @overload
    def add_improper(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom,
                     improper_type: int, type_name: str) -> None: ...
    def add_improper(self, improper: Improper | None = None, **kwargs) -> None:
        self._add(self.impropers, improper or Improper(**kwargs), '_next_improper_id')

    ### I/O ###
    @classmethod
    def from_lmp_datafile(cls, path: str, startfrom: str = 'Masses') -> 'Microstate':
        from pandiapy.lammps_io.read_lmp_data import read_lmp_datafile
        return read_lmp_datafile(path, startfrom)

    def to_lmp_datafile(self, path: str, atom_style: str = 'full',
                        comment: str = 'PANDIApy-generated LAMMPS data file') -> None:
        from pandiapy.lammps_io.write_lmp_data import write_lmp_datafile
        write_lmp_datafile(self, path, atom_style, comment)

    ### RDKIT COMPATIBILITY ###
    def to_rdkit(self, guess_bond_order=True):
        # read in a mol_system with atoms and bonds, and return a rdkit molecule object
        from rdkit import Chem

        mol = Chem.RWMol()
        conf = Chem.Conformer()

        rdkit_id = {}
        for i, (atomid, atom) in enumerate(self.atoms.items()):
            rdkit_id[atomid] = i
            mol.AddAtom(Chem.Atom(str(atom.element)))
            conf.SetAtomPosition(i, (atom.x, atom.y, atom.z))

        for bond in self.bonds.values():
            if guess_bond_order:
                bond.compute_rdkit_bond_order()
            bond_type = bond.bond_order if isinstance(bond.bond_order, Chem.BondType) else Chem.BondType.SINGLE
            mol.AddBond(rdkit_id[bond.atom1.id], rdkit_id[bond.atom2.id], bond_type)

        mol.AddConformer(conf, assignId=True)
        return mol.GetMol()

    def __str__(self):
        return f"Mol. System with {len(self.atoms)} atoms"

    ## helpers ##
    def _get(self, item: dict, id: int, label: str):
        if id not in item:
            raise KeyError(f"No {label} found with ID '{id}'")
        return item[id]

if __name__ == "__main__":
    mol = Microstate()
    atom = Atom(id=1, x=1.0, y=1.0, z=1.0)
    mol.add_atom(atom)
    # mol.update_atom(1, q=2)
    print(atom)
    print(mol)