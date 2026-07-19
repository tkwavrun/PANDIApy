from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pandiapy.forcefield._coeff import CoeffList


@dataclass
class ForceField(ABC):
    """
    Abstract base class for all force fields in PANDIApy.

    Holds atom-level data (masses and type label maps) that every concrete
    force field needs, regardless of whether it has discrete bond types.

    Attributes
    ----------
    name             : human-readable identifier for this FF instance
    source_file      : path to the parameter file used to build this instance
    masses           : {atom_type_id: mass}
    atom_type_labels : {atom_type_id: label_string}  — forward map
    atom_type_ids    : {label_string: atom_type_id}  — inverse map
    """
    name:              str
    source_file:       str | None
    masses:            dict[int, float]   = field(default_factory=dict)
    atom_type_labels:  dict[int, str]     = field(default_factory=dict)
    atom_type_ids:     dict[str, int]     = field(default_factory=dict)

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of the parameterisation."""
        ...


@dataclass
class FixedBondForceField(ForceField, ABC):
    """
    Base class for force fields with discrete bond types (PCFF, DREIDING, OPLS, …).

    Adds per-type coefficient dicts for bonded and non-bonded interactions, and
    forward/inverse type-label maps for bonds, angles, dihedrals, and impropers.

    All coefficient dicts are keyed by integer type ID (1-based, matching the
    LAMMPS data file convention).  Type label maps follow the convention:

        *_type_labels : {type_id: '-'.join(atom_type_names)}
        *_type_ids    : {tuple(atom_type_names): type_id}
    """

    # ------------------------------------------------------------------
    # Primary interaction coefficients
    # ------------------------------------------------------------------
    pair_coeffs:     dict[int, CoeffList] = field(default_factory=dict)
    bond_coeffs:     dict[int, CoeffList] = field(default_factory=dict)
    angle_coeffs:    dict[int, CoeffList] = field(default_factory=dict)
    dihedral_coeffs: dict[int, CoeffList] = field(default_factory=dict)
    improper_coeffs: dict[int, CoeffList] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Type label maps — forward: {type_id: label} and
    #                   inverse: {label_tuple: type_id}
    # ------------------------------------------------------------------
    bond_type_labels:     dict[int, str]    = field(default_factory=dict)
    angle_type_labels:    dict[int, str]    = field(default_factory=dict)
    dihedral_type_labels: dict[int, str]    = field(default_factory=dict)
    improper_type_labels: dict[int, str]    = field(default_factory=dict)

    bond_type_ids:        dict[tuple, int]  = field(default_factory=dict)
    angle_type_ids:       dict[tuple, int]  = field(default_factory=dict)
    dihedral_type_ids:    dict[tuple, int]  = field(default_factory=dict)
    improper_type_ids:    dict[tuple, int]  = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience counters
    # ------------------------------------------------------------------
    def n_atom_types(self)     -> int: return len(self.masses)
    def n_bond_types(self)     -> int: return len(self.bond_coeffs)
    def n_angle_types(self)    -> int: return len(self.angle_coeffs)
    def n_dihedral_types(self) -> int: return len(self.dihedral_coeffs)
    def n_improper_types(self) -> int: return len(self.improper_coeffs)


@dataclass
class ReactiveForceField(ForceField, ABC):
    """
    Base class for reactive force fields (ReaxFF, AIREBO, …).

    Reactive FFs do not have discrete bond types; bond order is computed
    dynamically.  Subclasses define their own parameter structure.
    """
    pass
