from __future__ import annotations
from dataclasses import dataclass, field

from pandiapy.forcefield._coeff import CoeffList
from pandiapy.forcefield._base import FixedBondForceField


@dataclass
class PCFF(FixedBondForceField):
    """
    PCFF (Polymer Consistent Force Field) -- class 2 force field.

    Inherits primary coefficient dicts and type maps from FixedBondForceField.
    Adds the eight PCFF cross-term sections and OOP/angle-angle split tracking.

    Cross term sections
    -------------------
    bond_bond_coeffs           : bond-bond coupling within an angle
    bond_angle_coeffs          : bond-angle coupling
    angle_angle_coeffs         : angle-angle coupling within an improper
    angle_torsion_coeffs       : angle-torsion coupling
    end_bond_torsion_coeffs    : end bond-torsion coupling
    middle_bond_torsion_coeffs : middle bond-torsion coupling
    bond_bond_13_coeffs        : 1-3 bond-bond coupling within a dihedral
    angle_angle_torsion_coeffs : angle-angle-torsion coupling

    flagged_angleangles
    -------------------
    List of improper type IDs that are angle-angle (not Wilson OOP).
    Populated by BADIEnumerator based on the nb==3 graph condition.
    """

    # Defaults for inherited abstract fields so that PCFF() still works
    # without positional arguments (backward-compatible with existing callers).
    name:        str       = field(default="PCFF")
    source_file: str|None  = field(default=None)

    # ------------------------------------------------------------------
    # PCFF class-2 cross-term coefficients
    # ------------------------------------------------------------------
    bond_bond_coeffs:           dict[int, CoeffList] = field(default_factory=dict)
    bond_angle_coeffs:          dict[int, CoeffList] = field(default_factory=dict)
    angle_angle_coeffs:         dict[int, CoeffList] = field(default_factory=dict)
    angle_torsion_coeffs:       dict[int, CoeffList] = field(default_factory=dict)
    end_bond_torsion_coeffs:    dict[int, CoeffList] = field(default_factory=dict)
    middle_bond_torsion_coeffs: dict[int, CoeffList] = field(default_factory=dict)
    bond_bond_13_coeffs:        dict[int, CoeffList] = field(default_factory=dict)
    angle_angle_torsion_coeffs: dict[int, CoeffList] = field(default_factory=dict)

    # Improper type IDs that are angle-angle (nb==3 center atoms)
    flagged_angleangles: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def all_coeff_sections(self) -> dict[str, dict[int, CoeffList]]:
        """Return all coefficient dicts as a named mapping for the file writer."""
        return {
            'Pair Coeffs'             : self.pair_coeffs,
            'Bond Coeffs'             : self.bond_coeffs,
            'Angle Coeffs'            : self.angle_coeffs,
            'Dihedral Coeffs'         : self.dihedral_coeffs,
            'Improper Coeffs'         : self.improper_coeffs,
            'BondBond Coeffs'         : self.bond_bond_coeffs,
            'BondAngle Coeffs'        : self.bond_angle_coeffs,
            'AngleAngle Coeffs'       : self.angle_angle_coeffs,
            'AngleTorsion Coeffs'     : self.angle_torsion_coeffs,
            'EndBondTorsion Coeffs'   : self.end_bond_torsion_coeffs,
            'MiddleBondTorsion Coeffs': self.middle_bond_torsion_coeffs,
            'BondBond13 Coeffs'       : self.bond_bond_13_coeffs,
            'AngleAngleTorsion Coeffs': self.angle_angle_torsion_coeffs,
        }

    def summary(self) -> str:
        lines = [f"PCFF ForceField: {self.name}"]
        if self.source_file:
            lines.append(f"  Source file   : {self.source_file}")
        lines.append(f"  Atom types    : {self.n_atom_types()}")
        lines.append(f"  Bond types    : {self.n_bond_types()}")
        lines.append(f"  Angle types   : {self.n_angle_types()}")
        lines.append(f"  Dihedral types: {self.n_dihedral_types()}")
        lines.append(f"  Improper types: {self.n_improper_types()}")
        cross_terms = {k: v for k, v in self.all_coeff_sections().items()
                       if k not in {'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs',
                                    'Dihedral Coeffs', 'Improper Coeffs'} and v}
        if cross_terms:
            lines.append("  Cross terms:")
            for name, section in cross_terms.items():
                lines.append(f"    {name}: {len(section)} entries")
        if self.flagged_angleangles:
            lines.append(f"  Angle-angle impropers: {len(self.flagged_angleangles)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PCFF(name={self.name!r}, "
                f"atom_types={self.n_atom_types()}, "
                f"bond_types={self.n_bond_types()})")
