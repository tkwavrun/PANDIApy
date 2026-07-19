"""
pandiapy.forcefield._badi
=========================
Graph-theory topology enumeration and canonical type-ID assignment.

BADIEnumerator is a direct port of LUNAR's find_BADI.py with the LUNAR
class/function boundaries collapsed into a single class.  All graph traversal
logic is identical to LUNAR — this guarantees that the canonical type ordering
(and therefore the integer type IDs) match LUNAR's output on the same input.

Design notes
------------
* BADIEnumerator does NOT mutate the Microstate object.
* The enumerator always graph-traverses from mol.bonds — it never trusts
  pre-parsed mol.angles / mol.dihedrals / mol.impropers.  This guarantees
  canonical ordering regardless of what the input .data file contained.
* 'nta' (name-to-atom dict in LUNAR parlance) is built here as
  {atom_id: atom.type_name} and exposed as self.nta for use by the assigner.
* For class-2 FFs the improper list is extended with the angleangle list
  (matching LUNAR's self.impropers.extend(self.angleangles)).  The
  flagged_angleangles list carries the integer type IDs of the angle-angle
  entries so the assigner can dispatch to the correct cross-term lookup.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandiapy.microstate import Microstate


# ---------------------------------------------------------------------------
# Graph construction helpers (ported directly from LUNAR find_BADI.py)
# ---------------------------------------------------------------------------

def _build_graph(mol: 'Microstate') -> dict[int, list[int]]:
    """Adjacency list graph: {atom_id: [neighbour_ids]}."""
    graph: dict[int, list[int]] = {i: [] for i in mol.atoms}
    for bond in mol.bonds.values():
        id1 = bond.atom1.id
        id2 = bond.atom2.id
        graph[id1].append(id2)
        graph[id2].append(id1)
    return graph


def _get_bonds(mol: 'Microstate') -> list[tuple[int, int]]:
    """Sorted list of (id1, id2) bond pairs, each with id1 < id2."""
    bonds: set[tuple[int, int]] = set()
    for bond in mol.bonds.values():
        bonds.add(tuple(sorted((bond.atom1.id, bond.atom2.id))))
    return sorted(bonds)


def _gt_angles(graph: dict, bonds: list) -> list[tuple[int, int, int]]:
    """
    Graph-theory angle enumeration.  Returns sorted list of (end, center, end)
    triples sorted by center atom ID then outer atom IDs.
    """
    angles: set[tuple[int, int, int]] = set()
    for id1, id2 in bonds:
        for id3 in graph[id1]:
            if len({id1, id2, id3}) == 3:
                angle = (id2, id1, id3) if id2 < id3 else (id3, id1, id2)
                if angle not in angles and tuple(reversed(angle)) not in angles:
                    angles.add(angle)
        for id3 in graph[id2]:
            if len({id1, id2, id3}) == 3:
                angle = (id1, id2, id3) if id1 < id3 else (id3, id2, id1)
                if angle not in angles and tuple(reversed(angle)) not in angles:
                    angles.add(angle)
    return sorted(angles, key=lambda x: x[1])


def _gt_dihedrals_impropers_angleangles(
    graph: dict, angles: list
) -> tuple[list, list, list]:
    """
    Graph-theory dihedral / improper / angle-angle enumeration.

    Improper criterion  (matches LUNAR exactly):
        nb == 3  → Wilson OOP improper
        nb >  3  → angle-angle cross-term

    Canonical improper form: (sorted_outer[0], center, sorted_outer[1], sorted_outer[2])
    """
    dihedrals:   set[tuple] = set()
    impropers:   set[tuple] = set()
    angleangles: set[tuple] = set()

    for id1, id2, id3 in angles:
        # --- dihedrals via id1 end ---
        for id4 in graph[id1]:
            if len({id1, id2, id3, id4}) == 4:
                dihedral = (id3, id2, id1, id4) if id3 < id4 else (id4, id1, id2, id3)
                if dihedral not in dihedrals and tuple(reversed(dihedral)) not in dihedrals:
                    dihedrals.add(dihedral)
        # --- dihedrals via id3 end ---
        for id4 in graph[id3]:
            if len({id1, id2, id3, id4}) == 4:
                dihedral = (id1, id2, id3, id4) if id1 < id4 else (id4, id3, id2, id1)
                if dihedral not in dihedrals and tuple(reversed(dihedral)) not in dihedrals:
                    dihedrals.add(dihedral)
        # --- impropers / angle-angles via center id2 ---
        for id4 in graph[id2]:
            if len({id4, id1, id2, id3}) == 4:
                outsides = sorted([id1, id3, id4])
                oop = (outsides[0], id2, outsides[1], outsides[2])
                nb  = len(graph[id2])
                if nb == 3:
                    impropers.add(oop)
                elif nb > 3:
                    angleangles.add(oop)

    return (
        sorted(dihedrals,   key=lambda x: x[1]),
        sorted(impropers,   key=lambda x: x[1]),
        sorted(angleangles, key=lambda x: x[1]),
    )


# ---------------------------------------------------------------------------
# Type canonicalization helpers
# ---------------------------------------------------------------------------

def _canon_bond(nta1: str, nta2: str) -> tuple[str, str]:
    return tuple(sorted([nta1, nta2]))


def _canon_angle(nta1: str, nta2: str, nta3: str) -> tuple[str, str, str]:
    """Outer atoms sorted alphabetically; center stays at position [1]."""
    if nta1 != nta3:
        return (nta1, nta2, nta3) if nta1 < nta3 else (nta3, nta2, nta1)
    return (nta1, nta2, nta3)   # same outer type — forward order


def _canon_dihedral(
    nta1: str, nta2: str, nta3: str, nta4: str
) -> tuple[str, str, str, str]:
    """Canonical dihedral: sort by outer pair first, then inner pair if outer equal."""
    if nta1 != nta4:
        return (nta1, nta2, nta3, nta4) if nta1 < nta4 else (nta4, nta3, nta2, nta1)
    elif nta2 != nta3:
        return (nta1, nta2, nta3, nta4) if nta2 < nta3 else (nta4, nta3, nta2, nta1)
    return (nta1, nta2, nta3, nta4)


def _canon_improper(
    nta1: str, nta2: str, nta3: str, nta4: str
) -> tuple[str, str, str, str]:
    """
    Canonical improper: center is always at position [1]; the three outer atoms
    (positions 0, 2, 3) are sorted alphabetically.
    Input BADI form: (outer_a, center, outer_b, outer_c).
    """
    outer = sorted([nta1, nta3, nta4])
    return (outer[0], nta2, outer[1], outer[2])


def _multisort(lst: list) -> list:
    """Sort a list of tuples using all positions as sort keys (last to first, stable)."""
    result = lst[:]
    for pos in range(len(result[0]) - 1, -1, -1):
        result = sorted(result, key=lambda x: x[pos])
    return result


# ---------------------------------------------------------------------------
# Type-ID assignment helpers
# ---------------------------------------------------------------------------

def _assign_type_ids(type_set: set) -> dict[tuple, int]:
    """
    Sort unique types and assign 1-based integer IDs, matching LUNAR's ordering.
    Multi-key sort: primary by position 0, secondary by position 1, etc.
    """
    unique = list(type_set)
    if not unique:
        return {}
    unique = _multisort(unique)
    return {t: n + 1 for n, t in enumerate(unique)}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class BADIEnumerator:
    """
    Enumerate bonds, angles, dihedrals, impropers, and angle-angles from a
    Microstate, assign canonical integer type IDs, and expose the results.

    Parameters
    ----------
    mol : Microstate
        The system to enumerate.  All atoms must have a non-None type_name.

    Attributes
    ----------
    nta                  : {atom_id: type_name_string}
    bonds                : sorted list of (id1, id2) tuples
    angles               : sorted list of (end, center, end) tuples
    dihedrals            : sorted list of (id1, id2, id3, id4) tuples
    impropers            : sorted list of OOP improper (id1..id4) tuples
    angleangles          : sorted list of angle-angle (id1..id4) tuples
    atom_type_ids        : {type_label: int}
    bond_type_ids        : {(t1,t2): int}
    angle_type_ids       : {(t1,t2,t3): int}
    dihedral_type_ids    : {(t1,t2,t3,t4): int}
    improper_type_ids    : {(t1,t2,t3,t4): int}  — OOP Wilson types
    angleangle_type_ids  : {(t1,t2,t3,t4): int}  — angle-angle types (IDs continue after OOP)
    flagged_angleangles  : list[int] — type IDs that are angle-angle, not Wilson OOP
    atom_type_labels     : {int: type_label}  — inverse of atom_type_ids
    bond_type_labels     : {int: str}  — '-'.join(type_tuple)
    angle_type_labels    : {int: str}
    dihedral_type_labels : {int: str}
    improper_type_labels : {int: str}  — covers both OOP and angle-angle
    """

    def __init__(self, mol: 'Microstate') -> None:
        # Build nta: {atom_id: type_name}
        self.nta: dict[int, str] = {}
        for atom_id, atom in mol.atoms.items():
            if atom.type_name is None:
                raise ValueError(
                    f"Atom {atom_id} has no type_name. "
                    "All atoms must be typed before calling BADIEnumerator."
                )
            self.nta[atom_id] = atom.type_name

        # Graph and topology
        graph             = _build_graph(mol)
        self.bonds        = _get_bonds(mol)
        self.angles       = _gt_angles(graph, self.bonds)
        self.dihedrals, self.impropers, self.angleangles = \
            _gt_dihedrals_impropers_angleangles(graph, self.angles)

        # Atom type IDs (alphabetical sort, 1-based)
        atom_type_set = set(self.nta.values())
        atom_types_sorted = sorted(atom_type_set)
        self.atom_type_ids: dict[str, int] = {
            t: n + 1 for n, t in enumerate(atom_types_sorted)
        }
        self.atom_type_labels: dict[int, str] = {
            v: k for k, v in self.atom_type_ids.items()
        }

        # Bond type IDs
        bond_type_set: set = set()
        for id1, id2 in self.bonds:
            bond_type_set.add(_canon_bond(self.nta[id1], self.nta[id2]))
        self.bond_type_ids = _assign_type_ids(bond_type_set)
        self.bond_type_labels = {
            v: '-'.join(k) for k, v in self.bond_type_ids.items()
        }

        # Angle type IDs
        angle_type_set: set = set()
        for id1, id2, id3 in self.angles:
            angle_type_set.add(_canon_angle(
                self.nta[id1], self.nta[id2], self.nta[id3]))
        self.angle_type_ids = _assign_type_ids(angle_type_set)
        self.angle_type_labels = {
            v: '-'.join(k) for k, v in self.angle_type_ids.items()
        }

        # Dihedral type IDs
        dihedral_type_set: set = set()
        for id1, id2, id3, id4 in self.dihedrals:
            dihedral_type_set.add(_canon_dihedral(
                self.nta[id1], self.nta[id2],
                self.nta[id3], self.nta[id4]))
        self.dihedral_type_ids = _assign_type_ids(dihedral_type_set)
        self.dihedral_type_labels = {
            v: '-'.join(k) for k, v in self.dihedral_type_ids.items()
        }

        # Improper and angle-angle type IDs
        # Impropers get IDs 1..N_oop; angle-angles continue at N_oop+1..N_total
        improper_type_set: set = set()
        for id1, id2, id3, id4 in self.impropers:
            improper_type_set.add(_canon_improper(
                self.nta[id1], self.nta[id2],
                self.nta[id3], self.nta[id4]))

        angleangle_type_set: set = set()
        for id1, id2, id3, id4 in self.angleangles:
            angleangle_type_set.add(_canon_improper(
                self.nta[id1], self.nta[id2],
                self.nta[id3], self.nta[id4]))

        # Sort both lists in the same way LUNAR does
        imp_sorted = _multisort(list(improper_type_set)) if improper_type_set else []
        aa_sorted  = _multisort(list(angleangle_type_set)) if angleangle_type_set else []

        # Assign IDs: OOP first, then angle-angle (continuous numbering)
        self.improper_type_ids:   dict[tuple, int] = {}
        self.angleangle_type_ids: dict[tuple, int] = {}
        self.flagged_angleangles: list[int]         = []
        self.improper_type_labels: dict[int, str]   = {}

        n = 1
        for t in imp_sorted:
            self.improper_type_ids[t] = n
            self.improper_type_labels[n] = '-'.join(t)
            n += 1
        for t in aa_sorted:
            self.angleangle_type_ids[t] = n
            self.improper_type_labels[n] = '-'.join(t)   # OOP section covers both
            self.flagged_angleangles.append(n)
            n += 1

    # ------------------------------------------------------------------
    # Convenience helpers used by the assigner
    # ------------------------------------------------------------------

    def bond_type_id(self, t1: str, t2: str) -> int | None:
        return self.bond_type_ids.get(_canon_bond(t1, t2))

    def angle_type_id(self, t1: str, t2: str, t3: str) -> int | None:
        return self.angle_type_ids.get(_canon_angle(t1, t2, t3))

    def dihedral_type_id(self, t1: str, t2: str, t3: str, t4: str) -> int | None:
        return self.dihedral_type_ids.get(_canon_dihedral(t1, t2, t3, t4))

    def improper_type_id(self, t1: str, t2: str, t3: str, t4: str) -> int | None:
        key = _canon_improper(t1, t2, t3, t4)
        return self.improper_type_ids.get(key) or self.angleangle_type_ids.get(key)

    def summary(self) -> str:
        lines = [
            "BADIEnumerator summary:",
            f"  Atom types    : {len(self.atom_type_ids)}",
            f"  Bond types    : {len(self.bond_type_ids)}",
            f"  Angle types   : {len(self.angle_type_ids)}",
            f"  Dihedral types: {len(self.dihedral_type_ids)}",
            f"  OOP impropers : {len(self.improper_type_ids)}",
            f"  Angle-angles  : {len(self.angleangle_type_ids)}",
            f"  Flagged AA IDs: {self.flagged_angleangles}",
            f"  Bonds         : {len(self.bonds)}",
            f"  Angles        : {len(self.angles)}",
            f"  Dihedrals     : {len(self.dihedrals)}",
            f"  Impropers     : {len(self.impropers)}",
            f"  Angle-angles  : {len(self.angleangles)}",
        ]
        return '\n'.join(lines)
