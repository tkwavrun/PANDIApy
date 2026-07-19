"""
pandiapy.forcefield.frc_file
============================
Reader for Biosym/MSI/Accelrys force-field parameter files (.frc).

Supports: PCFF/IFF (cff91), Compass, CVFF/ClayFF (cvff), OPLS-AA, DREIDING.

Design notes
------------
* All parameter types are frozen dataclasses — immutable after construction.
* Version selection: on duplicate keys the entry with the *higher* ver value
  wins (mirrors LUNAR's behaviour exactly).
* Wildcard entries (any atom-type token == '*') are stored in separate lists,
  in file order, rather than in the main dicts.  This mirrors lookup priority:
  scan the main dict first, then walk the wildcard list.
* The reader uses a single 'section' string instead of 33 boolean flags;
  it is reset to '' on every '#' header line, then set to the matching name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _has_wildcard(key: tuple) -> bool:
    return any(t == '*' for t in key)


def _vinsert(d: dict, key, entry) -> None:
    """Insert *entry* into *d[key]* keeping the highest-version entry."""
    if key not in d or entry.ver >= d[key].ver:
        d[key] = entry


def _winsert(d: dict, wlist: list, key: tuple, entry) -> None:
    """Route to wildcard list or main dict based on whether key contains '*'."""
    if _has_wildcard(key):
        for idx, (k, e) in enumerate(wlist):
            if k == key and entry.ver >= e.ver:
                wlist[idx] = (key, entry)
                return
        if not any(k == key for k, _ in wlist):
            wlist.append((key, entry))
    else:
        _vinsert(d, key, entry)


# ---------------------------------------------------------------------------
# Parameter dataclasses (all frozen=True)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomTypeDef:
    symbol:      str
    mass:        float
    element:     str
    connections: int | str   # int, or 'n.u.' if not in file
    comment:     str
    ver:         float
    ref:         float


@dataclass(frozen=True)
class EquivalenceEntry:
    """Equivalence table: maps one atom type to substitution types per interaction class."""
    type_label: str
    nonbond:    str
    bond:       str
    angle:      str
    torsion:    str
    oop:        str
    ver:        float
    ref:        float


@dataclass(frozen=True)
class AutoEquivalenceEntry:
    """
    Auto-equivalence table.

    'bond_inct' (column 4) is used for bond-increment charge lookups.
    'bond'      (column 5) is used for quadratic-bond parameter lookups.
    These are *different* columns — must not be conflated.
    """
    type_label:     str
    nonbond:        str
    bond_inct:      str
    bond:           str
    angle_end:      str
    angle_apex:     str
    torsion_end:    str
    torsion_center: str
    oop_end:        str
    oop_center:     str
    ver:            float
    ref:            float


@dataclass(frozen=True)
class BondIncEntry:
    """Asymmetric bond-increment: delta_ij to atom i, delta_ji to atom j."""
    ij:  float
    ji:  float
    ver: float
    ref: float


@dataclass(frozen=True)
class QuarticBond:
    r0: float; k2: float; k3: float; k4: float
    ver: float; ref: float


@dataclass(frozen=True)
class QuadraticBond:
    r0: float; k2: float
    ver: float; ref: float


@dataclass(frozen=True)
class MorseBond:
    r0: float; d: float; alpha: float
    ver: float; ref: float


@dataclass(frozen=True)
class QuarticAngle:
    theta0: float; k2: float; k3: float; k4: float
    ver: float; ref: float


@dataclass(frozen=True)
class QuadraticAngle:
    theta0: float; k2: float
    ver: float; ref: float


@dataclass(frozen=True)
class Torsion3:
    v1: float; phi1: float
    v2: float; phi2: float
    v3: float; phi3: float
    ver: float; ref: float


@dataclass(frozen=True)
class Torsion1:
    kphi: float; n: int; phi0: float
    ver: float; ref: float


@dataclass(frozen=True)
class WilsonOOP:
    """Class-2 improper: E = kchi*(chi - chi0)^2."""
    kchi: float; chi0: float
    ver: float; ref: float


@dataclass(frozen=True)
class OutOfPlane:
    """Class-1 OOP: E = kchi*(chi - chi0)^n."""
    kchi: float; n: int; chi0: float
    ver: float; ref: float


@dataclass(frozen=True)
class DREIDINGOOP:
    kl: float; phi0: float
    ver: float; ref: float


@dataclass(frozen=True)
class PairCoeff96:
    r: float; eps: float
    ver: float; ref: float


@dataclass(frozen=True)
class PairCoeff126:
    A: float; B: float
    ver: float; ref: float


@dataclass(frozen=True)
class PairBuckingham:
    A: float; B: float; C: float
    ver: float; ref: float


@dataclass(frozen=True)
class Bondbond:
    kb_bp: float; ver: float; ref: float


@dataclass(frozen=True)
class Bondbond13:
    kb_bp: float; ver: float; ref: float


@dataclass(frozen=True)
class Bondangle:
    """kb_theta for left bond, kbp_theta for right; equal when symmetric."""
    kb_theta: float; kbp_theta: float
    ver: float; ref: float


@dataclass(frozen=True)
class Angleangle:
    """Single coupling constant; assigner does three independent lookups."""
    k_theta_thetap: float; ver: float; ref: float


@dataclass(frozen=True)
class Endbondtorsion:
    l_f1: float; l_f2: float; l_f3: float
    r_f1: float; r_f2: float; r_f3: float
    ver: float; ref: float


@dataclass(frozen=True)
class Middlebondtorsion:
    f1: float; f2: float; f3: float
    ver: float; ref: float


@dataclass(frozen=True)
class Angletorsion:
    l_f1: float; l_f2: float; l_f3: float
    r_f1: float; r_f2: float; r_f3: float
    ver: float; ref: float


@dataclass(frozen=True)
class Angleangletorsion:
    k_ang_ang_tor: float; ver: float; ref: float


# ---------------------------------------------------------------------------
# FRC_File container
# ---------------------------------------------------------------------------

@dataclass
class FRC_File:
    """
    Parsed force-field parameter file.

    Sections that support wildcards expose both a main dict (exact entries) and
    a *_wildcard list (file-order list of (key, entry) tuples).  Resolvers
    scan the main dict first, then the wildcard list.
    """
    forcefield_type: str
    filename:        str

    # Atom type and mapping tables
    atom_types:        dict = field(default_factory=dict)   # str -> AtomTypeDef
    equivalences:      dict = field(default_factory=dict)   # str -> EquivalenceEntry
    auto_equivalences: dict = field(default_factory=dict)   # str -> AutoEquivalenceEntry
    bond_increments:   dict = field(default_factory=dict)   # (i,j) -> BondIncEntry

    # Bond params
    quartic_bonds:               dict = field(default_factory=dict)
    quartic_bonds_wildcard:      list = field(default_factory=list)
    quadratic_bonds:             dict = field(default_factory=dict)
    quadratic_bonds_wildcard:    list = field(default_factory=list)
    quadratic_bonds_auto:        dict = field(default_factory=dict)
    quadratic_bonds_auto_wildcard: list = field(default_factory=list)
    morse_bonds:                 dict = field(default_factory=dict)
    morse_bonds_wildcard:        list = field(default_factory=list)
    morse_bonds_auto:            dict = field(default_factory=dict)
    morse_bonds_auto_wildcard:   list = field(default_factory=list)

    # Angle params
    quartic_angles:              dict = field(default_factory=dict)
    quartic_angles_wildcard:     list = field(default_factory=list)
    quadratic_angles:            dict = field(default_factory=dict)
    quadratic_angles_wildcard:   list = field(default_factory=list)
    quadratic_angles_auto:       dict = field(default_factory=dict)
    quadratic_angles_auto_wildcard: list = field(default_factory=list)

    # Dihedral params
    torsion_3:               dict = field(default_factory=dict)
    torsion_3_wildcard:      list = field(default_factory=list)
    torsion_1:               dict = field(default_factory=dict)
    torsion_1_wildcard:      list = field(default_factory=list)
    torsion_1_auto:          dict = field(default_factory=dict)
    torsion_1_auto_wildcard: list = field(default_factory=list)
    torsion_1_opls:          dict = field(default_factory=dict)
    torsion_1_opls_wildcard: list = field(default_factory=list)

    # Improper / OOP params
    wilson_out_of_plane:               dict = field(default_factory=dict)
    wilson_out_of_plane_wildcard:      list = field(default_factory=list)
    wilson_out_of_plane_auto:          dict = field(default_factory=dict)
    wilson_out_of_plane_auto_wildcard: list = field(default_factory=list)
    out_of_plane:               dict = field(default_factory=dict)
    out_of_plane_wildcard:      list = field(default_factory=list)
    out_of_plane_auto:          dict = field(default_factory=dict)
    out_of_plane_auto_wildcard: list = field(default_factory=list)
    out_of_plane_dreiding:      dict = field(default_factory=dict)

    # Non-bonded pair params
    pair_coeffs_9_6:  dict = field(default_factory=dict)
    pair_coeffs_12_6: dict = field(default_factory=dict)
    pair_buckingham:  dict = field(default_factory=dict)

    # Class-2 cross terms
    bondbond:          dict = field(default_factory=dict)
    bondbond13:        dict = field(default_factory=dict)
    bondangle:         dict = field(default_factory=dict)
    angleangle:        dict = field(default_factory=dict)
    endbondtorsion:    dict = field(default_factory=dict)
    middlebondtorsion: dict = field(default_factory=dict)
    angletorsion:      dict = field(default_factory=dict)
    angleangletorsion: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> 'FRC_File':
        """
        Parse an .frc file and return an FRC_File instance.

        Version selection: on duplicate keys the entry with the *higher* .ver
        overwrites the existing one (ties go to the later occurrence).
        Wildcards ('*' in any type token) are stored in *_wildcard lists.
        """
        filename = os.path.abspath(path)
        forcefield_type = 'unknown'

        # Mutable working collections
        atom_types:        dict = {}
        equivalences:      dict = {}
        auto_equivalences: dict = {}
        bond_increments:   dict = {}

        quartic_bonds     = {}; quartic_bonds_wc     = []
        quadratic_bonds   = {}; quadratic_bonds_wc   = []
        quadratic_bonds_auto = {}; quadratic_bonds_auto_wc = []
        morse_bonds       = {}; morse_bonds_wc       = []
        morse_bonds_auto  = {}; morse_bonds_auto_wc  = []

        quartic_angles        = {}; quartic_angles_wc        = []
        quadratic_angles      = {}; quadratic_angles_wc      = []
        quadratic_angles_auto = {}; quadratic_angles_auto_wc = []

        torsion_3        = {}; torsion_3_wc        = []
        torsion_1        = {}; torsion_1_wc        = []
        torsion_1_auto   = {}; torsion_1_auto_wc   = []
        torsion_1_opls   = {}; torsion_1_opls_wc   = []

        wilson_oop      = {}; wilson_oop_wc      = []
        wilson_oop_auto = {}; wilson_oop_auto_wc = []
        out_of_plane      = {}; out_of_plane_wc      = []
        out_of_plane_auto = {}; out_of_plane_auto_wc = []
        out_of_plane_dreiding = {}

        pair_9_6  = {}; pair_12_6 = {}; pair_buck = {}

        bondbond          = {}; bondbond13        = {}
        bondangle         = {}; angleangle        = {}
        endbondtorsion    = {}; middlebondtorsion = {}
        angletorsion      = {}; angleangletorsion = {}

        section = ''   # name of the active section ('quartic_bond', etc.)

        with open(filename, 'r', errors='replace') as fh:
            for raw in fh:
                line = raw.strip()
                tok  = line.split()
                if not tok:
                    continue

                # ----------------------------------------------------------
                # Detect forcefield type from !define / @define line
                # ----------------------------------------------------------
                if tok[0] in ('!define', '@define') or tok[0] == '!Ver':
                    pass   # version header — ignore
                if len(tok) >= 2 and tok[0].startswith('!') and 'define' not in line.lower():
                    pass
                # Detect from first token containing a ff name on a define line
                for ff in ('cff91', 'compass', 'cvff', 'oplsaa', 'opls', 'dreiding'):
                    if ff in line.lower() and 'define' in line.lower():
                        forcefield_type = ff
                        break

                # ----------------------------------------------------------
                # Section header line — identified by '#' start token
                # ----------------------------------------------------------
                if tok[0].startswith('#'):
                    section = ''   # reset

                    h = tok[0]   # '#atom_types', '#quartic_bond', etc.

                    if '#atom_types' in h:
                        section = 'atom_types'
                    elif '#auto_equivalence' in h:
                        section = 'auto_equivalence'
                    elif '#equivalence' in h:
                        section = 'equivalence'
                    elif '#bond_increments' in h:
                        section = 'bond_increments'
                    elif '#quartic_bond' in h:
                        section = 'quartic_bond'
                    elif '#quadratic_bond' in h:
                        if 'cff91_auto' in line:
                            section = 'quadratic_bond'        # class2 auto-equiv
                        elif 'cvff' in line and '_auto' not in line:
                            section = 'quadratic_bond'        # class1 equiv
                        elif 'cvff' in line and '_auto' in line:
                            section = 'quadratic_bond_auto'   # class1 auto-equiv
                    elif '#morse_bond' in h:
                        if '_auto' in line:
                            section = 'morse_bond_auto'
                        else:
                            section = 'morse_bond'
                    elif '#quartic_angle' in h:
                        section = 'quartic_angle'
                    elif '#quadratic_angle' in h:
                        if 'cff91_auto' in line:
                            section = 'quadratic_angle'
                        elif 'cvff' in line and '_auto' not in line:
                            section = 'quadratic_angle'
                        elif 'cvff' in line and '_auto' in line:
                            section = 'quadratic_angle_auto'
                    elif '#torsion_3' in h:
                        section = 'torsion_3'
                    elif '#torsion_1' in h:
                        if 'cff91_auto' in line:
                            section = 'torsion_1'
                        elif 'opls' in line.lower():
                            section = 'torsion_1_opls'
                        elif 'cvff' in line and '_auto' not in line:
                            section = 'torsion_1'
                        elif 'cvff' in line and '_auto' in line:
                            section = 'torsion_1_auto'
                    elif '#wilson_out_of_plane' in h:
                        section = 'wilson_auto' if '_auto' in line else 'wilson'
                    elif '#out_of_plane' in h:
                        if 'DREIDING' in line:
                            section = 'oop_dreiding'
                        elif '_auto' in line:
                            section = 'oop_auto'
                        else:
                            section = 'oop'
                    elif '#nonbond(9-6)' in h:
                        section = 'pair_9_6'
                    elif '#nonbond(12-6)' in h:
                        section = 'pair_12_6'
                    elif '#nonbond(Buckingham)' in h:
                        section = 'pair_buck'
                    elif '#bond-bond_1_3' in h:
                        section = 'bondbond13'
                    elif '#bond-bond' in h:
                        section = 'bondbond'
                    elif '#bond-angle' in h:
                        section = 'bondangle'
                    elif '#angle-angle-torsion_1' in h:
                        section = 'aat'
                    elif '#angle-angle' in h:
                        section = 'angleangle'
                    elif '#end_bond-torsion_3' in h:
                        section = 'ebt'
                    elif '#middle_bond-torsion_3' in h:
                        section = 'mbt'
                    elif '#angle-torsion_3' in h:
                        section = 'at'

                    continue   # header carries no parameter data

                # ----------------------------------------------------------
                # Data lines — must start with a float (ver column)
                # ----------------------------------------------------------
                if not _is_float(tok[0]):
                    continue

                n = len(tok)

                if section == 'atom_types' and n >= 5:
                    ver = float(tok[0]); ref = float(tok[1]); sym = tok[2]
                    try:
                        conn = int(tok[5])
                        comment = ' '.join(tok[6:])
                    except (IndexError, ValueError):
                        conn = 'n.u.'
                        comment = ' '.join(tok[5:])
                    _vinsert(atom_types, sym, AtomTypeDef(
                        symbol=sym, mass=float(tok[3]), element=tok[4],
                        connections=conn, comment=comment, ver=ver, ref=ref))

                elif section == 'equivalence' and n >= 8:
                    ver = float(tok[0]); ref = float(tok[1]); t = tok[2]
                    _vinsert(equivalences, t, EquivalenceEntry(
                        type_label=t, nonbond=tok[3], bond=tok[4],
                        angle=tok[5], torsion=tok[6], oop=tok[7],
                        ver=ver, ref=ref))

                elif section == 'auto_equivalence' and n >= 12:
                    ver = float(tok[0]); ref = float(tok[1]); t = tok[2]
                    _vinsert(auto_equivalences, t, AutoEquivalenceEntry(
                        type_label=t, nonbond=tok[3],
                        bond_inct=tok[4], bond=tok[5],
                        angle_end=tok[6], angle_apex=tok[7],
                        torsion_end=tok[8], torsion_center=tok[9],
                        oop_end=tok[10], oop_center=tok[11],
                        ver=ver, ref=ref))

                elif section == 'bond_increments' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _vinsert(bond_increments, (i, j), BondIncEntry(
                        ij=float(tok[4]), ji=float(tok[5]), ver=ver, ref=ref))

                elif section == 'quartic_bond' and n >= 8:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _winsert(quartic_bonds, quartic_bonds_wc, (i, j),
                             QuarticBond(r0=float(tok[4]), k2=float(tok[5]),
                                         k3=float(tok[6]), k4=float(tok[7]),
                                         ver=ver, ref=ref))

                elif section == 'quadratic_bond' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _winsert(quadratic_bonds, quadratic_bonds_wc, (i, j),
                             QuadraticBond(r0=float(tok[4]), k2=float(tok[5]),
                                           ver=ver, ref=ref))

                elif section == 'quadratic_bond_auto' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _winsert(quadratic_bonds_auto, quadratic_bonds_auto_wc, (i, j),
                             QuadraticBond(r0=float(tok[4]), k2=float(tok[5]),
                                           ver=ver, ref=ref))

                elif section == 'morse_bond' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _winsert(morse_bonds, morse_bonds_wc, (i, j),
                             MorseBond(r0=float(tok[4]), d=float(tok[5]),
                                       alpha=float(tok[6]), ver=ver, ref=ref))

                elif section == 'morse_bond_auto' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]
                    _winsert(morse_bonds_auto, morse_bonds_auto_wc, (i, j),
                             MorseBond(r0=float(tok[4]), d=float(tok[5]),
                                       alpha=float(tok[6]), ver=ver, ref=ref))

                elif section == 'quartic_angle' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]
                    _winsert(quartic_angles, quartic_angles_wc, (i, j, k),
                             QuarticAngle(theta0=float(tok[5]), k2=float(tok[6]),
                                          k3=float(tok[7]), k4=float(tok[8]),
                                          ver=ver, ref=ref))

                elif section == 'quadratic_angle' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]
                    _winsert(quadratic_angles, quadratic_angles_wc, (i, j, k),
                             QuadraticAngle(theta0=float(tok[5]), k2=float(tok[6]),
                                            ver=ver, ref=ref))

                elif section == 'quadratic_angle_auto' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]
                    _winsert(quadratic_angles_auto, quadratic_angles_auto_wc, (i, j, k),
                             QuadraticAngle(theta0=float(tok[5]), k2=float(tok[6]),
                                            ver=ver, ref=ref))

                elif section == 'torsion_3' and n >= 12:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(torsion_3, torsion_3_wc, (i, j, k, l),
                             Torsion3(v1=float(tok[6]), phi1=float(tok[7]),
                                      v2=float(tok[8]), phi2=float(tok[9]),
                                      v3=float(tok[10]), phi3=float(tok[11]),
                                      ver=ver, ref=ref))

                elif section == 'torsion_1' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(torsion_1, torsion_1_wc, (i, j, k, l),
                             Torsion1(kphi=float(tok[6]), n=int(tok[7]),
                                      phi0=float(tok[8]), ver=ver, ref=ref))

                elif section == 'torsion_1_auto' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(torsion_1_auto, torsion_1_auto_wc, (i, j, k, l),
                             Torsion1(kphi=float(tok[6]), n=int(tok[7]),
                                      phi0=float(tok[8]), ver=ver, ref=ref))

                elif section == 'torsion_1_opls' and n >= 10:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    # OPLS has 4 constants k1..k4; store k1 in kphi, n=4 sentinel, phi0=k2
                    _winsert(torsion_1_opls, torsion_1_opls_wc, (i, j, k, l),
                             Torsion1(kphi=float(tok[6]), n=4,
                                      phi0=float(tok[7]), ver=ver, ref=ref))

                elif section == 'wilson' and n >= 8:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(wilson_oop, wilson_oop_wc, (i, j, k, l),
                             WilsonOOP(kchi=float(tok[6]), chi0=float(tok[7]),
                                       ver=ver, ref=ref))

                elif section == 'wilson_auto' and n >= 8:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(wilson_oop_auto, wilson_oop_auto_wc, (i, j, k, l),
                             WilsonOOP(kchi=float(tok[6]), chi0=float(tok[7]),
                                       ver=ver, ref=ref))

                elif section == 'oop' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(out_of_plane, out_of_plane_wc, (i, j, k, l),
                             OutOfPlane(kchi=float(tok[6]), n=int(tok[7]),
                                        chi0=float(tok[8]), ver=ver, ref=ref))

                elif section == 'oop_auto' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _winsert(out_of_plane_auto, out_of_plane_auto_wc, (i, j, k, l),
                             OutOfPlane(kchi=float(tok[6]), n=int(tok[7]),
                                        chi0=float(tok[8]), ver=ver, ref=ref))

                elif section == 'oop_dreiding' and n >= 8 and _is_float(tok[6]):
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _vinsert(out_of_plane_dreiding, (i, j, k, l),
                             DREIDINGOOP(kl=float(tok[6]), phi0=float(tok[7]),
                                         ver=ver, ref=ref))

                elif section == 'pair_9_6' and n >= 5:
                    ver = float(tok[0]); ref = float(tok[1]); sym = tok[2]
                    _vinsert(pair_9_6, sym, PairCoeff96(
                        r=float(tok[3]), eps=float(tok[4]), ver=ver, ref=ref))

                elif section == 'pair_12_6' and n >= 5:
                    ver = float(tok[0]); ref = float(tok[1]); sym = tok[2]
                    _vinsert(pair_12_6, sym, PairCoeff126(
                        A=float(tok[3]), B=float(tok[4]), ver=ver, ref=ref))

                elif section == 'pair_buck' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1]); sym = tok[2]
                    _vinsert(pair_buck, sym, PairBuckingham(
                        A=float(tok[3]), B=float(tok[4]), C=float(tok[5]),
                        ver=ver, ref=ref))

                elif section == 'bondbond' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]
                    _vinsert(bondbond, (i, j, k), Bondbond(
                        kb_bp=float(tok[5]), ver=ver, ref=ref))

                elif section == 'bondbond13' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _vinsert(bondbond13, (i, j, k, l), Bondbond13(
                        kb_bp=float(tok[6]), ver=ver, ref=ref))

                elif section == 'bondangle' and n >= 6:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]
                    kb = float(tok[5])
                    try:
                        kbp = float(tok[6])
                    except (IndexError, ValueError):
                        kbp = kb
                    _vinsert(bondangle, (i, j, k), Bondangle(
                        kb_theta=kb, kbp_theta=kbp, ver=ver, ref=ref))

                elif section == 'angleangle' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _vinsert(angleangle, (i, j, k, l), Angleangle(
                        k_theta_thetap=float(tok[6]), ver=ver, ref=ref))

                elif section == 'ebt' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    lf1 = float(tok[6]); lf2 = float(tok[7]); lf3 = float(tok[8])
                    try:
                        rf1 = float(tok[9]); rf2 = float(tok[10]); rf3 = float(tok[11])
                    except (IndexError, ValueError):
                        rf1 = lf1; rf2 = lf2; rf3 = lf3
                    _vinsert(endbondtorsion, (i, j, k, l), Endbondtorsion(
                        l_f1=lf1, l_f2=lf2, l_f3=lf3,
                        r_f1=rf1, r_f2=rf2, r_f3=rf3, ver=ver, ref=ref))

                elif section == 'mbt' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _vinsert(middlebondtorsion, (i, j, k, l), Middlebondtorsion(
                        f1=float(tok[6]), f2=float(tok[7]), f3=float(tok[8]),
                        ver=ver, ref=ref))

                elif section == 'at' and n >= 9:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    lf1 = float(tok[6]); lf2 = float(tok[7]); lf3 = float(tok[8])
                    try:
                        rf1 = float(tok[9]); rf2 = float(tok[10]); rf3 = float(tok[11])
                    except (IndexError, ValueError):
                        rf1 = lf1; rf2 = lf2; rf3 = lf3
                    _vinsert(angletorsion, (i, j, k, l), Angletorsion(
                        l_f1=lf1, l_f2=lf2, l_f3=lf3,
                        r_f1=rf1, r_f2=rf2, r_f3=rf3, ver=ver, ref=ref))

                elif section == 'aat' and n >= 7:
                    ver = float(tok[0]); ref = float(tok[1])
                    i = tok[2]; j = tok[3]; k = tok[4]; l = tok[5]
                    _vinsert(angleangletorsion, (i, j, k, l), Angleangletorsion(
                        k_ang_ang_tor=float(tok[6]), ver=ver, ref=ref))

        return cls(
            forcefield_type=forcefield_type,
            filename=filename,
            atom_types=atom_types,
            equivalences=equivalences,
            auto_equivalences=auto_equivalences,
            bond_increments=bond_increments,
            quartic_bonds=quartic_bonds,
            quartic_bonds_wildcard=quartic_bonds_wc,
            quadratic_bonds=quadratic_bonds,
            quadratic_bonds_wildcard=quadratic_bonds_wc,
            quadratic_bonds_auto=quadratic_bonds_auto,
            quadratic_bonds_auto_wildcard=quadratic_bonds_auto_wc,
            morse_bonds=morse_bonds,
            morse_bonds_wildcard=morse_bonds_wc,
            morse_bonds_auto=morse_bonds_auto,
            morse_bonds_auto_wildcard=morse_bonds_auto_wc,
            quartic_angles=quartic_angles,
            quartic_angles_wildcard=quartic_angles_wc,
            quadratic_angles=quadratic_angles,
            quadratic_angles_wildcard=quadratic_angles_wc,
            quadratic_angles_auto=quadratic_angles_auto,
            quadratic_angles_auto_wildcard=quadratic_angles_auto_wc,
            torsion_3=torsion_3,
            torsion_3_wildcard=torsion_3_wc,
            torsion_1=torsion_1,
            torsion_1_wildcard=torsion_1_wc,
            torsion_1_auto=torsion_1_auto,
            torsion_1_auto_wildcard=torsion_1_auto_wc,
            torsion_1_opls=torsion_1_opls,
            torsion_1_opls_wildcard=torsion_1_opls_wc,
            wilson_out_of_plane=wilson_oop,
            wilson_out_of_plane_wildcard=wilson_oop_wc,
            wilson_out_of_plane_auto=wilson_oop_auto,
            wilson_out_of_plane_auto_wildcard=wilson_oop_auto_wc,
            out_of_plane=out_of_plane,
            out_of_plane_wildcard=out_of_plane_wc,
            out_of_plane_auto=out_of_plane_auto,
            out_of_plane_auto_wildcard=out_of_plane_auto_wc,
            out_of_plane_dreiding=out_of_plane_dreiding,
            pair_coeffs_9_6=pair_9_6,
            pair_coeffs_12_6=pair_12_6,
            pair_buckingham=pair_buck,
            bondbond=bondbond,
            bondbond13=bondbond13,
            bondangle=bondangle,
            angleangle=angleangle,
            endbondtorsion=endbondtorsion,
            middlebondtorsion=middlebondtorsion,
            angletorsion=angletorsion,
            angleangletorsion=angleangletorsion,
        )

    def summary(self) -> str:
        lines = [
            f"FRC_File: {os.path.basename(self.filename)}",
            f"  Forcefield type : {self.forcefield_type}",
            f"  Atom types      : {len(self.atom_types)}",
            f"  Equivalences    : {len(self.equivalences)}",
            f"  Auto-equiv      : {len(self.auto_equivalences)}",
            f"  Bond increments : {len(self.bond_increments)}",
            f"  Quartic bonds   : {len(self.quartic_bonds)} (+{len(self.quartic_bonds_wildcard)} wildcards)",
            f"  Quadratic bonds : {len(self.quadratic_bonds)} (+{len(self.quadratic_bonds_wildcard)} wildcards)",
            f"  Quartic angles  : {len(self.quartic_angles)} (+{len(self.quartic_angles_wildcard)} wildcards)",
            f"  Quadratic angles: {len(self.quadratic_angles)} (+{len(self.quadratic_angles_wildcard)} wildcards)",
            f"  Torsion-3       : {len(self.torsion_3)} (+{len(self.torsion_3_wildcard)} wildcards)",
            f"  Torsion-1       : {len(self.torsion_1)} (+{len(self.torsion_1_wildcard)} wildcards)",
            f"  Wilson OOP      : {len(self.wilson_out_of_plane)} (+{len(self.wilson_out_of_plane_wildcard)} wildcards)",
            f"  Wilson OOP auto : {len(self.wilson_out_of_plane_auto)} (+{len(self.wilson_out_of_plane_auto_wildcard)} wildcards)",
            f"  Pair (9-6)      : {len(self.pair_coeffs_9_6)}",
            f"  Bond-bond       : {len(self.bondbond)}",
            f"  Bond-bond-1-3   : {len(self.bondbond13)}",
            f"  Bond-angle      : {len(self.bondangle)}",
            f"  Angle-angle     : {len(self.angleangle)}",
            f"  EndBondTorsion  : {len(self.endbondtorsion)}",
            f"  MidBondTorsion  : {len(self.middlebondtorsion)}",
            f"  AngleTorsion    : {len(self.angletorsion)}",
            f"  AngleAngleTors  : {len(self.angleangletorsion)}",
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"FRC_File(type={self.forcefield_type!r}, "
                f"atom_types={len(self.atom_types)}, "
                f"quartic_bonds={len(self.quartic_bonds)})")
