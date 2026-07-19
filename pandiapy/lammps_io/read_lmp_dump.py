# pandiapy/lammps_io/read_lmp_dump.py

from typing import Iterator

from pandiapy.microstate import Microstate
from pandiapy.topology import Atom


def subsample_dump(dumpfile: str, nevery: int, write=False) -> Iterator[Microstate]:
    import itertools
    gen = read_lmp_dump(dumpfile, write_data=write)
    return itertools.islice(gen, 0, None, nevery)


def read_lmp_dump(path: str, write_data: bool= False) -> Iterator[Microstate]:
    """
    Generator that yields one Microstate per frame in a LAMMPS dump file
    """
    with open(path, "r") as f:
        line = f.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(f.readline())
                mol = Microstate()
                mol.timestep = timestep  # dataclass has no slots, so this is fine
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(f.readline())
            elif line.startswith("ITEM: BOX BOUNDS"):
                _parse_dump_box_bounds(f, mol, line)
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]  # e.g. ["id", "type", "element", "x", "y", "z"]
                _populate_atoms_from_dump(f, mol, cols, n_atoms)
                if write_data:
                    mol.to_lmp_datafile(f"{path.split('.')[:-1]}_{timestep}.lmp")
                yield mol
            line = f.readline()
# def read_lmp_dump(path: str, nevery: int=10, write_data: bool= False) -> Iterator[Microstate]:
#     """
#     Generator that yields one Microstate per frame in a LAMMPS dump file
#     (atoms only, no bond/angle/dihedral/improper topology).
#
#     Usage:
#         for mol in read_lmp_dump("traj.dump"):
#             ...              # process frame-by-frame, O(1 frame) memory
#
#         last = None
#         for last in read_lmp_dump("traj.dump"):
#             pass             # cheap way to get just the last frame
#
#         frames = list(read_lmp_dump("traj.dump"))   # only if you truly
#                                                       # want all frames at once
#     """
#     counter = 0
#     with open(path, "r") as f:
#         line = f.readline()
#         while line:
#             if line.startswith("ITEM: TIMESTEP"):
#                 if counter % nevery != 0:
#                     while line:
#                         if line.startswith("ITEM: TIMESTEP"):
#                             counter += 1
#                             if counter % nevery-1 == 0:
#                                 break
#                         line = f.readline()
#
#                 timestep = int(f.readline())
#                 mol = Microstate()
#                 mol.timestep = timestep  # dataclass has no slots, so this is fine
#                 counter += 1
#                 if counter > 100: quit()
#             elif line.startswith("ITEM: NUMBER OF ATOMS"):
#                 n_atoms = int(f.readline())
#             elif line.startswith("ITEM: BOX BOUNDS"):
#                 _parse_dump_box_bounds(f, mol, line)
#             elif line.startswith("ITEM: ATOMS"):
#                 cols = line.split()[2:]  # e.g. ["id", "type", "element", "x", "y", "z"]
#                 _populate_atoms_from_dump(f, mol, cols, n_atoms)
#                 if write_data:
#                     mol.to_lmp_datafile(f"{path.split('.')[:-1]}_{timestep}.lmp")
#                 yield mol
#             line = f.readline()


## UTIL FUNCTIONS
def _parse_dump_box_bounds(f, mol: Microstate, header_line: str) -> None:
    is_triclinic = "xy" in header_line and "xz" in header_line and "yz" in header_line

    xvals = list(map(float, f.readline().split()))
    yvals = list(map(float, f.readline().split()))
    zvals = list(map(float, f.readline().split()))

    if is_triclinic:
        xlo_b, xhi_b, xy = xvals
        ylo_b, yhi_b, xz = yvals
        zlo, zhi, yz = zvals

        # LAMMPS dump stores "bound" box edges, not the true tilt-adjusted box.
        # Standard conversion (see LAMMPS manual, "Triclinic boxes"):
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)

        mol.box.xy, mol.box.xz, mol.box.yz = xy, xz, yz
        mol.box.triclinic = True
    else:
        xlo, xhi = xvals[:2]
        ylo, yhi = yvals[:2]
        zlo, zhi = zvals[:2]
        mol.box.triclinic = False

    mol.box.xlo, mol.box.xhi = xlo, xhi
    mol.box.ylo, mol.box.yhi = ylo, yhi
    mol.box.zlo, mol.box.zhi = zlo, zhi

## POPULATORS
def _populate_atoms_from_dump(f, mol: Microstate, cols: list[str], n_atoms: int) -> None:
    # map column name -> position once per frame, then index directly;
    # avoids reparsing the header for every atom line
    col_idx = {name: i for i, name in enumerate(cols)}
    has_element = "element" in col_idx

    for _ in range(n_atoms):
        values = f.readline().split()
        atom = Atom(
            id=int(values[col_idx["id"]]),
            # atom_type=int(values[col_idx["type"]]),
            x=float(values[col_idx["x"]]),
            y=float(values[col_idx["y"]]),
            z=float(values[col_idx["z"]]),
        )
        if has_element:
            atom.element = values[col_idx["element"]]
        mol.add_atom(atom)

if __name__ == "__main__":
    myslice = subsample_dump("../d1B-r1_TGA_PEEK.dump", nevery=1)
    for i, mol in enumerate(myslice):
        print(mol.timestep)
        if i > 5: break