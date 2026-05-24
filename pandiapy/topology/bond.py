from dataclasses import dataclass, field, fields
from functools import cache
from typing import Union
import numpy as np

from pandiapy.topology.atom import Atom

@dataclass(slots=True)
class Bond:
    id: int
    atom1: Atom # first atom in the bond
    atom2: Atom # second atom in the bond
    bond_type: int = field(default=None) # type ID in fixed-bond FF
    type_name: str = field(default=None) # type name in fixed-bond FF (ie 'C-C')
    bond_order: Union[str, float] = field(default=None)  # bond order (ie 'single' or '1.5')

    @property
    def vector(self) -> np.ndarray:
        return self.atom2.coords - self.atom1.coords

    @property
    def length(self) -> float:
            return float(np.linalg.norm(self.vector))

    @property
    def ordered(self) -> tuple: # sorted pair for graphs
        if self.atom2.id < self.atom1.id:
            return self.atom2.id, self.atom1.id
        else:
            return self.atom1.id, self.atom2.id

    def __str__(self):
        return f"pandiapy {str(self.bond_order)} Bond (atoms {self.atom1.id} and {self.atom2.id})"

    def info(self):
        lines = [f"  {f.name}: {getattr(self, f.name)}" for f in fields(self)]
        return "pandiapy Bond object with\n" + "\n".join(lines)

    def compute_rdkit_bond_order(self) -> None:
        from rdkit import Chem
        nominal_bond_orders = _bond_order_table()

        el_a = str(self.atom1.element)
        el_b = str(self.atom2.element)
        pair = (el_a, el_b) if (el_a, el_b) in nominal_bond_orders else (el_b, el_a)

        if pair not in nominal_bond_orders:
            self.bond_order = Chem.BondType.SINGLE
            return

        orders = nominal_bond_orders[pair]
        nearest_length = min(orders, key=lambda ref: abs(ref - self.length))
        self.bond_order = orders[nearest_length]

@cache
def _bond_order_table() -> dict:
    # cached definition of rdkit bond types
    from rdkit import Chem
    s, ar, d, t = Chem.BondType.SINGLE, Chem.BondType.AROMATIC, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE
    return {
        ("C", "C"): {1.54: s, 1.39: ar, 1.34: d, 1.20: t},
        ("C", "N"): {1.47: s, 1.27: d, 1.15: t},
        ("C", "O"): {1.43: s, 1.23: d},
        ("O", "O"): {1.48: s, 1.21: d},
    }

## Example
if __name__ == "__main__":
    from pandiapy.periodic_table import PeriodicTable
    pt = PeriodicTable()
    my_atom1 = Atom(id=1, x=0.0, y=0.0, z=0.0, element='C')
    my_atom2 = Atom(id=2, x=1.14, y=0.0, z=0.0, element='H')

    my_bond = Bond(atom1=my_atom1, atom2=my_atom2)

    # my_bond.compute_rdkit_bond_order()
    print(my_bond)
