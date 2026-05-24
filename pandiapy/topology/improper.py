from dataclasses import dataclass, field
import numpy as np
from pandiapy.topology.atom import Atom


@dataclass(slots=True)
class Improper:
    id: int
    atom1: Atom
    atom2: Atom
    atom3: Atom
    atom4: Atom
    improper_type: int = field(default=None) # bond_type ID in fixed-bond FF
    type_name: str = field(default=None) # bond_type name in fixed-bond FF

    def __str__(self):
        dihedral_label = f"{list(filter(None, self.type_name))} " if self.type_name else ""
        return f"Angle {dihedral_label}between atoms ({self.atom1.id}, {self.atom2.id}, {self.atom3.id}, and {self.atom4.id})"

    @property
    def theta(self) -> float:
        l1 = self.atom2.coords - self.atom1.coords
        l2 = self.atom3.coords - self.atom1.coords
        l3 = self.atom4.coords - self.atom1.coords

        n1 = np.cross(l1, l2)
        n1 = n1 / np.linalg.norm(n1)
        n2 = l3 / np.linalg.norm(l3)

        return np.degrees(np.arcsin(np.dot(n1, n2)))

    @property
    def ordered(self) -> tuple: # sorted canonical form for graphs
        ids = (self.atom1.id, self.atom2.id, self.atom3.id, self.atom4.id)
        return ids if ids[0] < ids[-1] else ids[::-1]
