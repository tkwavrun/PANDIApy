from dataclasses import dataclass, field
import numpy as np
from pandiapy.topology.atom import Atom


@dataclass(slots=True)
class Dihedral:
    id: int
    atom1: Atom # first atom in the dihedral
    atom2: Atom # second atom in the dihedral
    atom3: Atom # third atom in the dihedral
    atom4: Atom  # fourth atom in the dihedral
    dihedral_type: int = field(default=None) # type ID in fixed-bond FF
    type_name: str = field(default=None) # type name in fixed-bond FF

    def __str__(self):
        dihedral_label = f"{self.type_name} " if self.type_name else ""
        return f"Dihedral {dihedral_label}between atoms ({self.atom1.id}, {self.atom2.id}, {self.atom3.id}, and {self.atom4.id})"

    @property
    def theta(self) -> float:
        l1 = self.atom2.coords - self.atom1.coords
        l2 = self.atom3.coords - self.atom2.coords
        l3 = self.atom4.coords - self.atom3.coords

        n1 = np.cross(l1, l2); n2 = np.cross(l2, l3)
        l2_norm = l2 / np.linalg.norm(n2)
        m = np.cross(n1, l2_norm)
        a = np.dot(n1, n2); b = np.dot(m, n2)
        return float(np.degrees(np.arctan2(b, a)))

    @property
    def ordered(self) -> tuple: # sorted pair for graphs
        if self.atom4.id < self.atom1.id:
            return self.atom4.id, self.atom3.id, self.atom2.id, self.atom1.id
        else:
            return self.atom1.id, self.atom2.id, self.atom3.id, self.atom4.id