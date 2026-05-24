from dataclasses import dataclass, field, fields
import numpy as np
from pandiapy.topology.atom import Atom


@dataclass(slots=True)
class Angle:
    id: int
    atom1: Atom # first atom in the angle
    atom2: Atom # second atom in the angle (vertex)
    atom3: Atom # third atom in the angle
    angle_type: int = field(default=None) # bond_type ID in fixed-bond FF
    type_name: str = field(default=None) # bond_type name in fixed-bond FF

    def __str__(self):
        return f"pandiapy Angle (atoms {self.atom1.id}-{self.atom2.id}-{str(self.atom3)})"

    def info(self):
        lines = [f"  {f.name}: {getattr(self, f.name)}" for f in fields(self)]
        return "pandiapy Angle object with\n" + "\n".join(lines)

    @property
    def theta(self) -> float:
        ba = self.atom1.coords - self.atom2.coords
        bc = self.atom3.coords - self.atom2.coords
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return float(np.arccos(cosine_angle))

    @property
    def ordered(self) -> tuple: # sorted pair for graphs
        if self.atom3.id < self.atom1.id:
            return self.atom3.id, self.atom2.id, self.atom1.id
        else:
            return self.atom1.id, self.atom2.id, self.atom3.id