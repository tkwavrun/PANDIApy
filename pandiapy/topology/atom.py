from dataclasses import dataclass, field, fields
from typing import Optional, Union
import numpy as np

from pandiapy.periodic_table import PERIODIC_TABLE, Element

@dataclass(slots=True)
class Atom:
    id: int = field(default=None)
    x: float = field(default=None)
    y: float = field(default=None)
    z: float = field(default=None)
    q: Optional[float] = field(default=None)
    molecule: Optional[int] = field(default=None)
    element: Optional[Union[str, Element]] = field(default=None)
    type_name: Optional[str] = field(default=None)  # forcefield atom_type if applicable
    atom_type: Optional[int] = field(default=None)  # forcefield atom_type if applicable

    #internal
    _coords: np.ndarray = field(default=None, init=False, repr=False)

    @property
    def coords(self) -> np.ndarray:
        # x, y, z coordinates
        if self._coords is None:
            self._coords = np.array([self.x, self.y, self.z])
        return self._coords

    def __str__(self):
        return f"pandiapy {str(self.element)} Atom (id={self.id})"

    def info(self):
        lines = [f"  {f.name}: {getattr(self, f.name)}" for f in fields(self)]
        return "pandiapy Atom object with\n" + "\n".join(lines)

    def populate_element(self, element_symbol: str) -> None:
        # (re)define atom.element as an element class, or overwrite
        self.element = PERIODIC_TABLE.get(element_symbol)


## Example
if __name__ == "__main__":
    my_atom = Atom(id=1, x=0.0, y=0.0, z=0.0, type_name="hc")
    my_atom.populate_element(element_symbol="H")
    print(my_atom)
