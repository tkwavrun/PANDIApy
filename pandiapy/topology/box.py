# Simulation box dimensions

from dataclasses import dataclass, field

@dataclass
class Box:
    xlo: float = field(default=None);  xhi: float = field(default=None)
    ylo: float = field(default=None);  yhi: float = field(default=None)
    zlo: float = field(default=None);  zhi: float = field(default=None)
    xy:  float = 0.0;  xz:  float = 0.0;  yz: float = 0.0
    triclinic: bool = field(default=False)

    @property
    def lengths(self) -> tuple[float, float, float]:
        return self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo

    def __str__(self):
        lx, ly, lz = self.lengths
        return (f"Box: x=[{self.xlo}, {self.xhi}] ({lx:.4f}) "
                f"y=[{self.ylo}, {self.yhi}] ({ly:.4f}) "
                f"z=[{self.zlo}, {self.zhi}] ({lz:.4f})")