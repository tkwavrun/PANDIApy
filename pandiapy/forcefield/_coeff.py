from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CoeffList:
    """
    Container for per-line parameter storage of FF parameters.

    Attributes
    ----------
    coeffs  : tuple of coefficient values
    style   : functional form (e.g. 'harmonic', 'morse', 'class2')
    comment : type label string written as an inline comment in the file,
              e.g. '# c  c2  c' for a three-body interaction
    match   : FRC atom-type key that was successfully matched during lookup
    source  : how the match was found — one of:
              'exact' | 'equiv' | 'auto-equiv' | 'wildcard' | 'auto-equiv+wildcard' | 'zero'
    """
    coeffs:  tuple[float, ...]
    style:   str
    comment: str   = ""
    match:   tuple = field(default_factory=tuple)
    source:  str   = ""

    def __repr__(self) -> str:
        return (f"CoeffList(style={self.style!r}, coeffs={self.coeffs}, "
                f"source={self.source!r}, comment={self.comment!r})")


@dataclass
class CoeffResult:
    """
    Intermediate result returned by resolve_* functions.

    Converted to a CoeffList for storage in the PCFF coefficient dicts.
    Not intended for direct use outside of _resolve.py and _assigner.py.

    Attributes
    ----------
    coeffs  : coefficient values
    style   : functional form string
    match   : the FRC key that was actually found (after any equivalence substitution)
    equiv   : the substituted atom-type tuple used for the lookup
              (same as match if source == 'exact')
    source  : lookup step that succeeded
    comment : optional inline comment (default: derived from match at write time)
    """
    coeffs:  tuple[float, ...]
    style:   str
    match:   tuple
    equiv:   tuple
    source:  str
    comment: str = ""

    def to_coeff_list(self) -> CoeffList:
        """Convert to CoeffList for storage."""
        comment = self.comment or '-'.join(str(t) for t in self.match)
        return CoeffList(
            coeffs=self.coeffs,
            style=self.style,
            comment=comment,
            match=self.match,
            source=self.source,
        )
