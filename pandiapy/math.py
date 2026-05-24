import re
from typing import Union

def string2digit(a: Union[int, float, str]) -> Union[int, float, str]:
    if isinstance(a, (int, float)):
        return a
    try:
        return int(a)
    except ValueError:
        try:
            return float(a)
        except ValueError:
            return a