import math
import re


def pi_converter(value: str | float | int):
    """
    Converts a string like "pi", "2pi", "3 pi" or a number to a multiple of pi.
    """
    if isinstance(value, str):
        value = value.lower().replace(" ", "")
        match = re.match(r"^(-?[+]?[\d\.]*)pi$", value)
        if not match:
            raise ValueError(f"Invalid format for pi conversion: {value}")
        coefficient_str = match.group(1)
        if coefficient_str in ["", "+"]:
            coefficient = 1.0
        elif coefficient_str == "-":
            coefficient = -1.0
        else:
            coefficient = float(coefficient_str)
        return coefficient * math.pi
    else:
        return value * math.pi
