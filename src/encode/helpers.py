def normalize(value:int | float, cap: int | float) -> int | float:
    return min(max(float(value), 0.0), cap) / cap
