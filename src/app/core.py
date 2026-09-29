def normalize(floats: list[float]) -> list[float]:
    """Normalize a list of floats.

    Args:
        floats (list[float]): A list of floats to normalize.

    Returns:
        list[float]: A list of normalized floats.
    """
    if not floats:
        raise ValueError("Input list cannot be empty.")

    lo, hi = min(floats), max(floats)

    if hi == lo:
        return [0.0] * len(floats)

    normalized = []
    for x in floats:
        normalized.append((x - lo) / (hi - lo))

    return normalized
