import numpy as np


def total_lstsq(regressors: np.ndarray, wrenches: np.ndarray) -> np.ndarray:
    """Solve unweighted linear TLS by SVD of [A, b], without column scaling.

    Inputs must be finite, with A shaped (m, n), b shaped (m,), and m > n.
    Return the parameter vector; reject problems without a unique finite solution.
    """
    a = np.asarray(regressors, dtype=float)
    b = np.asarray(wrenches, dtype=float)
    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.size or not 0 < a.shape[1] < a.shape[0]:
        raise ValueError("TLS requires A shaped (m, n) and b shaped (m,), with m > n > 0")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("TLS inputs must be finite")

    augmented = np.column_stack((a, b))
    _, singular_values, vh = np.linalg.svd(augmented, full_matrices=False)
    tolerance = np.finfo(float).eps * max(augmented.shape)
    # A strict singular-value gap is the existence/uniqueness condition for linear TLS.
    smallest_a = np.linalg.svd(a, compute_uv=False)[-1]
    if smallest_a - singular_values[-1] <= tolerance * singular_values[0] or abs(vh[-1, -1]) <= tolerance:
        raise ValueError("TLS has no numerically unique finite solution")
    return -vh[-1, :-1] / vh[-1, -1]
