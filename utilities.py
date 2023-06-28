import numpy as np
from collections.abc import Sequence


def classify_dict_kargs(dict_kargs):
    arr_like = {}
    others = {}

    for k, v in dict_kargs.items():
        if isinstance(v, str):
            others[k] = v
        elif isinstance(v, (Sequence, np.ndarray)):
            arr_like[k] = v
        else:
            others[k] = v

    return arr_like, others


def store(
        data: np.ndarray,
        destination: np.ndarray) -> np.ndarray:

    return np.append(destination, data[np.newaxis, :], axis=0)


def check_and_expand(
        arr: np.ndarray,
        dim: int) -> np.ndarray:

    return np.expand_dims(arr, axis=0) if arr.ndim < dim else arr 
