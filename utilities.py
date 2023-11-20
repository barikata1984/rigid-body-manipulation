from collections.abc import Iterable


def classify_dict_kargs(dict_kargs):
    arr_like = {}
    others = {}

    for k, v in dict_kargs.items():
        if isinstance(v, str):
            others[k] = v
        elif isinstance(v, Iterable):
            arr_like[k] = v
        else:
            others[k] = v

    return arr_like, others

