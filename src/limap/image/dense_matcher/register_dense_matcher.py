def get_dense_matcher(method: str):
    if method == "roma":
        from .RoMa import RoMa

        return RoMa()
    elif method == "roma_indoor":
        from .RoMa import RoMa

        return RoMa(mode="indoor")
    elif method == "tiny_roma":
        from .RoMa import RoMa

        return RoMa(mode="tiny_outdoor")
    else:
        raise NotImplementedError
