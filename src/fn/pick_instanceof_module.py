def pick_instanceof_module(arr: list, superclass):
    result = []
    if superclass is None:
        return result

    for i in arr:
        if isinstance(i, superclass):
            result.append(i)

    return result
