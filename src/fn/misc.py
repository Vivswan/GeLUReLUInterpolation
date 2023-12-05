def select_class(main_object, name, class_list):
    value = getattr(main_object, name)
    if value is None and None in class_list:
        setattr(main_object, name, None)
        return None
    if value is None and None not in class_list:
        raise Exception(f"{name} must be in {class_list}")

    for cl in class_list:
        if cl is None:
            continue
        if value == cl:
            setattr(main_object, name, cl)
            return cl
        if isinstance(value, str):
            if value == cl.__name__:
                setattr(main_object, name, cl)
                return cl

    raise Exception(f"{name} must be in {class_list}")


def check(main_obj, first, second):
    if getattr(main_obj, first) is None:
        if getattr(main_obj, second) is not None:
            raise Exception(f'{first}=None then {second} must be None')
    else:
        if getattr(main_obj, second) is None:
            raise Exception(f'{second} must not be None')
