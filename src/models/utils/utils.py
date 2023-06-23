import importlib


def get_class(class_name, modules):
    """Get class from a string"""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported class: {class_name}")
