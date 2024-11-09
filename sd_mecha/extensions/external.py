import importlib.metadata


def load_extensions():
    for entry_point in importlib.metadata.entry_points(group="sd_mecha.init"):
        importlib.import_module(entry_point.module)

    for entry_point in importlib.metadata.entry_points(group="sd_mecha.extensions"):
        importlib.import_module(entry_point.module)
