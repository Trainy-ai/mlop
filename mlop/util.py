from typing import Union, Sequence
import json
import logging
import os

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Util"


def to_dict(obj):
    attrs = {}
    for name in dir(obj):
        if not name.startswith("__") and not callable(getattr(obj, name)):
            attrs[name] = getattr(obj, name)
    return attrs


def to_json(data, file):
    if os.path.exists(file):
        with open(file, "r+") as f:
            try:
                content = json.load(f)
                if not isinstance(content, list):
                    logger.error(f"{tag}: file content must be a json list")
                    return
            except json.JSONDecodeError:
                logger.error(f"{tag}: file is not in json format")
                return
            content.extend(data)
            f.seek(0)
            json.dump(content, f, indent=4)
            f.truncate()
    else:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)


def to_human(n):
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return "%.1f%s" % (value, s)
    return "%sB" % n


def dict_to_json(data: dict[str, any]) -> dict:
    for key in list(data):  # avoid RuntimeError if dict size changes
        val = data[key]
        if isinstance(val, dict):
            data[key] = dict_to_json(val)
        else:
            data[key] = val_to_json(val)
    return data


def val_to_json(val: any) -> Union[Sequence, dict]:
    if isinstance(val, (int, float, str, bool)):
        return val
    elif isinstance(val, (list, tuple, range)):
        raise NotImplementedError()  # for files


def get_class(val: any) -> str:
    module_class = val.__class__.__module__ + "." + val.__class__.__name__
    return (
        val.__name__
        if module_class in ["builtins.module", "__builtin__.module"]
        else module_class
    )
