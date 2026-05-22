# Test deserialization and serialization for all the Bricks in CLASS_REGISTRY

from urban.base.registry import CLASS_REGISTRY
from urban.base.registry_object import RegistryObject
from urban.base.brick import Brick
from typing import Any, Union, get_origin, List, Tuple, Sequence, Literal
from random import choice, randint, random
from collections.abc import Sequence as ABCSequence, Iterable as ABCIterable
from copy import deepcopy

# exclude from tests
objects_to_exclude = {'Brick', 'VectorProcessingBrick', 'PolygonProcessingBrick', 'RegistryObject',
                      'BrightnessNormalization', 'MultiThresholding'}

allowed_random_bricks = {'SplitRaster', 'VectorizeMasks', 'NMS'}


def random_field(ant: type) -> Any:
    """Generates random value according to the Field annotation"""
    if ant is type(None):
        return None
    if ant == bool:
        return choice((True, False))
    if ant == str:
        return choice(('foo', 'bar', 'baz'))
    if ant == int:
        return randint(0, 10)
    if ant == float:
        return random()
    if get_origin(ant) is Union:
        return random_field(choice(ant.__args__))
    if ant in {list, tuple}:
        return [random_field(str) for _ in range(3)]

    if get_origin(ant) in {List, Sequence, ABCIterable, ABCSequence, list}:
        t = ant.__args__[0] if ant.__args__ else str
        return [random_field(t) for _ in range(3)]

    if get_origin(ant) in {Tuple, tuple}:
        types = ant.__args__ if ant.__args__ else [str]*3
        return [random_field(t) for t in types]

    if get_origin(ant) == Literal:
        return choice(ant.__args__)

    if ant == Brick:
        return build_yaml_config(choice([CLASS_REGISTRY[i] for i in allowed_random_bricks]))
    if isinstance(ant, type) and issubclass(ant, RegistryObject):
        return build_yaml_config(ant)


def build_yaml_config(cls: type) -> dict:
    """builds dict config for a certain class"""
    config = {'_class': cls.__name__}
    for field_name, info in cls.model_fields.items():
        config[field_name] = random_field(info.annotation)
    return config


def format_nested_dict(d, indent=0):
    """Recursively formats and prints a nested dictionary with sequences as values."""
    result = ""
    if not isinstance(d, (dict, list, tuple)):
        return str(d)
    if isinstance(d, dict):
        for k, v in d.items():
            result += f'\n{"  " * indent}{k}: {format_nested_dict(v, indent + 1)}'
    elif isinstance(d, (list, tuple)):
        for v in d:
            result += f'\n{"  " * indent}{format_nested_dict(v, indent + 1)}'
    return result


def prepare_for_comparsion(d: dict) -> dict:
    """Recursively replaces tuples with lists and removes 'output_...' keys to correctly compare two dicts"""
    res = dict()
    for k, v in d.items():
        # kostyl to avoid 'out' or 'output' fields check, because they are not necessarily equal
        if not k.startswith('out'):
            if isinstance(v, dict):
                res[k] = prepare_for_comparsion(v)
            elif isinstance(v, tuple):
                v = list(v)
            if isinstance(v, list):
                res[k] = [prepare_for_comparsion(i) if isinstance(i, dict) else i for i in v]
            else:
                res[k] = v
    return res


def test_serialization():
    for k in CLASS_REGISTRY:
        if not k.startswith('_') and k not in objects_to_exclude:
            input_config = build_yaml_config(CLASS_REGISTRY[k])
            output_config = CLASS_REGISTRY[k].from_config(deepcopy(input_config)).get_config()
            assert prepare_for_comparsion(input_config) == prepare_for_comparsion(output_config)
