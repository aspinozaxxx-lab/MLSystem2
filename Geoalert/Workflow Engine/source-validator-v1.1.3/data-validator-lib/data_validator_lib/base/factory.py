import inspect

from .validator import Validator
from data_validator_lib import validators, errors

# Extracts all the members of validators submodule in a dictionary name: class,
# where names are defined in validators.__init__
# developers must guarantee that these names strictly correspond to that in Dataloader brick 'source_type' params
# This is made to ensure that we must add these nams only once
source_types = dict(inspect.getmembers(validators, inspect.isclass))


def get_validator(source_type: str, **kwargs) -> Validator:
    try:
        validator = source_types[source_type](**kwargs)
    except KeyError:
        raise ModuleNotFoundError(f'The source type {source_type} has no validator in the library.'
                                  f'Available source types: {list(source_types.keys())}')
    else:
        return validator
