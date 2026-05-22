from ...base import Brick
from ..adapters import ALL_ADAPTER_TYPES
from .postprocess import ALL_POSTPROCESSOR_TYPES
from typing import Optional, Sequence
from pydantic import Field, ConfigDict


class ModelBrick(Brick):
    """Abstract superclass for all those bricks, which process some data using model adapter
    Args:
        adapter: urban.ModelAdapter or it's dict config
        postprocessors (list[Postprocessor]): list of Postprocessor objects to process every sample model returns
        verbose: verbose"""
    adapter: ALL_ADAPTER_TYPES
    postprocessors: Sequence[ALL_POSTPROCESSOR_TYPES] = Field(default_factory=list)
    verbose: bool =  Field(default=False)

    model_config = ConfigDict(discriminator="brick_class")
