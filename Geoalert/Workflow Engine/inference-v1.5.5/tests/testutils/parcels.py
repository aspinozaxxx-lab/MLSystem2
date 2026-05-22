from typing import Tuple, Final
from urban.base.defaults import PARCEL_CONFIDENCE_TAG
from urban.functional.postprocessing.constants import Shape, Tag
from dataclasses import dataclass, field
from .utils import building_shapes_generators


DEFAULT_PARCEL_SHAPE: Final[str] = Shape.RECTANGLE
DEFAULT_PARCEL_CREATION_PARAMS: Final[dict] = {'width': 10, 'height': 10}



@dataclass
class Parcel:
    center: Tuple[float, float]
    shape_type: str = field(default=Shape.RECTANGLE)
    score: float = field(default=1)
    generator_kwargs: dict = field(default_factory=lambda: {k: v for k, v in DEFAULT_PARCEL_CREATION_PARAMS})

    def as_feature(self) -> dict:
        return {'geometry': building_shapes_generators[self.shape_type](self.center, 0, **self.generator_kwargs),
                PARCEL_CONFIDENCE_TAG: self.score}
