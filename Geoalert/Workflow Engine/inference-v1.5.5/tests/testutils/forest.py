from typing import Tuple, Final
from urban.functional.postprocessing.constants import Shape
from dataclasses import dataclass, field
from .utils import building_shapes_generators
from urban.base.defaults import DEFINITIVE_HEIGHT_TAG


DEFAULT_FOREST_SHAPE: Final[str] = Shape.RECTANGLE
DEFAULT_FOREST_CREATION_PARAMS: Final[dict] = {'width': 10, 'height': 10, 'height_range': (2, 10)}
DEFAULT_FOREST_HEIGHT: Final[float] = 2


@dataclass
class Forest:  # single tree
    center: Tuple[float, float]
    shape_type: str = field(default=Shape.RECTANGLE)
    height: float = field(default=DEFAULT_FOREST_HEIGHT)
    generator_kwargs: dict = field(default_factory=lambda: {k: v for k, v in DEFAULT_FOREST_CREATION_PARAMS})


    def as_feature(self) -> dict:
        return {'geometry': building_shapes_generators[self.shape_type](self.center, 0, **self.generator_kwargs),
                DEFINITIVE_HEIGHT_TAG: self.height}
