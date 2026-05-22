from ...functional import io
from ...functional import metaangles
from typing import Tuple, List
from .modelbrick import ModelBrick
from pydantic import Field

class MetaAnglesRegression(ModelBrick):
    """Predicts meta angles with model"""
    input_rasters: List[str]
    output_labels: List[str]
    stride: Tuple[int, int] = Field(default=(256, 256))
    sample_size: Tuple[int, int] = Field(default=(256, 256))


    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        fc = metaangles.predict_angles(bc=bc,
                                       model=self._adapter,
                                       sample_size=self.sample_size,
                                       stride=self.stride)
        io.save_fc(fc, path, self.output_vector)
