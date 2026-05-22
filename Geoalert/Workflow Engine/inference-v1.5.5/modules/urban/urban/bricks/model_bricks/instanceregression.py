from ...base import defaults
from ...functional import io
from ...functional import regression as R
from typing import Tuple, List, Optional, Union, Literal
from .modelbrick import ModelBrick
from pydantic import Field


class InstanceRegression(ModelBrick):
    """Regression model, crops instances by mask (feature) and generates output into feature properties
    Args:
        input_rasters (List[str]): filenames of input bands
        input_vector (str): filename of featurecollection with rooftops
        output_vector (str): filename of resulting featurecollection. If None, same as input_vector
        height_tag: the name of feature property to write the result to
        undefined_result: value to return if model fails
        failsafe: if an instance fails, continues with default height
        meta: Optional[str] path to meta.geojson file with meta angles
        buffer: float buffer around mask geometry
        proposal_height_tag: Optional[str] height tag to use when obtaining crop mask for a building
        default_proposal_height: float default proposal height value
        subtract_neighbours: bool if True subtracts neighbour rooftops from crop mask
        negative_buffer: float buffer for neighbour rooftops
        height_mul: float coefficient to multiply proposal height
        padding_value: int value to fill zero pixels
        method: 'crop_by_mask' or 'add_mask'
        prediction: whether to predict 'shifts' or 'heights'
    """
    input_rasters: List[str]
    input_vector: str
    output_vector: str = Field(default=None)
    sample_size: Tuple[int, int] = Field(default=defaults.DEFAULT_SAMPLE_SIZE)
    height_tag: str = Field(default=defaults.REGR_HEIGHT_TAG)
    tag: str = Field(default='')
    undefined_result: float = Field(default=defaults.UNDEFINED_HEIGHT)
    failsafe: bool = Field(default=True)
    meta: Optional[str] = Field(default=None)
    buffer: float = Field(default=R.DEFAULT_CROP_BUFFER)
    proposal_height_tag: Optional[str] = Field(default=None)
    default_proposal_height: float = Field(default=20)
    subtract_neighbours: bool = Field(default=False)
    negative_buffer: float = Field(default=R.DEFAULT_NEGATIVE_BUFFER)
    height_mul: float = Field(default=R.DEFAULT_HEIGHT_MULTIPLIER)
    padding_value: int = Field(default=R.PADDING_VALUE)
    method: Literal['crop_by_mask', 'add_mask'] = Field(default='crop_by_mask')
    prediction: Literal['shifts', 'heights', 'simple'] = Field(default='heights')

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output_vector = self.output_vector or self.input_vector


    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        fc = io.read_fc(path, self.input_vector)
        if self.meta:
            meta = io.read_fc(path, self.meta)
            angles = {key: meta[0].properties[key] for key in defaults.angle_tags}
        else:
            angles = None

        if fc.crs != bc.crs:
            fc = fc.to_crs(bc.crs)
        if self.prediction == 'heights':
            fc = R.predict_heights_with_regression(bc=bc,
                                                   fc=fc,
                                                   model=self.adapter,
                                                   sample_size=self.sample_size,
                                                   regr_height_tag=self.height_tag,
                                                   undefined_result=self.undefined_result,
                                                   failsafe=self.failsafe,
                                                   angles=angles,
                                                   buffer=self.buffer,
                                                   proposal_height_tag=self.proposal_height_tag,
                                                   default_proposal_height=self.default_proposal_height,
                                                   subtract_neighbours=self.subtract_neighbours,
                                                   negative_buffer=self.negative_buffer,
                                                   height_mul=self.height_mul,
                                                   padding_value=self.padding_value,
                                                   method=self.method)
        elif self.prediction == 'shifts':
            fc = R.predict_fp_shift_with_regression(bc=bc,
                                                    fc=fc,
                                                    model=self.adapter,
                                                    sample_size=self.sample_size,
                                                    failsafe=self.failsafe,
                                                    angles=angles,
                                                    buffer=self.buffer,
                                                    proposal_height_tag=self.proposal_height_tag,
                                                    default_proposal_height=self.default_proposal_height,
                                                    subtract_neighbours=self.subtract_neighbours,
                                                    negative_buffer=self.negative_buffer,
                                                    height_mul=self.height_mul,
                                                    padding_value=self.padding_value,
                                                    method=self.method)

        else:
            fc = R.simple_regression(bc=bc,
                                     fc=fc,
                                     model=self.adapter,
                                     tag=self.tag,
                                     sample_size=self.sample_size,
                                     failsafe=self.failsafe,
                                     buffer=self.buffer,
                                     subtract_neighbours=self.subtract_neighbours,
                                     negative_buffer=self.negative_buffer,
                                     padding_value=self.padding_value,
                                     method=self.method)
        io.save_fc(fc, path, self.output_vector)
