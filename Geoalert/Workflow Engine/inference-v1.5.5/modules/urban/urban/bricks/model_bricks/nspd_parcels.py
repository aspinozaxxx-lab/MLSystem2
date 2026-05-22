from typing import Optional, Sequence
from .segmentation import Segmentation
from .modelbrick import ModelBrick
from ...functional import io
from ...functional.utils.to_fc_processor import ToFeatureCollectionProcessor
from pydantic import Field
from loguru import logger

class NSPDParcels(Segmentation):
    """
    Same as segmentation model, but instead of aeronetlib.CollectionProcessor uses ToFeatureCollectionProcessor,
    which vectorize and saves every sample to FeatureCollection

    Args:
        output_fc (str): filename of output FeatureCollection
        simplify: float - simplification param for detected features
        area_threshold: float - features with area less than threshold will be deleted
        score_threshold: float - features with confidence score less than threshold will be deleted
    """
    output_fc: str
    output_labels: Optional[Sequence[str]] = None
    simplify: float = Field(1)
    area_threshold: float = Field(10)
    score_threshold: float = Field(0)

    def model_post_init(self, __context):
        ModelBrick.model_post_init(self, __context)
        self._predictor = ToFeatureCollectionProcessor(input_channels=self.input_rasters,
                                                       processing_fn=self.processing_fn,
                                                       sample_size=self.sample_size,
                                                       bound=self.bounds,
                                                       verbose=self.verbose,
                                                       src_nodata=self.nodata,
                                                       nodata_mask_mode=self.nodata_mask_mode,
                                                       padding=self.padding,
                                                       simplify=self.simplify,
                                                       area_threshold=self.area_threshold,
                                                       score_threshold=self.score_threshold)

    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        if self.crs is not None and self.res is not None:
            bc = self.preprocess(bc)
        fc = self._predictor.process(bc)
        io.save_fc(fc, path=path, name=self.output_fc)

    def processing_fn(self, x):
        x = self.adapter(x)
        for postprocessor in self.postprocessors:
            logger.trace(f'Applying {postprocessor.__class__.__name__}')
            x = postprocessor(x)
            logger.trace(f'shape = {x.shape}, dtype = {x.dtype}')
        return x
