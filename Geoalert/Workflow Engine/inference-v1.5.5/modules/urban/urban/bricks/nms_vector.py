from typing import Any
from ..base.brick import PolygonProcessingBrick
from gpdadapter import FeatureCollection
from ..functional.nms import nms
from pydantic import Field
from loguru import logger


class NMSVector(PolygonProcessingBrick):
    """Non-maximum suppression
    Probably, redundant, see vector_ops.NMS
    Args:
        input: filename of feature collection, e.g. 'roofs' (/path/<input>.geojson)
        output:  output feature collection filename, e.g. 'roofs' (if None - same as input)
        corr_coef: float - threshold for each feature relative intersection area (intersection_area / feature_area)
        iou_threshold: float - threshold for a pair of features to consider deleting one of them
        score_tag: str - tag for score value
    """
    corr_coef: float = Field(0.75)
    iou_threshold: float =Field(0.75)
    score_tag: str = Field('score', alias='confidence_tag')

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        logger.warning('NMSVector Brick is deprecated, use NMS instead')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        return nms(fc, self.score_tag, 'intersects', 0, self.corr_coef, self.iou_threshold)
