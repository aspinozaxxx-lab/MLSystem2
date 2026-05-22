from typing import Optional
from ..functional import io
from ..functional import select_definitive_heights as sdh
from ..functional.zkh import ZKH_HEIGHT_TAG
from ..base import Brick, defaults
from pydantic import Field

class DefinitiveHeights(Brick):
    """
    Select best height estimation result from different models
    Args:
        fc: str - input vector file name, must contain fields with height estimations
        output: Optional[str] - output vector file name, if None - same as input
        definitive_height_tag: str - name of resulting field with height
        definitive_height_source_tag: str - name of resulting field with algorithm name
        default_height: float - default height value (if everything else failed)
        area_height_tag: str - name of field with estimation by area
        area_confidence_tag: str - name of field with area estimation confidence score
        regr_height_tag: str - name of field with regression_model result
        emb_height_tag: str - name of field with embedding result
        emb_confidence_tag: str - name of field with embedding model confidence score
        emb_threshold: float - emb_confidence threshold
        sw_height_tag: str - name of field with shadows_walls model result
        sw_confidence_tag: str - name of field with shadows_walls model confidence score
        sw_threshold: float - sw_confidence threshold
        zkh_height_tag: str - zkh_height_tag
    Returns:
        writes {output} file with {definitive_height} property
    """
    fc: str
    output: Optional[str] = Field(None)
    definitive_height_tag: str = Field(defaults.DEFINITIVE_HEIGHT_TAG)
    definitive_height_source_tag: str = Field(defaults.DEFINITIVE_HEIGHT_SOURCE_TAG)
    default_height: float = Field(defaults.DEFAULT_HEIGHT)
    area_height_tag: str = Field(defaults.AREA_HEIGHT_TAG)
    area_confidence_tag: str = Field(defaults.AREA_HEIGHT_CONFIDENCE_TAG)
    regr_height_tag: str = Field(defaults.REGR_HEIGHT_TAG)
    emb_height_tag: str = Field(defaults.EMB_HEIGHT_TAG)
    emb_confidence_tag: str = Field(defaults.EMB_CONFIDENCE_TAG)
    emb_threshold: float = Field(defaults.EMB_CONFIDENCE_THRESHOLD)
    sw_height_tag: str = Field(defaults.SW_HEIGHT_TAG)
    sw_confidence_tag: str = Field(defaults.SW_CONFIDENCE_TAG)
    sw_threshold: float = Field(defaults.SW_CONFIDENCE_THRESHOLD)
    zkh_height_tag: str = Field(ZKH_HEIGHT_TAG)
    building_class_tag: str = Field(defaults.BUILDING_CLASS_TAG)

    def __call__(self, path):
        fc = io.read_fc(path, self.fc)
        fc = sdh.select_definitive_heights(fc=fc,
                                           definitive_height_tag=self.definitive_height_tag,
                                           definitive_height_source_tag=self.definitive_height_source_tag,
                                           default_height=self.default_height,
                                           regr_height_tag=self.regr_height_tag,
                                           emb_height_tag=self.emb_height_tag,
                                           emb_confidence_tag=self.emb_confidence_tag,
                                           emb_threshold=self.emb_threshold,
                                           sw_height_tag=self.sw_height_tag,
                                           sw_confidence_tag=self.sw_confidence_tag,
                                           sw_threshold=self.sw_threshold,
                                           zkh_height_tag=self.zkh_height_tag,
                                           building_class_tag=self.building_class_tag
                                           )
        io.save_fc(fc, path, self.output)
