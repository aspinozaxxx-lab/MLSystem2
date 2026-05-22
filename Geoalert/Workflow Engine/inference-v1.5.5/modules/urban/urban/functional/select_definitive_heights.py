from ..base import defaults
from gpdadapter import FeatureCollection
from .zkh import ZKH_HEIGHT_TAG
import numpy as np
from typing import Final
from loguru import logger

DEFAULT_HEIGHT_LIMIT: Final[tuple] = (4, 999)

limit_height_for_building_class: Final[dict] = {
    '101': (4, 999),
    '102': (4, 7),
    '103': (4, 999),
    '104': (4, 999),
    '105': (4, 999)
}


def select_definitive_heights(
        fc: FeatureCollection,
        definitive_height_tag: str = defaults.DEFINITIVE_HEIGHT_TAG,
        definitive_height_source_tag: str = defaults.DEFINITIVE_HEIGHT_SOURCE_TAG,
        default_height: float = defaults.DEFAULT_HEIGHT,
        area_height_tag: str = defaults.AREA_HEIGHT_TAG,
        area_confidence_tag: str = defaults.AREA_HEIGHT_CONFIDENCE_TAG,
        regr_height_tag: str = defaults.REGR_HEIGHT_TAG,
        emb_height_tag: str = defaults.EMB_HEIGHT_TAG,
        emb_confidence_tag: str = defaults.EMB_CONFIDENCE_TAG,
        emb_threshold: float = defaults.EMB_CONFIDENCE_THRESHOLD,
        sw_height_tag: str = defaults.SW_HEIGHT_TAG,
        sw_confidence_tag: str = defaults.SW_CONFIDENCE_TAG,
        sw_threshold: float = defaults.SW_CONFIDENCE_THRESHOLD,
        zkh_height_tag: str = ZKH_HEIGHT_TAG,
        building_class_tag: str = defaults.BUILDING_CLASS_TAG
) -> FeatureCollection:
    """
    Select best height estimation result from different models
    Args:
        fc: str - input vector file name, must contain fields with height estimations
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
        zkh_height_tag: str - name of field with height from ZKH
        building_class_tag: str - name of field with building class to limit height
    Returns:
        FeatureCollection with {definitive_height} property
    """
    fc[:, definitive_height_tag] = default_height
    fc[:, definitive_height_source_tag] = 'DEFAULT'
    for idx in range(len(fc)):
        try:
            if area_height_tag in fc.columns and area_confidence_tag in fc.columns:
                fc[idx, definitive_height_tag] = fc[idx, area_height_tag]
                fc[idx, definitive_height_source_tag] = 'AREA'
            if regr_height_tag in fc.columns and float(fc[idx, regr_height_tag]) > 2:
                fc[idx, definitive_height_tag] = fc[idx, regr_height_tag]
                fc[idx, definitive_height_source_tag] = 'REGRESSION'
            if emb_height_tag in fc.columns and emb_confidence_tag in fc.columns and \
                    float(fc[idx, emb_confidence_tag]) > emb_threshold:
                fc[idx, definitive_height_tag] = fc[idx, emb_height_tag]
                fc[idx, definitive_height_source_tag] = 'EMBEDDINGS'
            if sw_height_tag in fc.columns and sw_confidence_tag in fc.columns and \
                    float(fc[idx, sw_confidence_tag]) > sw_threshold:
                fc[idx, definitive_height_tag] = fc[idx, sw_height_tag]
                fc[idx, definitive_height_source_tag] = 'SHADOWS_WALLS'
            if building_class_tag in fc.columns:
                limit = limit_height_for_building_class.get(fc[idx, str(building_class_tag)], DEFAULT_HEIGHT_LIMIT)
                if not (limit[0]<=fc[idx, definitive_height_tag]<=limit[1]):
                    fc[idx, definitive_height_tag] = min(limit[1], max(limit[0], fc[idx, definitive_height_tag]))
                    fc[idx, definitive_height_source_tag] = 'LIMIT_BY_CLASS'
            if zkh_height_tag in fc.columns and isinstance(fc[idx, zkh_height_tag], (int, float)) and \
                    not np.isnan(fc[idx, zkh_height_tag]):
                fc[idx, definitive_height_tag] = fc[idx, zkh_height_tag]
                fc[idx, definitive_height_source_tag] = 'ZKH'
        except Exception as e:
            logger.warning(str(e))
            pass
    return fc
