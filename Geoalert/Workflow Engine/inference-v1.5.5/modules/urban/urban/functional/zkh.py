import numpy as np
import pandas as pd
from gpdadapter import FeatureCollection, concatenate
from loguru import logger
from typing import Any, Sequence, Final

FLOOR_HEIGHT = 3.0
HEIGHT_TO_BUFFER_COEF = 0.5
FLOOR_COUNT_MAX_TAG: Final[str] = "floor_count_max"
FLOOR_COUNT_MIN_TAG: Final[str] = "floor_count_min"
IN_ZKH_TAG: Final[str] = "in_zkh"
BUILT_YEAR_TAG: Final[str] = "built_year"
PROJECT_TYPE_TAG: Final[str] = "project_type"
HOUSE_TYPE_TAG: Final[str] = "house_type"
IS_ALARM_TAG: Final[str] = "is_alarm"
QUARTERS_COUNT_TAG: Final[str] = "quarters_count"
LIVING_QUARTERS_COUNT_TAG: Final[str] = "living_quarters_count"
AREA_TOTAL_TAG: Final[str] = "area_total"
AREA_RESIDENTIAL_TAG: Final[str] = "area_residential"
ZKH_HEIGHT_TAG: Final[str] = "zkh_height"

ZKH_TAGS: Final[tuple] = (FLOOR_COUNT_MIN_TAG, FLOOR_COUNT_MAX_TAG, BUILT_YEAR_TAG, PROJECT_TYPE_TAG, HOUSE_TYPE_TAG,
                          IS_ALARM_TAG, QUARTERS_COUNT_TAG, LIVING_QUARTERS_COUNT_TAG, AREA_TOTAL_TAG,
                          AREA_RESIDENTIAL_TAG)

# tags to merge with footprints
ZKH_TAGS_FOR_FOOTPRINTS: Final[tuple] = (FLOOR_COUNT_MIN_TAG, BUILT_YEAR_TAG, PROJECT_TYPE_TAG, HOUSE_TYPE_TAG,
                                         IS_ALARM_TAG, QUARTERS_COUNT_TAG, LIVING_QUARTERS_COUNT_TAG, AREA_TOTAL_TAG,
                                         AREA_RESIDENTIAL_TAG)


def buffer_according_to_height(
    fc: FeatureCollection,
    height_buffer_coef: float = 0.5,
    floor_height: float = 3.0,
) -> FeatureCollection:
    """Buffer geometry (ZKH point) according to height of building specified in properties"""

    def _buffer(f: pd.Series):
        if FLOOR_COUNT_MAX_TAG in f.index:
            floors_num = f[FLOOR_COUNT_MAX_TAG]
        elif FLOOR_COUNT_MIN_TAG in f.index:
            floors_num = f[FLOOR_COUNT_MIN_TAG]
        else:
            floors_num = 1
        if not isinstance(floors_num, (int, float)) or floors_num < 1:
            floors_num = 1
        buffer = floors_num * floor_height * height_buffer_coef
        return f['geometry'].buffer(buffer if not np.isnan(buffer) else 0)

    fc[:, 'geometry'] = fc.apply(_buffer)
    return fc


def correct_height(fc: FeatureCollection,
                   height_tag: str = ZKH_HEIGHT_TAG,
                   floor_height: float = 3.0,
                   min_height: float = 3.0,
                   set_min_height: bool = True):
    # TODO: DEPRECATED
    """Replace feature `building_height` with another one calculated by ZKH properties"""
    if FLOOR_COUNT_MAX_TAG not in fc.columns:
        logger.warning(f'{FLOOR_COUNT_MAX_TAG} not in {fc.columns}')
        return fc
    if height_tag not in fc.columns:
        fc[:, height_tag] = pd.Series(dtype=float)
    for idx in range(len(fc)):
        num_floors = fc[idx, FLOOR_COUNT_MAX_TAG]
        if not np.isnan(num_floors) and num_floors > 0:
            zkh_height = num_floors * floor_height
            fc[idx, height_tag] = zkh_height
    return fc


def correct_class(fc: FeatureCollection,
                  class_tag: str = "class_id",
                  class_value: str = "101"):
    """Set class for ZKH features to 'apartments' (class_id = 101)"""
    if LIVING_QUARTERS_COUNT_TAG not in fc.columns or AREA_RESIDENTIAL_TAG not in fc.columns:
        logger.warning(f'{LIVING_QUARTERS_COUNT_TAG} or {AREA_RESIDENTIAL_TAG} not in {fc.columns}')
        return fc
    for idx in range(len(fc)):
        living_quarters = fc[idx, LIVING_QUARTERS_COUNT_TAG]
        area_residential = fc[idx, AREA_RESIDENTIAL_TAG]
        if living_quarters and area_residential and living_quarters > 0 and area_residential > 0:
            fc[idx, class_tag] = class_value
    return fc


def reduce_group(fc: FeatureCollection) -> FeatureCollection:
    """Given group from groupby() where all the geometries are supposed to be the same (single building) but
    ZKH properties are taken from different points, reduces it into a single feature"""
    if fc.empty or (fc[:, IN_ZKH_TAG] == 0).all():
        return fc[0]

    def reduce_property(fc: FeatureCollection, tag: str, how: str = 'sum', default: Any = None):
        # how must be one of (min, max, sum, mode todo: mean, median)
        if tag not in fc.columns:
            return default
        if how == 'sum':
            return fc[:, tag].sum()
        if how == 'min':
            return fc[:, tag].min()
        if how == 'max':
            return fc[:, tag].max()
        if how == 'mode':
            mode = fc[:, tag].mode()  # mode is always pd.Series, maybe empty
            return mode.iloc[0] if len(mode) > 0 else default
        return default

    for tag in (FLOOR_COUNT_MAX_TAG, ZKH_HEIGHT_TAG):
        fc[0, tag] = reduce_property(fc, tag, 'max')
    for tag in (FLOOR_COUNT_MIN_TAG, ):
        fc[0, tag] = reduce_property(fc, tag, 'min')
    for tag in (QUARTERS_COUNT_TAG, LIVING_QUARTERS_COUNT_TAG, AREA_TOTAL_TAG, AREA_RESIDENTIAL_TAG):
        fc[0, tag] = reduce_property(fc, tag, 'sum')
    for tag in (PROJECT_TYPE_TAG, BUILT_YEAR_TAG, HOUSE_TYPE_TAG):
        fc[0, tag] = reduce_property(fc, tag, 'mode')
    return fc[0]


def merge_with_zkh_points(input_fc: FeatureCollection, properties_fc: FeatureCollection,
                          tags_to_merge: Sequence[str] = ZKH_TAGS_FOR_FOOTPRINTS):
    """Merge buildings features and zkh points with attributes. 
    Aggregate attributes for `many points -> one feature` case.
    Merge with best feature by area and distance attributes for `one point -> many features`
    """
    if input_fc.empty or properties_fc.empty:
        return input_fc
    if input_fc.crs != properties_fc.crs:
        properties_fc.to_crs(input_fc.crs, inplace=True)

    # drop all columns from zkh except for tags_to_merge and geometry
    properties_fc.drop(list(set(properties_fc.columns).difference(set(list(tags_to_merge) + ['geometry']))),
                       axis=1, inplace=True)
    properties_fc[:, IN_ZKH_TAG] = 1

    # drop tags_to_merge from input (we rewrite them)
    input_fc.drop(list(tags_to_merge)+[IN_ZKH_TAG], axis=1, inplace=True)
    original_columns = set(input_fc.columns)

    joined = input_fc.sjoin(properties_fc, how='left')
    input_fc = concatenate([reduce_group(group) for group in joined.groupby(by=joined.index).values()])

    input_fc[:, IN_ZKH_TAG] = input_fc[:, IN_ZKH_TAG].fillna(value=0)  # replace NONE values with 0

    columns_to_drop = set(input_fc.columns).difference(original_columns.union(set(list(tags_to_merge)+[IN_ZKH_TAG])))
    input_fc.drop(list(columns_to_drop), axis=1, inplace=True)
    return input_fc


def merge_with_zkh_points_old(input_fc: FeatureCollection, properties_fc: FeatureCollection):
    """
    !!!DEPRECATED!!!
    Merge buildings features and zkh points with attributes.
    Aggregate attributes for `many points -> one feature` case.
    Merge with best feature by area and distance attributes for `one point -> many features`
    """
    if input_fc.empty or properties_fc.empty:
        return input_fc
    assert input_fc.crs == properties_fc.crs
    properties_fc[:, IN_ZKH_TAG] = 1
    joined = input_fc.sjoin(buffer_according_to_height(properties_fc,
                                                       height_buffer_coef=HEIGHT_TO_BUFFER_COEF,
                                                       floor_height=FLOOR_HEIGHT), how='left')

    return concatenate([reduce_group(group) for group in joined.groupby(by=joined.index).values()])


def infer_heights_from_floor_count(fc: FeatureCollection,
                                   height_tag: str = ZKH_HEIGHT_TAG,
                                   floor_count_tags: Sequence[str] = (FLOOR_COUNT_MAX_TAG, FLOOR_COUNT_MIN_TAG),
                                   floor_height: float = 3.0,
                                   min_height: float = 3.0) -> FeatureCollection:
    """Adds property 'building_height' to ZKH points collection based on 'floors_count_max' or
    'floors_count_min' properties
    Args:
        fc (FeatureCollection): FeatureCollection
        height_tag (str): property name to write inferred height to
        floor_count_tags (Sequence[str]): sequence of tags to take floors number from, in order of relevance
        floor_height (float): floor_height
        min_height (float): min height
    """
    fc[:, height_tag] = pd.Series(dtype=float)
    for idx in range(len(fc)):
        for tag in floor_count_tags[::-1]:
            try:
                fc[idx, height_tag] = int(fc[idx, tag]) * floor_height
            except:
                continue
        if not isinstance(fc[idx, height_tag], (float, int)) or fc[idx, height_tag] < min_height:
            fc[idx, height_tag] = min_height
    return fc
