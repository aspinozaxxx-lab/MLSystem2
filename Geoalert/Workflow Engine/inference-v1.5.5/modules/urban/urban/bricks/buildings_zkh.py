from typing import Optional, Sequence
import shapely
from ..base import Brick
from ..functional import io
from ..functional import mongo
from ..functional.zkh import FLOOR_COUNT_MIN_TAG, FLOOR_COUNT_MAX_TAG, merge_with_zkh_points, merge_with_zkh_points_old,\
    infer_heights_from_floor_count, ZKH_HEIGHT_TAG, correct_class, correct_height, ZKH_TAGS_FOR_FOOTPRINTS
from ..base.defaults import AOI_FILENAME, VECTOR_EXT, BUILDING_CLASS_TAG, DEFINITIVE_HEIGHT_TAG
from gpdadapter import FeatureCollection
from loguru import logger
import os
from pydantic import Field, ConfigDict


class LoadZKHDataLike(Brick):
    """Load features from ZKH database (self-hosted with mongo)

    Args:
        output: name of output GeoJSON feature collection
        aoi: name of input GeoJSON feature collection for bounds
        db: name of database
        collection: name of collection
        uri: uri to connect to database, can be exported as env variable `MONGODB_URI`
    """
    output: str
    _client: mongo.MongoAPI
    aoi: str = Field(AOI_FILENAME)
    db: str = Field('atlas')
    collection: str = Field('reestr')
    uri: Optional[str] = Field(None)

    model_config = ConfigDict(extra = "allow")   # Allow extra fields

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if 'input' in self.model_extra:
            logger.warning("'input' property is deprecated in LoadZKHDataLike, by default it is now bounded by aoi,"
                           " use 'aoi' property to specify another file")
        self._client = mongo.MongoAPI(self.db, self.collection, uri=self.uri)

    def __call__(self, path):
        if os.path.exists(os.path.join(path, self.aoi+VECTOR_EXT)):
            aoi = io.read_fc(path, self.aoi)

            if not aoi.empty:
                bbox = shapely.box(*aoi.total_bounds)
                reestr_fc = self._client.get(bbox)
                # convert points to polygons for correct intersection work
                reestr_fc.buffer(1e-7, inplace=True)

            else:
                logger.warning('AOI file is empty, returning empty ZKH')
                reestr_fc = FeatureCollection()

        else:
            logger.warning('AOI file not found, returning empty ZKH')
            reestr_fc = FeatureCollection()

        logger.debug(f"Got {len(reestr_fc)} features from ZKH MongoDB")
        io.save_fc(reestr_fc, path, self.output)


class MergeWithZKH(Brick):
    """
    !!!DEPRECATED!!!
    Merge properties of features f input feature collection with matched features
    from ZKH feature collection.

    Args:
        input (str): filename of input feature collection
        zkh (str): filename of ZKH feature collection
        correct_heights (bool, optional): if True replace height in ``building_height``
            attribute with ZKH height (num_floors * 3). Defaults to True.
        correct_class (bool, optional): if True set for matched feature class_id=class_value. Defaults to True.
        output (Optional[str], optional): filename of output feature collection (if None same as input).
            Defaults to None.
    """
    input: str
    zkh: str
    correct_heights: bool = Field(True)
    correct_class: bool = Field(True)
    height_tag: str = Field(DEFINITIVE_HEIGHT_TAG)
    class_tag: str = Field(BUILDING_CLASS_TAG)
    class_value: str = Field("101")
    output: Optional[str] = Field(None)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        logger.warning('MergeWithZKH is deprecated, use PropertiesFromZKH, HeightFromZKH or ClassFromZKH')
        self.output = self.output or self.input


    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs="utm")
        zkh_fc = io.read_fc(path, self.zkh, crs=fc.crs)
        fc = merge_with_zkh_points_old(fc, zkh_fc)
        if self.correct_heights:
            fc = correct_height(fc,
                                height_tag=self.height_tag,
                                floor_height=3.0,
                                min_height=3.0,
                                set_min_height=True)

        if self.correct_class:
            fc = correct_class(fc,
                               class_tag=self.class_tag,
                               class_value=self.class_value)
        io.save_fc(fc, path, self.output)


class PropertiesFromZKH(Brick):
    """Merge properties of features of input feature collection with matched features
    from ZKH feature collection.
    Args:
        input (str): filename of input feature collection
        zkh (str): filename of ZKH feature collection
        output (Optional[str], optional): filename of output feature collection (if None same as input).
        tags_to_merge: sequence of property names to merge
    """
    input: str
    zkh: str
    output: Optional[str] = Field(None)
    tags_to_merge: Sequence[str] = Field(ZKH_TAGS_FOR_FOOTPRINTS)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output = self.output or self.input

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs="utm")
        zkh_fc = io.read_fc(path, self.zkh, crs=fc.crs)
        fc = merge_with_zkh_points(fc, zkh_fc, self.tags_to_merge)
        io.save_fc(fc, path, self.output)


class BuildingHeightFromZKH(Brick):
    """Adds property 'zkh_height' to ZKH points collection based on 'floors_count_max' or
    'floors_count_min' properties
    Args:
        input (str): filename of ZKH feature collection
        output (str): filename of output feature collection
        height_tag (str): property name to write inferred height to
        floor_count_tags (Sequence[str]): sequence of tags to take floors number from, in order of relevance
        floor_height (float): floor_height
        min_height (float): min height
    """
    input: str
    output: Optional[str] = Field(None)
    height_tag: str = Field(ZKH_HEIGHT_TAG)
    floor_count_tags: Sequence[str] = Field((FLOOR_COUNT_MAX_TAG, FLOOR_COUNT_MIN_TAG))
    floor_height: float = Field(3.0)
    min_height: float = Field(3.0)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output = self.output or self.input

    def __call__(self, path):
        fc = io.read_fc(path, self.input)
        fc = infer_heights_from_floor_count(fc, self.height_tag, self.floor_count_tags, self.floor_height,
                                            self.min_height)
        io.save_fc(fc, path, self.output)


class BuildingClassFromZKH(Brick):
    """Adds property 'class_id' to ZKH points collection based on 'area_residential' property
    Args:
        input (str): filename of ZKH feature collection
        output (str): filename of output feature collection
        class_tag (str): property name to write inferred height to
        class_value (str):
    """
    input: str
    output: Optional[str] = Field(None)
    class_tag: str = Field(BUILDING_CLASS_TAG)
    class_value: str = Field('101')

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output = self.output or self.input

    def __call__(self, path):
        fc = io.read_fc(path, self.input)
        fc = correct_class(fc, self.class_tag, self.class_value)
        io.save_fc(fc, path, self.output)
