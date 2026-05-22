import os
import datetime as dt
from math import pi
from loguru import logger
from gpdadapter import FeatureCollection
from ..functional.rasterize import rasterize
from typing import Optional, List, Dict, Any, Tuple, Literal
from ..base.brick import Brick, VectorProcessingBrick, PolygonProcessingBrick
from ..functional import io
from ..functional import postprocessing
from ..functional.changes import detect_changes
from ..functional.postprocessing.difference import simple_fc_difference
from functools import partial
from typing import Union, Sequence
from ..functional.nms import nms
from pydantic import Field


class UnifiedVectorProcessing(PolygonProcessingBrick):
    """
    Class which allows to incorporate several vector processing bricks into one.
    As it calls to `process` function of each brick, it makes read-reproject-reproject-write operation only once,
    which can improve performance.

    To replace a set of regular VectorProcessingBricks with UnifiedVectorProcessing, you need to define
    a UnifiedVectorProcessing instance with the same input and output names and a list of bricks to be applied.
    NOTE: crs is defined in the UnifiedVectorProcessing instance,
    and must be the same for all the bricks that are merged.
    """
    bricks: Sequence[Dict[str, Any]]
    _bricks = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._bricks = list()
        for brick_params in self.bricks:
            # so that we would not have to specify input and output for each brick
            # they are not used anyway, but we need to pass them to the constructor
            brick_params.update(input=self.input, output=self.output)
            self._bricks.append(VectorProcessingBrick.from_config(brick_params))

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        for brick in self._bricks:
            logger.info(f"Processing with {brick.__class__.__name__}")
            fc = brick.process(fc)
        return fc

    def get_config(self) -> dict:
        config = super().get_config()
        config['bricks'] = list()
        for brick in self._bricks:
            config['bricks'].append(brick.get_config())
        return config


class Simplify(VectorProcessingBrick):
    """Simplify features in collection with Douglas-Peucker algorithm
    Args:
        input: name of feature collection
        output: name of output feature collection
        crs: projection for simplification
        rate: simplification rate in projection (target crs) units
    """
    crs: Optional[str] = Field(default='utm', description='CRS to reproject before processing')
    rate: float = Field(1.0, gt=0, description='Douglas-Peucker simplification rate in meters')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        return fc.simplify(self.rate)


class Smooth(PolygonProcessingBrick):
    """Smooth the features by a simple angle-cutting algorithm, like in QGIS

    Args:
        input: filename of GeoJSON collection (e.g. predicted features)
        iterations: number of iterations for the algorithm
        offset: a fraction of the edge, which is cut. Bigger offset means bigger radius of smoothing.
            Values vary from 0 to 0.5 however the optimal value is 0.25, which leads the square 
            to converge to a circle
        output: name of output GeoJSON feature collection, if `None` overwrite `input_vector`
    """
    iterations: int = Field(3, gt=0, lt=6, description='Number of iterations for the algorithm, between 1 and 5')  # exponential complexity and file size, so 3 is optimal choice
    offset: float = Field(0.25, gt=0,
                          description='A fraction of the edge, which is cut. Bigger offset means bigger radius of smoothing.')  # from 0 to 0.5

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        return postprocessing.smooth.smooth(fc, self.iterations, self.offset)


class FilterBigObjects(PolygonProcessingBrick):
    """Remove objects which are bigger than specified area
    Args:
        input: filename of feature collection
        max_area_sq_m: if feature area is bigger than `max_area_sq_m` it would be removed. Defaults to None
        area_proportion: used to calculate `max_area_sq_m` if `max_area_sq_m` is None. Defaults to 0.3
        sample_size_from_segm_brick: in pixels, used to calculate `max_area_sq_m` if `max_area_sq_m` is None.
        Defaults to (768, 768)
        output: filename of feature collection for output
        area_tag: str type, if provided area of each feature will be added to its properties, with area_tag key
    """
    max_area_sq_m: Optional[float] = None  # TODO: figure out, how to validate here
    area_proportion: float = Field(0.3, gt=0, le=1, description='used to calculate `max_area_sq_m` if `max_area_sq_m` is None')
    sample_size_from_segm_brick: Tuple[int, int] = (768, 768)  # For SAM model TODO: to a separate brick
    area_tag: Optional[str] = Field(None, description='Property name for feature area')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        # if `max_area_sq_m` is not defined we need to reproject `fc` from `utm`
        # to image crs and calculate `max_area_sq_m`
        # (from pixels to meters) - x_res (meters/pixel) * 768 pixels = 768*x_res meters
        if self.max_area_sq_m is None and '_crs' in fc.columns and '_x_res' in fc.columns and '_y_res' in fc.columns:
            if len(fc) > 0:
                crs = fc[0, '_crs']
                # reproject because we work with objects in input image crs
                fc.to_crs(crs, inplace=True)

                x_res = fc[0, '_x_res']
                y_res = fc[0, '_y_res']

                self.max_area_sq_m = self.area_proportion * x_res * self.sample_size_from_segm_brick[1] \
                                     * y_res * self.sample_size_from_segm_brick[0]
        length = len(fc)
        fc = fc[fc.geometry.area < self.max_area_sq_m]
        logger.trace(f'{length - len(fc)} polygons deleted')
        # add area_tag as property and add area as value
        if self.area_tag:
            fc[:, self.area_tag] = fc.apply(lambda x: round(x.geometry.area))
        return fc


class FilterSmallObjects(PolygonProcessingBrick):
    """Remove objects which are smaller than specified area
    Args:
        input: filename of feature collection
        min_area: if feature area is less than `min_area` it would be removed 
            (all features are reprojected to UTM zone)
        output: filename of feature collection for output
        area_tag: str type, if provided area of each feature will be added to its properties, with area_tag key
    """
    min_area: float
    area_tag: Optional[str] = Field(None, description='Property name for feature area')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        length = len(fc)
        fc = fc[fc.geometry.area > self.min_area]
        logger.trace(f'{length - len(fc)} polygons deleted')
        # add area_tag as property and add area as value
        if self.area_tag:
            fc[:, self.area_tag] = fc.geometry.area
        return fc


class FilterNarrowObjects(PolygonProcessingBrick):
    """
    Filters too narrow polygons which are not likely to represent an object, like a building
    Args:
        input: filename of input feature collection with buildings
        output: filename of output feature collection; if None (default) same as input.
        max_area: If positive max_area is set, geometries larger than this area are not affected
        min_width: if any place of the object is within half of this value from boundary, the feature is excluded
            Defaults to 0.0 (no objects affected)
        min_isoperimetric_quotient: if the object has insufficient area to squared perimeter ratio, it is excluded.
            The parameter means the ratio of the shape area to the area of circle with the same perimeter
            (see https://en.wikipedia.org/wiki/Isoperimetric_inequality#On_a_plane )
            it is 1 for circle and less for any other shape
            (must be from 0 to 1, defaults to 0 which means no requirements)
    """
    max_area: float = Field(0.0, ge=0, description='If positive max_area is set, geometries larger than this area are not affected')
    min_width: float = Field(0.0, ge=0, description='If any place of the object is within half of this value from boundary, the feature is excluded')
    min_isoperimetric_quotient: float = Field(0.0, ge=0, description='The ratio of the shape area to the area of circle with the same perimeter')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        length = len(fc)
        fc.filter(lambda x: (x.geometry.buffer(-self.min_width / 2).area > 0 and
                             x.geometry.area > self.min_isoperimetric_quotient * (x.geometry.length ** 2) / (
                                     4 * pi)), inplace=True)
        logger.trace(f'{length - len(fc)} polygons deleted')
        return fc


class RemoveOverlappingObjects(PolygonProcessingBrick):
    """  TODO: deprecated, use NMS instead
    Remove the features that intersect with another, bigger feature.
    Args:
        input: filename of input feature collection with buildings
        output: filename of output feature collection; if None (default) same as input.
        max_area: If positive max_area is set, geometries larger than this area are not affected
        area_fraction_threshold is in range [0:1],
    represents minimum overlap area to feature area ratio for the feature to be removed
    Default is 1.0 which means that only fully overlapped features are removed.
    """
    max_area: float = Field(0.0, ge=0, description='If positive max_area is set, geometries larger than this area are not affected')
    area_fraction_threshold: float = Field(1.0, ge=0, le=1, description=' minimum overlap area to feature area ratio for the feature to be removed')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        length = len(fc)
        fc = postprocessing.topology.remove_overlapped_features(fc, self.max_area, self.area_fraction_threshold)
        logger.trace(f'{length - len(fc)} polygons deleted')
        return fc


class DissolveSmallObjects(PolygonProcessingBrick):
    """
    Dissolves small objects in geojson, that is attaching the feature to the closest of the 'big' ones.
    If not `flatten`, the dissolved small objects form a multipolygon with the big one,
    otherwise only the properties for the small object are inherited from the big one.
    Args:
        input: filename of feature collection
        min_area: if feature area is less than `min_area` it will be removed
            (all features are reprojected to UTM zone)
        max_distance: if the small objects is farther than this value from all big ones, it is not dissolved
        delete_detached: if True, all small objects that cannot be dissolved will be removed
        flatten: if True, transforms output multipolygons to set of polygons
        output: filename of feature collection for output
    """
    min_area: float = Field(gt=0, description='If feature area is less than `min_area` it will be removed')
    max_distance: float = Field(ge=0, description='If the small objects is farther than this value from all the big ones, it will not be dissolved')
    delete_detached: bool = Field(False, description='If True, all small objects that cannot be dissolved will be removed')
    explode_output: bool = Field(False, alias='flatten')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        filter_ = fc.geometry.area > self.min_area
        fc_big = fc[filter_]
        fc_small = fc[~filter_]

        fc = postprocessing.dissolve.dissolve_fc(fc_small, fc_big, self.max_distance, self.delete_detached)
        return fc


class RemoveSmallHoles(PolygonProcessingBrick):
    """Remove small holes from polygons in feature collection
    
    Args:
        input: filename of GeoJSON collection
        min_hole_area: all holes with area less than `min_hole_area` will be removed
        output: filename of GeoJSON collection for output
    """
    min_hole_area: float = Field(gt=0, description='Area threshold for the holes, in meters')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        fc.map(partial(postprocessing.filters.remove_small_holes, min_hole_area=self.min_hole_area),
               'geometry', inplace=True)
        return fc


class FilterOutput(PolygonProcessingBrick):
    """ TODO: deprecate, it is the same as UnifiedVectorProcessing
    Combines the following procedures in one brick to eliminate multiple read-write:

    - split multipolygons into separate polygons
    - remove holes in polygons
    - remove polygons with area less than specified

    Args:
        input: filename of input feature collection
        output: filename of output feature collection (if None same as input)
        flatten_multipolygons: convert multipolygons to polygons
        min_hole_area: all holes with area less than `min_hole_area` will be removed
        min_poly_area: all the separated in flattening instances with area less than specified will be deleted
    """
    min_hole_area: float = Field(gt=0, description='Area threshold for the holes, in meters')
    explode_output: bool = Field(False, alias='flatten_multipolygons')
    min_poly_area: float = Field(0., gt=0,
                                 description='all the separated in flattening instances with area less than specified will be deleted')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        if self.explode_output:
            fc = postprocessing.flatten_multipolygons.flatten_multipolygons(fc)
        if self.min_hole_area > 0.:
            fc.geometry = fc.geometry.map(partial(postprocessing.filters.remove_small_holes,
                                                  min_hole_area=self.min_hole_area))
        if self.min_poly_area > 0.:
            fc = fc[fc.geometry.area > self.min_poly_area]
        return fc


class MergeByReplace(Brick):
    """Merge (and deduplicate) two feature collections.
    Replace features from `input` by features from `prior` if they match (IoU > replace_threshold)

    Args:
        input: filename of GeoJSON collection (e.g. predicted features)
        prior: filename of GeoJSON collection (e.g. OSM features)
        min_iou: features from fc_1 that have IoU with features from fc_2 above this value will be replaced
        tag: add `tag` to properties (replaced/not replaced)
        value_if_input: value on `tag` attribute if the feature is from 'input' fc
        value_if_prior: value on `tag` attribute if the feature is from 'prior' fc
        prior_attributes: attributes from prior collection to use in replaced features. Default ``None``
        output: filename of output GeoJSON collection
    """
    input: str = Field(pattern=r'^[^\.\\\/]+$', description='Input file name, without extension')
    prior: str = Field(pattern=r'^[^\.\\\/]+$', description='Prior file name, without extension')
    output: Optional[str] = Field(None, pattern=r'^[^\.\\\/]+$',
                                  description='Output file name, without extension')
    min_iou: float = Field(0.7, gt=0, lt=1, description='IoU threshold for replacement')
    tag: Optional[str] = Field(None, description='Property name for a source fc marker')
    value_if_input: Optional[str] = Field(None, description="value on `tag` attribute if the feature is from 'input' fc")
    value_if_prior: Optional[str] = Field(None, alias='value', description="value on `tag` attribute if the feature is from 'prior' fc")
    prior_attributes: Optional[List[str]] = Field(None, description='attributes from prior collection to write to replaced features')

    def __call__(self, path):
        fc_1 = io.read_fc(path, self.input)
        fc_2 = io.read_fc(path, self.prior)

        fc = postprocessing.merge.replace_with_best_suitable(
            fc_1,
            fc_2,
            min_iou=self.min_iou,
            tag=self.tag,
            value_if_input=self.value_if_input,
            value_if_prior=self.value_if_prior,
            attributes=self.prior_attributes,
        )
        io.save_fc(fc, path, self.output)


class Merge(Brick):
    """Joins the vector layers into one, and adds the property corresponding to the object class
    Args:
        input_vectors: list of base filenames, containing input vector data
        output_vector: base filename where the result will be saved
        class_tag: name of the field to be added to the features to represent the input file from which the feature
                   was taken
        class_labels: labels to be added to properties instead of the input_vectors filenames
        subtract: denotes how layers should be stacked.
            From each `key` - name of layer - all the layers in `value` are subtracted before merging.
            For example, if it goes {"Forest": ["Roads", "Buildings"]},
            the brick will cut the forest out for every building and road where they intersect with Forest.
            In case of circular dependency or if some keys do not correspond to input_vectors, error will be raised
    """
    input_vectors: Sequence[str] = Field(description='List of the input file names, without extensions')
    output_vector: str = Field(pattern=r'^[^\.\\\/]+$', description='Output file name, without extension')
    class_tag: Optional[str] = Field("class_id", description='Property name to represent the class')
    class_labels: Optional[Sequence[str]] = Field(None, description='Labels to represent the class')
    subtract: Optional[Dict[str, List[str]]] = Field(None, description='Denotes how layers should be stacked')

    def model_post_init(self, __context):
        if self.class_labels is not None:
            if len(self.class_labels) != len(self.input_vectors):
                raise ValueError('Number of class labels must be equal to the number of input vector layers')
        else:
            self.class_labels = self.input_vectors

        self.subtract = self.validate_subtraction(self.input_vectors, self.subtract)

    @staticmethod
    def validate_subtraction(input_vectors: Sequence[str],
                             subtract: Optional[Dict[str, List[str]]] = None,
                             ):
        if not subtract:
            return {}

        if not set(subtract.keys()).issubset(set(input_vectors)) \
                or not set(sum(subtract.values(), start=[])).issubset(set(input_vectors)):
            raise ValueError("Only vector layers from input_vectors can be listed in subtract param")

        for key, value in subtract.items():
            for subtracted_layer in value:
                if key in subtract.get(subtracted_layer, []):
                    raise ValueError(f"Layers {key} and {subtracted_layer} are subtracted from each other")
        return subtract

    def __call__(self, path):
        all_vectors = [io.read_fc(path, vector) for vector in self.input_vectors]

        res = FeatureCollection()
        for layer, file_name, name in zip(all_vectors, self.input_vectors, self.class_labels):
            for feat_idx in range(len(layer)):
                layer[feat_idx, self.class_tag] = name
            # Subtract all the necessary layers from the current
            for subtracted in self.subtract.get(file_name, []):
                subtracted_layer = all_vectors[self.input_vectors.index(subtracted)]

                layer = simple_fc_difference(layer,
                                             subtracted_layer,
                                             flat_multipolygons=True,
                                             allow_fails=False)
            res.append(layer)

        io.save_fc(res, path, self.output_vector, explode=True)  # save


class RemoveTags(VectorProcessingBrick):
    """Delete specified tags from each feature in feature collection
    Args:
        input: filename of input feature collection
        output: filename of output feature collection (if None - same as input)
        tags: tags to remove
        remove_underscore_tags: remove tags which starts with "_" symbol
    """
    tags: Sequence[str] = Field(default_factory=list, description='tags to remove')
    remove_underscore_tags: bool = Field(False, description='remove tags which starts with "_" symbol')

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        columns_to_drop = list(set(self.tags).intersection(set(fc.columns)))
        if self.remove_underscore_tags:
            columns_to_drop.extend(list(filter(lambda x: x.startswith('_'), fc.columns)))
        return fc.drop(columns_to_drop, axis=1)


class AddAttributes(VectorProcessingBrick):
    """
    Add attributes to the FC feattures
    Args:
        add_timestamp: if True, adds current date to feature attributes, to the "date" field
        add_processing_id: if True, adds current processing ID (from ENV) to "processing_id" field
        attributes: adds custom set of key:value attributes to each feature
    """
    add_timestamp: bool = Field(True)
    add_processing_id: bool = Field(True)
    attributes: Dict[str, str] = Field(default_factory=dict)

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        if self.add_timestamp:
            fc[:, "date"] = str(dt.datetime.now().date())
        if self.add_processing_id:
            fc[:, "processing_id"] = os.getenv("PROCESSING_ID", None)
        for key, value in self.attributes.items():
            fc[:, key] = value
        return fc

class RasterizeLike(Brick):
    """Rasterize feature collection with transform and bounds of given raster band
    Args:
        band: band name, e.g. 'RED' (/path/<band>.geojson)
        input: filename of feature collection, e.g. 'roofs' (/path/<input>.geojson)
        output:  output band filename, e.g. 'roofs' (if None - same as input)
    """
    input: str = Field(pattern=r'^[^\.\\\/]+$', description='Input fc file name, without extension')
    band: str = Field(pattern=r'^[^\.\\\/]+$', description='bc file name to calculate extent and transform, without extension')
    output: Optional[str] = Field(None, description='Output fc file name, without extension')

    def model_post_init(self, __context):
        self.output = self.output or self.input

    def __call__(self, path):
        band = io.read_bc(path, [self.band])[0]
        fc = io.read_fc(path, self.input)
        sample = rasterize(fc, transform=band.transform, out_shape=band.shape, crs=band.crs, name=self.output)
        sample.save(path)  # TODO: shouldn't we use io.save_bc() here to keep the abstraction?


class ChangeDetection(Brick):
    """Compare two feature collections and detect new or disappeared objects
    Args:
        pre: filename of feature collection (pre event)
        post: filename of feature collection (post event)
        output: filename of output feature collection
        mode: one of
            'dis' - detect disappeared objects
            'app' - detect new objects
        mode: one of
            'cover' - compare features by coverage rate. If features from second
                collection cover feature of first collection in total `>=`
                threshold feature is considered as NOT disappeared.
            'iou' - compare features with intersection over union.
                If any of features from second collection has iou `>=` threshold
                feature is considered as NOT disappeared.
            'distance' - compare features with distance between centroids.
                If any feature from second collection has distance `<=`
                threshold to feature from first collection feature is considered
                as NOT disappeared.
        metric:
        threshold: value to compare metric with
    """
    pre: str
    post: str
    output: str
    mode: Literal['dis', 'app'] = Field('dis')
    metric: Literal['cover', 'iou', 'distance'] = Field('cover')
    threshold: float = Field(0.4)

    def __call__(self, path):
        fc_1 = io.read_fc(path, self.pre, crs='utm')
        fc_2 = io.read_fc(path, self.post, crs=fc_1.crs)
        # switch collections if we want to detect new buildings
        if self.mode == 'app':
            fc_1, fc_2 = fc_2, fc_1

        fc = detect_changes(fc_1, fc_2, metric=self.metric, threshold=self.threshold)
        io.save_fc(fc, path, self.output)


class DivideUnpolygonized(Brick):
    """Divide vector file based on simplification attributes

    Args:
        input: filename of feature collection (pre event)
        output_poly: filename of feature collection with simplified features (post event)
        output_nonpoly: filename of feature collection with unsimplified features
    """
    input: str
    output_poly: str
    output_nonpoly: str

    def __call__(self, path):
        fc = io.read_fc(path, self.input_vector)

        poly_idxs = (fc[:, 'shape_type'] != None)  # bool Series

        io.save_fc(fc[poly_idxs], path, self.output_poly)
        io.save_fc(fc[~poly_idxs], path, self.output_nonpoly)


class MergeCloseObjects(PolygonProcessingBrick):
    """
    Merges all the objects, closing small gaps, similarly to binary closing mask operation
    THIS BRICK DISCARDS ALL FEATURE PROPERTIES!
    Args:
        input: filename of feature collection
        max_distance: (meters) if the objects are farther than this value, they are not modified
        output: filename of feature collection for output
    """
    max_distance: float

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        fc = postprocessing.merge_close_objects.merge_close_objects(fc, self.max_distance)
        return fc


class Intersect(Brick):
    """
    Intersect two feature collections where the intersection of features is returned
    Args:
        input_vector1: filename of feature collection
        input_vector2: filename of feature collection
        output_vector: filename of feature collection for output
        flatten: if True, transforms output multipolygons to set of polygons
    """
    input_vector1: str
    input_vector2: str
    output_vector: str
    flatten: bool = Field(True)

    def __call__(self, path):
        input_fc1 = io.read_fc(path, self.input_vector1)
        input_fc2 = io.read_fc(path, self.input_vector2)

        if input_fc1.crs != input_fc2.crs:
            input_fc2 = input_fc2.to_crs(input_fc1.crs)
        out_fc: FeatureCollection = input_fc1.overlay(input_fc2)
        if self.flatten:
            out_fc.explode(inplace=True)
        io.save_fc(out_fc, path, self.output_vector)


class NMS(PolygonProcessingBrick):
    """Non-maximum suppression, deletes features according to the predicate and confidence property
        Args:
            input: filename of feature collection
            output: filename of feature collection for output
            confidence_tag: confidence property name
            predicate: spatial predicate, one of “contains”, “contains_properly”, “covered_by”, “covers”, “crosses”,
                       “intersects”, “overlaps”, “touches”, “within”, “dwithin”
            buffer: buffer to apply before calculating spatial query, in UTM
            corr_coef: float - threshold for each feature relative intersection area (intersection_area / feature_area)
            iou_threshold: float - threshold for a pair of features to consider deleting one of them
    """
    confidence_tag: Union[str, None] = Field('confidence')
    predicate: Literal['contains', 'contains_properly', 'covered_by', 'covers',
    'crosses', 'intersects', 'overlaps', 'touches', 'within', 'dwithin'] = Field('intersects')

    buffer: float = Field(0)
    corr_coef: float = Field(0.0)
    iou_threshold: float = Field(0.0)

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        return nms(fc, self.confidence_tag, self.predicate, self.buffer, self.corr_coef, self.iou_threshold)


class FilterByProperty(Brick):
    """Filters FeatureCollection by some property and saves filtered output
    Args:
        input: filename of feature collection
        output: filename of feature collection for output
        property_tag: property tag to check
        predicate: one of 'equals', 'more_than', 'less_than', 'more_equal_than', 'less_equal_than', 'not_none',
            default is 'not_none'
        value: value to compare with if predicate is in ('equals', 'more_than', 'less_than', 'more_equal_than',
            'less_equal_than')
        return_no_property: 'original' or 'empty' feature collection if `property_tag` is not found
    """
    input: str
    property_tag: str
    value: Any = Field(None)
    output: Optional[str] = Field(None)
    predicate: Literal['equals', 'more_than', 'less_than', 'more_equal_than', 'less_equal_than', 'not_none'] \
        = Field('not_none')
    return_no_property: Literal['original', 'empty'] = Field('original')

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output = self.output or self.input
        if self.predicate != 'not_none' and self.value is None:
            raise ValueError('FilterByProperty `value` must be set if `predicate` is not "not_none"')


    def __call__(self, path):
        fc = io.read_fc(path, self.input)

        if self.property_tag not in fc.columns:
            if self.return_no_property == 'empty':
                logger.warning(f'Property {self.property_tag} not in {fc.columns}. '
                               f'Returning empty FeatureCollection')
                io.save_fc(FeatureCollection(), path, self.output)
            elif self.return_no_property == 'original':
                logger.warning(f'Property {self.property_tag} not in {fc.columns}. '
                               f'Returning original FeatureCollection {self.output}')
                io.save_fc(fc, path, self.output)
            else:
                raise ValueError(f'Unknown value for `return_no_property`: {self.return_no_property}')
            return

        if self.predicate == 'equals':
            fc = fc[fc[:, self.property_tag] == self.value]
        elif self.predicate == 'more_than':
            fc = fc[fc[:, self.property_tag] > self.value]
        elif self.predicate == 'less_than':
            fc = fc[fc[:, self.property_tag] < self.value]
        elif self.predicate == 'more_equal_than':
            fc = fc[fc[:, self.property_tag] >= self.value]
        elif self.predicate == 'less_equal_than':
            fc = fc[fc[:, self.property_tag] <= self.value]
        else:  # 'not_none'
            fc.dropna(subset=self.property_tag, inplace=True)

        io.save_fc(fc, path, self.output)
