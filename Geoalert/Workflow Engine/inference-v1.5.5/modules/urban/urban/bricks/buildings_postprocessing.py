import shapely
from gpdadapter import FeatureCollection
from loguru import logger
from typing import Optional, List, Mapping, Any, Literal, Dict, Hashable, Sequence, Union
from ..base.brick import Brick, VectorProcessingBrick
from ..functional import io
from ..functional import josm
from ..functional import postprocessing
from ..functional.postprocessing.constants import Shape
from ..functional.osm import map_osm_landuse_
from pydantic import Field, BaseModel, ConfigDict


class SimplificationParams(BaseModel):
    shape_type: str
    min_iou: float = Field(0.8)
    max_hausdorff: float = Field(10.)
    min_area: float = Field(0.)
    func_kwargs: dict = Field(default_factory=dict)


class SimplifyAsShapes(VectorProcessingBrick):
    """
    One-brick replacement for a sequence of SimplifyAsShape bricks
    Is more efficient because it doesn't need to read and write the same file multiple times
    todo: add single-pass featureCollection simplification in functional.postprocessing.simplify_fc_with_shapes
    """
    # read_params
    make_valid_input: bool  = Field(True)
    dropna_input: bool  = Field(True)
    drop_empty_input: bool  = Field(True)
    explode_input: bool  = Field(True)
    remove_repeated_points_input: bool  = Field(True)
    input_geom_types: Union[str, Sequence[str], None] = 'Polygon'

    # write params
    make_valid_output: bool  = Field(True)
    dropna_output: bool  = Field(True)
    drop_empty_output: bool  = Field(True)
    explode_output: bool  = Field(True)
    remove_repeated_points_output: bool = Field(True)
    output_geom_types: Union[str, Sequence[str], None] = 'Polygon'
    crs: str = Field('utm')

    simplification_params: Sequence[Mapping[str, Any]]
    _simplification_params = None
    iou_confidence_tag: Optional[str] = None

    model_config = ConfigDict(extra = "allow")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._simplification_params = list()
        for params_dict in self.simplification_params:
            kwargs = {k: v for k, v in params_dict.items() if k in SimplificationParams.model_fields}
            func_kwargs = {k: v for k, v in params_dict.items() if k not in SimplificationParams.model_fields}
            self._simplification_params.append(SimplificationParams(**kwargs, func_kwargs=func_kwargs))

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        # simplify collection
        for stage in self._simplification_params:
            logger.info(f"{stage.shape_type} simplification")
            fc = postprocessing.simplify_fc_with_shapes(
                fc,
                shape_type=stage.shape_type,
                min_iou=stage.min_iou,
                max_hausdorff=stage.max_hausdorff,
                min_area=stage.min_area,
                iou_confidence_tag=self.iou_confidence_tag,
                func_params=stage.func_kwargs
            )
        return fc


class SimplifyAsShape(VectorProcessingBrick):
    """Simplify geometries in feature collection with simple shapes like circle, rectangle, etc.

    Args:
        input: name of feature collection
        shape_type: one of ["CIRCLE", "RECTANGLE", "L-SHAPE", "GRID_SNAP"]
        min_iou: minimum intersection over union value to simplify feature
        max_hausdorff: minimum hausdorff value to simplify feature
        min_area: if feature area less than `min_area` feature will be replaced with simplified anyway
        verbose: if `True` progress bar is shown
        output: name of output feature collection
        iou_confidence_tag: if not empty/None, uses this value as a tag to write iou between simplified and original
            shape to the output featurecollection. You can use, for example, "simplification_confidence"
            Actually, we could use '_simplification_iou' but it will be removed as underscore by default,
            and can't be changed in config
        kwargs: simplification function keyword arguments.
                Currently, are supported for GRID_SNAP function:
                    `min_values` - float or tuple, min value of radius on the grid for each angle
                    `radius range` - max grid (geometry) size, float
                Currently are supported for RECTANGLE function:
                    `rect_neg_buffer` - negative buffer for rectangle simplification
                    `rect_min_hole_area` - min hole area for rectangle simplification
    """
    # read_params
    make_valid_input: bool  = Field(True)
    dropna_input: bool  = Field(True)
    drop_empty_input: bool  = Field(True)
    explode_input: bool  = Field(True)
    remove_repeated_points_input: bool  = Field(True)
    input_geom_types: Union[str, Sequence[str], None] = 'Polygon'

    # write params
    make_valid_output: bool  = Field(True)
    dropna_output: bool  = Field(True)
    drop_empty_output: bool  = Field(True)
    explode_output: bool  = Field(True)
    remove_repeated_points_output: bool = Field(True)
    output_geom_types: Union[str, Sequence[str], None] = 'Polygon'
    crs: str = Field('utm')

    shape_type: Literal["CIRCLE", "RECTANGLE", "L-SHAPE", "GRID_SNAP", "DYNAMIC_GRID"]
    min_iou: float = Field(0.8)
    max_hausdorff: float = Field(10.)
    min_area: float = Field(0.)
    iou_confidence_tag: Optional[str] = Field(None)

    model_config = ConfigDict(extra="allow")  # Allow extra fields

    def get_config(self):
        config = super().get_config()
        config.update(self.model_extra)
        return config

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        fc = postprocessing.simplify_fc_with_shapes(
            fc,
            shape_type=self.shape_type,
            min_iou=self.min_iou,
            max_hausdorff=self.max_hausdorff,
            min_area=self.min_area,
            iou_confidence_tag=self.iou_confidence_tag,
            func_params=self.model_extra
        )
        return fc


class CorrectTopology(VectorProcessingBrick):
    """Remove intersection between closely standing buildings by moving their edges in normal direction

    Args:
        input (str): filename of input feature collection with buildings
        distance_step (float, optional): step of edge shift. Defaults to 1.0. If 0 or less,
            the edge shift is skipped
        output (Optional[str], optional): filename of output feature collection (if None same as input).
            Defaults to None.
        correct_by_subtraction (int, optional): remove intersection by subtraction of one polygon from another.
            If 0, no subtraction
            If 1, smaller polygons are subtracted from bigger
            If -1, bigger polygons are subtracted from smaller
        buffer (float, optional): if positive, the smaller geometries are buffered
            when subtracting to make a gap between resulting geometries
    """
    make_valid_input: bool  = Field(True)
    dropna_input: bool  = Field(True)
    drop_empty_input: bool  = Field(True)
    explode_input: bool  = Field(True)
    remove_repeated_points_input: bool  = Field(True)
    input_geom_types: Union[str, Sequence[str], None] = 'Polygon'

    # write params
    make_valid_output: bool  = Field(True)
    dropna_output: bool  = Field(True)
    drop_empty_output: bool  = Field(True)
    explode_output: bool  = Field(True)
    remove_repeated_points_output: bool = Field(True)
    output_geom_types: Union[str, Sequence[str], None] = 'Polygon'
    crs: str = Field('utm')

    distance_step: float = Field(1.0)
    correct_by_subtraction: int =  Field(0)
    buffer: float =  Field(0.0)


    def process(self, fc: FeatureCollection) -> FeatureCollection:
        # replace features with prior
        if self.distance_step > 0:
            fc = postprocessing.topology.correct_topology_by_edge_shift(
                fc, distance_step=self.distance_step
            )

        # this applies only to the features which were not corrected by edge shift
        if self.correct_by_subtraction in {-1, 1}:
            fc = postprocessing.topology.correct_topology_by_subtraction(fc,
                                                                         buffer=self.buffer,
                                                                         direction=self.correct_by_subtraction)
        return fc


class AlignBuildings(Brick):
    """Align small buildings to the nearest roads and also align in clusters

    Args:
        input (str): filename of input feature collection with buildings
        roads (str): filename of input feature collection with roads (linestrings)
        max_building_area (float, optional): align only buildings with area less than specified.
            Defaults to 600.
        cluster_distance (float, optional): merge buildings in clusters if they are as close
            as cluster_distance. Defaults to 5.
        shape_types (Tuple[str], optional): it is possible to align only rectangular shapes.
            Defaults to (RECTANGLE, L-SHAPE, GRID_SNAP).
        output (Optional[str], optional): filename of output feature collection (if None same as input).
            Defaults to None.
    """
    input: str
    roads: str
    output: Optional[str] = Field(None)

    max_building_area: float = Field(600.)
    cluster_distance: float = Field(5.)
    shape_types: Sequence[str] = Field((Shape.RECTANGLE, Shape.LSHAPE, Shape.GRID_SNAP))

    def model_post_init(self, __context):
        self.output = self.output or self.input


    def __call__(self, path):
        buildings_fc = io.read_fc(path, self.input, crs='utm', explode=True,
                                  keep_only_geometry_types=shapely.Polygon)
        roads_fc = io.read_fc(path, self.roads, explode=True,
                              keep_only_geometry_types=shapely.LineString).to_crs(buildings_fc.crs)
        buildings_fc = postprocessing.aligning.align_buildings(
            buildings_fc,
            roads_fc,
            max_area=self.max_building_area,
            cluster_distance=self.cluster_distance,
            shape_types=self.shape_types,
        )
        io.save_fc(buildings_fc, path, self.output, keep_only_geometry_types=shapely.Polygon)


class SplitByRoads(Brick):
    """Split building instances with road lines, preserving from emerging of small and thin objects.
    Every split part is checked for the criteria (area, compactness and rectangle ratio thresholds) and if any of the
    criteria is met (too small, to scattered or too elongated object), it is not returned as a separate instance.
    Instead, it is merged back to the nearest `good` polygon, which does not meet the criteria.
    If there is ambiguity (there are more than one `good` object to be merged), the `bad` object is removed.

    If all the thresholds are zero or less, the merge is not applied

    Args:
        input: name of input feature collection
        roads: name of roads feature collection
        roads_thickness: buffer for road lines
        flat_multipolygons: if True, convert resulted MultiPolygons to Polygons with same properties
        verbose: if True progress bar is enabled
        output: name of output feature collection (if None - same as input)
        failsafe: if True, any features that fail to be cut/merged are left `as is` with a warning,
        and the processing is not interrupted
        area_threshold: minimum area for an object to be left in the output. Smaller objects are merged or removed
        compactness_threshold: minimum Richardson compactness for an object to be left in the output.
                                      More complex objects are merged or removed
        rectangle_ratio_threshold: maximum ratio of the rotated bounding rectangle edges for
                    an object to be left in the output
    """
    input: str
    roads: str
    output: Optional[str] = Field(None)
    roads_thickness: float = Field(1.)
    flat_multipolygons: bool = Field(True)
    area_threshold: float = Field(25)
    compactness_threshold: float = Field(0.20)
    rectangle_ratio_threshold: float = Field(5)
    failsafe: bool = Field(True)
    verbose: bool = Field(False)

    def model_post_init(self, __context):
        self.output = self.output or self.input

    def __call__(self, path):
        # read fc
        buildings_fc = io.read_fc(path, self.input, crs='utm', explode=True,
                                  keep_only_geometry_types=shapely.Polygon)
        roads_fc = io.read_fc(path, self.roads, explode=True,
                              keep_only_geometry_types=shapely.LineString).to_crs(buildings_fc.crs)

        # prepare (reproject and buffer)
        roads_fc.buffer(self.roads_thickness, resolution=1, inplace=True)

        # subtract roads from buildings
        buildings_fc = postprocessing.difference.fc_difference(
            buildings_fc, roads_fc, verbose=self.verbose, flat_multipolygons=self.flat_multipolygons,
            allow_fails=self.failsafe,
            area_threshold=self.area_threshold,
            compactness_threshold=self.compactness_threshold,
            rectangle_ratio_threshold=self.rectangle_ratio_threshold
        )

        # save results
        io.save_fc(buildings_fc, path, self.output, explode=True, keep_only_geometry_types=shapely.Polygon)


class CorrectClassesWithOSM(Brick):
    """Correct classes generated by model by OSM Landuse KEY

    Args:
        input: name of input FeatureCollection
        landuse: name of landuse FeatureCollection
        class_map: dict mapping necessary classes to OSM classes 
            (e.g. {101: ["residential", "houses"], 103: ["commercial]"})
        class_tag: attribute name where to write corrected classes from OSM
        output: name of output FeatureCollection (if None - same as input)

    Example:
        .. code:: python

            class_map = {
                "residential": [
                    "residential"
                ],
                "non-residential": [
                    "commercial",
                    "industrial",
                    "retail",
                    "construction",
                    "allotments",
                    "cemetery",
                    "military",
                    "port",
                    "quarry",
                    "railway",
                    "religious",
                    "salt_pond",
                    "harbour",
                ],
            }

    """
    input: str
    landuse: str
    output: Optional[str] = Field(None)
    class_tag: str
    class_map: Dict[Hashable, List[Hashable]] = Field(default_factory=dict)

    def model_post_init(self, __context):
        self.output = self.output or self.input
    
    def __call__(self, path):
        fc = io.read_fc(path, self.input)
        landuse = io.read_fc(path, self.landuse)
        fc = map_osm_landuse_(fc, landuse, self.class_map, tag=self.class_tag)
        io.save_fc(fc, path, self.output)


class InstanceSeparation(Brick):
    """Separates the instances that stick together in semantic segmentation output
    using the proposed instance centroids

    Args:
        input: filename of input feature collection with buildings
        markers: filename of feature collection with blobs indicating the predicted instance centroids
        output: filename of output feature collection with buildings (if None same as input)
        max_instance_area: if the average instance area within the input feature exceeds this value, this feature
            is kept as is
        ids_tag: tag name for split instances
        flat_multipolygons: if true, the resulting multipolygons will be separated to polygons
        min_instance_area: all the separated in flattening instances with area less than specified will be deleted
    """
    input: str
    markers: str
    output: Optional[str] = Field(None)
    max_instance_area: float = Field(300.)
    ids_tag: str = Field("_block_id")
    flat_multipolygons: bool = Field(True)
    min_instance_area: float = Field(0.)

    def model_post_init(self, __context):
        self.output = self.output or self.input
        
    def __call__(self, path):
        buildings_fc: FeatureCollection = io.read_fc(path, self.input, "utm", explode=True,
                                                     keep_only_geometry_types=shapely.Polygon)
        markers_fc: FeatureCollection = io.read_fc(path, self.markers, "utm", explode=True,
                                                   keep_only_geometry_types=shapely.Polygon)

        # set id tag for features before instance separation
        buildings_fc[:, self.ids_tag] = range(len(buildings_fc))

        split_fc = postprocessing.instance_separation.split_instances(
            buildings_fc, markers_fc, max_average_area=self.max_instance_area,
            flat_multipolygons=self.flat_multipolygons, min_area=self.min_instance_area
        )
        io.save_fc(split_fc, path, self.output, explode=True, keep_only_geometry_types=shapely.Polygon)


class SimplifyWithJOSM(VectorProcessingBrick):
    """Simplify features (orthogonalize) with third-party service - JOSM
    
    Note:
        should be used together with ~urban.InstanceSeparation and ``ids_tag`` 
        should be the same for both steps

    Args:
        input: filename of feature collection 
        output: filename of feature collection (if None same as input)
        endpoint_url: url of service to call (service is called with 
            response = requests.post(
                    endpoint_url, 
                    headers={'cache-control': "no-cache"}, 
                    files={"file": data}, 
            )
        ids_tag: tag to group features for metrics measuring (useful in case of terrace buildings 
            to compare whole blocks, not separate small buildings)
        iou_threshold: iou threshold to replace building by simplified one 
            (should be satisfied along with Hausdorff distance)
        hausdorff_threshold: hausdorff distance threshold to replace building by simplified one 
            (should be satisfied along with IoU)
        failsafe: if True, the stage will not fail in case of errors at the JOSM service,
            but the features will be returned as is with a warning
        timeout: time in seconds to wait until the JOSM endpoint response

    """
    make_valid_input: bool  = Field(True)
    dropna_input: bool  = Field(True)
    drop_empty_input: bool  = Field(True)
    explode_input: bool  = Field(True)
    remove_repeated_points_input: bool  = Field(True)
    input_geom_types: Union[str, Sequence[str], None] = 'Polygon'

    # write params
    make_valid_output: bool  = Field(True)
    dropna_output: bool  = Field(True)
    drop_empty_output: bool  = Field(True)
    explode_output: bool  = Field(True)
    remove_repeated_points_output: bool = Field(True)
    output_geom_types: Union[str, Sequence[str], None] = 'Polygon'
    crs: str = Field('utm')

    endpoint_url: str
    ids_tag: str = Field("_block_id")
    iou_threshold: float = Field(0.9)
    hausdorff_threshold: float = Field(5.0)
    timeout: int = Field(300)
    failsafe: bool = Field(False)

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        try:
            josm_fc = josm.process_fc_with_josm(fc, self.endpoint_url, timeout=self.timeout)  # 5 min timeout
        except Exception as e:
            if self.failsafe:
                logger.warning('JOSM orthogonalization failed: ' + str(e) + '\n returning result without simplification')
                merged_fc = fc
            else:
                raise e
        else:
            fc = fc.to_utm()
            josm_fc = josm_fc.to_crs(fc.crs)

            # merge together initial vector and JOSM-processed vector
            merged_fc = josm.replace_blocks_by_josm(
                fc,
                josm_fc,
                union_tag=self.ids_tag,
                iou_threshold=self.iou_threshold,
                hausdorff_threshold=self.hausdorff_threshold,
            )
        # save result
        return merged_fc


class Polygons2Points(VectorProcessingBrick):
    """
    Transforms every Polygon into a single point
    Three methods are allowed:
    - centroid makes the point in the geometrical center of the geometry, but for convex polygons
        it may lie outside the shape
    - representative_point guarantees the point inside the shape, but not necessarily in the center
    - optimal uses centroid if it is inside and representative points else; may be slower
    """
    method: Literal['centroid', 'representative_point', 'optimal'] = Field('centroid')

    make_valid_input: bool  = Field(True)
    dropna_input: bool  = Field(True)
    drop_empty_input: bool  = Field(True)
    explode_input: bool  = Field(True)
    remove_repeated_points_input: bool  = Field(True)
    input_geom_types: Union[str, Sequence[str], None] = 'Polygon'

    # write params
    output_geom_types: Union[str, Sequence[str], None] = 'Point'
    crs: str = Field('utm')

    def process(self, fc: FeatureCollection):
        if self.method == 'centroid':
            fc.geometry = fc.geometry.centroid
        elif self.method == 'representative_point':
            fc.geometry = fc.geometry.representative_point()
        else:  # optimal. Not checking as it is checked in init
            def optimal_point(polygon):
                point = polygon.centroid
                if not point.within(polygon):
                    point = polygon.representative_point()
                return point
            fc.geometry = fc.geometry.map(optimal_point)
        return fc
