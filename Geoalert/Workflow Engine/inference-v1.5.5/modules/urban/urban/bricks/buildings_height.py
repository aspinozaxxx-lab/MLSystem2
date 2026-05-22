import importlib.resources
import os.path
import shapely
from loguru import logger
from typing import Optional, Tuple
from gpdadapter import FeatureCollection
from ..base.brick import Brick
from ..base import defaults
from ..functional import io, metaangles, measure
from .. import height_histograms
import csv
from ..functional.generate import generate_footprint_from_roof, generate_roof_from_footprint, translate_polygon
from ..functional.metaangles import assign_meta_angles_by_aoi, check_aoi_meta_intersection
from ..base.brick import VectorProcessingBrick
import pandas as pd
from pydantic import Field


class ComputeMetaAngles(Brick):
    """Compute satellite and sun angles for each area of interest in meta collection
    based on shadows and walls markup

    If meta already contains some necessary meta angle tags,
    they are not updated, and the corresponding calculations are not done

    Args:
        shadows: filename of shadows angles markup feature collection
        walls: filename of walls angles markup feature collection
        meta: filename of meta feature collection
        output: filename of output feature `meta` collection (if None same as meta)
        sun_azimuth_tag: tag name for sun azimuth
        sun_elevation_tag: tag name for sun elevation
        sat_azimuth_tag: tag name for satellite azimuth
        sat_elevation_tag: tag name for satellite elevation
        height_tag: name of Feature property, containing height
    """
    shadows: str
    walls: str
    meta: str
    output: Optional[str] = Field(None)
    sun_azimuth_tag: str = Field(defaults.SUN_AZIMUTH_TAG)
    sun_elevation_tag: str = Field(defaults.SUN_ELEVATION_TAG)
    sat_azimuth_tag: str = Field(defaults.SAT_AZIMUTH_TAG)
    sat_elevation_tag: str = Field(defaults.SAT_ELEVATION_TAG)
    height_tag: str = Field(defaults.DEFINITIVE_HEIGHT_TAG)
    aoi: str = Field(defaults.AOI_FILENAME)

    def model_post_init(self, __context):
        self.output = self.output or self.meta

    def __call__(self, path):
        meta = io.read_fc(path, self.meta, explode=True, keep_only_geometry_types=shapely.Polygon)
        if meta.empty:
            logger.warning('Meta is empty, aborting ComputeMetaAngles')
            return

        if os.path.exists(os.path.join(path, self.aoi+defaults.VECTOR_EXT)):
            aoi = io.read_fc(path, self.aoi)
            check_aoi_meta_intersection(aoi, meta)
        else:
            logger.warning('AOI file not found, skipping the AOI-meta intersection check')

        if metaangles.check_meta_angles(meta, self.sun_azimuth_tag, self.sun_elevation_tag,
                                        self.sat_azimuth_tag, self.sat_elevation_tag):
            logger.info('All angles already present')
            io.save_fc(meta, path, self.output, make_valid=True, drop_empty=True, dropna=True, explode=True,
                       remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)
            return

        walls = io.read_fc(path, self.walls, explode=True, keep_only_geometry_types=shapely.Polygon)
        shadows = io.read_fc(path, self.shadows, explode=True, keep_only_geometry_types=shapely.Polygon)

        if walls.empty:
            logger.warning('Walls labels are empty, aborting ComputeMetaAngles')
            io.save_fc(meta, path, self.output, make_valid=False)
            return
        if shadows.empty:
            logger.warning('Shadows labels are empty, aborting ComputeMetaAngles')
            io.save_fc(meta, path, self.output, make_valid=False)

        crs = walls.estimate_utm_crs()
        meta = metaangles.compute_meta_angles(meta.to_crs(crs), walls.to_crs(crs), shadows.to_crs(crs),
                                              self.sun_azimuth_tag, self.sun_elevation_tag,
                                              self.sat_azimuth_tag, self.sat_elevation_tag, self.height_tag)
        io.save_fc(meta, path, self.output, make_valid=False)  # we already made it valid when read


class MeasureHeight(Brick):
    """Calculate height for buildings with given roof, walls, and shadow collections and metadata
    (satellite and sun angles)
            
    Args:
        roofs: filename of feature collection with roofs geometries
        walls: filename of feature collection with walls geometries
        shadows: filename of feature collection with shadows geometries
        meta: filename of feature collection with area properties
        sun_azimuth_tag: tag name for sun azimuth
        sun_elevation_tag: tag name for sun elevation
        sat_azimuth_tag: tag name for satellite azimuth
        sat_elevation_tag: tag name for satellite elevation
        sw_height_tag: tag to write the resulting height to
        sw_confidence_tag: tag to write height confidence score to
        simplification: simplification for feature collections to reduce computation complexity in UTM units
        height_range: range of heights for convergence
        default_height: default height for not converged heights buildings
        max_iterations: maximum iterations for optimization algorithm
        height_addition: float value to add for each measured height, due to algorithm underestimation
        output: filename of output feature collection (if None same as `roofs`)
    """

    roofs: str
    walls: str
    shadows: str
    meta: str
    output: Optional[str] = Field(None)
    sun_azimuth_tag: str = Field(defaults.SUN_AZIMUTH_TAG)
    sun_elevation_tag: str = Field(defaults.SUN_ELEVATION_TAG)
    sat_azimuth_tag: str = Field(defaults.SAT_AZIMUTH_TAG)
    sat_elevation_tag: str = Field(defaults.SAT_ELEVATION_TAG)
    sw_height_tag: str = Field(defaults.DEFINITIVE_HEIGHT_TAG)
    sw_confidence_tag: str = Field(defaults.SW_CONFIDENCE_TAG)
    simplification: float = Field(1)
    height_range: tuple = Field((3, 200))
    default_height: float = Field(6.0)
    height_addition: float = Field(0.0)
    max_iterations: int = Field(30)
    verbose: bool = Field(True)

    def model_post_init(self, __context):
        self.output = self.output or self.roofs

    def __call__(self, path):
        # read all vectors
        roofs_fc = io.read_fc(path, self.roofs, make_valid=False)  # presume we already made it valid in Vectorize
        walls_fc = io.read_fc(path, self.walls, make_valid=False)
        shadows_fc = io.read_fc(path, self.shadows, make_valid=False)
        meta = io.read_fc(path, self.meta, make_valid=False)  # presume we already made it valid in ComputeMetaAngles
        roofs_fc = measure.measure_heights(
                roofs_fc, walls_fc, shadows_fc, meta,
                sw_height_tag=self.sw_height_tag,
                sw_confidence_tag=self.sw_confidence_tag,
                simplification=self.simplification,
                height_range=self.height_range,
                default_height=self.default_height,
                max_iterations=self.max_iterations,
                height_addition=self.height_addition,
                verbose=self.verbose,
            )
        # save result
        logger.info(f"Saving to {path}")
        io.save_fc(roofs_fc, path, self.output, make_valid=False)  # geometry hasn't been changed, so no make_valid


class MeasureShift(Brick):
    """
    Finds optimal shift between rooftops and footprints based on walls
    Args:
        roofs: filename of feature collection with roofs geometries
        walls: filename of feature collection with walls geometries
        output: filename of feature collection with resulting geometries
        max_shift: maximum shift in UTM units
        rt_buffer: rooftop buffer for selection of walls in UTM units
        closing: closing in UTM units
        simplify: simplification in UTM units
        max_iterations: maximum iterations for optimization algorithm
    """
    roofs: str
    walls: str
    output: Optional[str] = Field(None)
    max_shift: float = Field(15.0)
    rt_buffer: float = Field(1.0)
    closing: float = Field(0.0)
    simplify: float = Field(0.0)
    max_iterations: int = Field(30)
    x_shift_tag: str = Field('_x_shift')
    y_shift_tag: str = Field('_y_shift')
    confidence_shift_tag: str = Field('_confidence_shift')

    def model_post_init(self, __context):
        self.output = self.output or self.roofs

    def __call__(self, path):
        # read all vectors
        roofs_fc = io.read_fc(path, self.roofs, crs='utm', make_valid=False) # presume we already made it valid in Vectorize
        walls_fc = io.read_fc(path, self.walls, crs=roofs_fc.crs, make_valid=False)

        roofs_fc = measure.measure_shifts(roofs_fc,
                                          walls_fc,
                                          max_iterations=self.max_iterations,
                                          closing=self.closing,
                                          simplify=self.simplify,
                                          rt_buffer=self.rt_buffer,
                                          max_shift=self.max_shift,
                                          x_shift_tag=self.x_shift_tag,
                                          y_shift_tag=self.y_shift_tag,
                                          confidence_shift_tag=self.confidence_shift_tag)
        # save result
        logger.info(f"Saving to {path}")
        io.save_fc(roofs_fc, path, self.output, make_valid=False)  # geometry hasn't been changed, so no make_valid


class CorrectShift(Brick):
    """
    Corrects shift for intersected footprints
    Args:
        footprints: filename of feature collection with footprints
        walls: filename of feature collection with walls
        output: filename of feature collection with resulting geometries
        x_shift_tag: tag for x shift
        y_shift_tag: tag for y shift
        confidence_shift_tag: tag for confidence
        corr_x_shift_tag: tag for corrected x shift
        corr_y_shift_tag: tag for corrected y shift
        corr_confidence_shift_tag: tag for corrected confidence
        corr_threshold: threshold for corrected confidence
        closing: closing in UTM units
        simplify: simplification in UTM units
        rt_buffer: rooftop buffer for selection of walls in UTM units
    """
    footprints: str
    walls: str
    x_shift_tag: str
    y_shift_tag: str
    confidence_shift_tag: str
    corr_x_shift_tag: str
    corr_y_shift_tag: str
    corr_confidence_shift_tag: str
    corr_threshold: float
    output: Optional[str] = Field(None)
    rt_buffer: float = Field(1.0)
    closing: float = Field(0.0)
    simplify: float = Field(0.0)

    def model_post_init(self, __context):
        self.output = self.output or self.footprints

    def __call__(self, path):
        # read all vectors
        fps_fc = io.read_fc(path, self.footprints, crs='utm', make_valid=False)  # presume we already made it valid in Vectorize
        walls_fc = io.read_fc(path, self.walls, crs=fps_fc.crs, make_valid=False)

        if (self.x_shift_tag not in fps_fc.columns or self.y_shift_tag not in fps_fc.columns
                or self.confidence_shift_tag not in fps_fc.columns):
            logger.warning('XY shift not found in footprints, skipping correction')
        else:
            fps_fc = measure.correct_shifts(fps_fc,
                                            walls_fc,
                                            x_shift_tag=self.x_shift_tag,
                                            y_shift_tag=self.y_shift_tag,
                                            confidence_shift_tag=self.confidence_shift_tag,
                                            corr_x_shift_tag=self.corr_x_shift_tag,
                                            corr_y_shift_tag=self.corr_y_shift_tag,
                                            corr_confidence_shift_tag=self.corr_confidence_shift_tag,
                                            corr_threshold=self.corr_threshold,
                                            rt_buffer=self.rt_buffer,
                                            closing=self.closing,
                                            simplify=self.simplify)

        # save result
        logger.info(f"Saving to {path}")
        io.save_fc(fps_fc, path, self.output, make_valid=False)  # geometry hasn't been changed, so no make_valid


class GenerateFootprints(Brick):
    """Generate footprints from given roofs

    Note:
        Roofs features should contain "sat_azimuth", "sat_elevation" and "building_height" attributes

    Args:
        roofs: filename of feature collection with roofs geometries
        output: filename of output footprints feature collection
        meta: filename of meta feature collection with angles. If provided, overrides values in roofs fc
        sat_azimuth_tag: tag name for satellite azimuth
        sat_elevation_tag: tag name for satellite elevation
        building_height_tag: tag name for building height
        x_shift_tag: tag name for x shift
        y_shift_tag: tag name for y shift
        confidence_shift_tag: tag name for confidence shift
        confidence_shift_thr: threshold for confidence shift
    """
    roofs: str
    output: Optional[str] =Field(None)
    meta: Optional[str] =Field(None)
    sat_azimuth_tag: str = Field(defaults.SAT_AZIMUTH_TAG)
    sat_elevation_tag: str = Field(defaults.SAT_ELEVATION_TAG)
    building_height_tag: str = Field(defaults.DEFINITIVE_HEIGHT_TAG)
    x_shift_tag: Optional[str] = Field(None)
    y_shift_tag: Optional[str] = Field(None)
    confidence_shift_tag: Optional[str] = Field(None)
    confidence_shift_thr: Optional[float] = Field(None)

    def model_post_init(self, __context):
        self.output = self.output or self.roofs

    def __call__(self, path):
        roofs = io.read_fc(path, self.roofs).to_utm()
        if roofs.empty:
            io.save_fc(roofs, path, self.output, make_valid=False)
            return

        if self.x_shift_tag and self.y_shift_tag:  # from xy shift
            if self.x_shift_tag not in roofs.columns or self.y_shift_tag not in roofs.columns:
                logger.warning('XY shift not found in roofs, skipping footprints generation')
                io.save_fc(roofs, path, self.output, make_valid=False)
                return

            def _generate_footprint_by_confidence(roof: pd.Series) -> shapely.Polygon:
                if roof[self.confidence_shift_tag] < self.confidence_shift_thr:
                    return roof.geometry
                else:
                    return translate_polygon(roof.geometry, roof[self.x_shift_tag], roof[self.y_shift_tag])

            def _generate_footprint(roof: pd.Series) -> shapely.Polygon:
                return translate_polygon(roof.geometry, roof[self.x_shift_tag], roof[self.y_shift_tag])

            if self.confidence_shift_tag is not None and self.confidence_shift_thr is not None:
                roofs[:, 'geometry'] = roofs.apply(_generate_footprint_by_confidence)
            else:
                roofs[:, 'geometry'] = roofs.apply(_generate_footprint)

            io.save_fc(roofs, path, self.output, make_valid=True, drop_empty=True, dropna=True, explode=True,
                       remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)

        else:  # from height and satellite angles
            if defaults.SAT_AZIMUTH_TAG not in roofs.columns or defaults.SAT_ELEVATION_TAG not in roofs.columns:
                if self.meta:
                    meta = io.read_fc(path, self.meta, make_valid=False).to_crs(roofs.crs)
                    roofs = assign_meta_angles_by_aoi(roofs, meta)
                else:
                    logger.warning('Satellite angles not found in roofs and meta not provided,'
                                   ' skipping footprints generation')
                    io.save_fc(roofs, path, self.output, make_valid=False)
                    return

            def _generate_footprint(roof: pd.Series) -> shapely.Polygon:
                return generate_footprint_from_roof(roof.geometry, roof[self.building_height_tag],
                                                    roof[defaults.SAT_AZIMUTH_TAG], roof[defaults.SAT_ELEVATION_TAG])

            roofs[:, 'geometry'] = roofs.apply(_generate_footprint)
            io.save_fc(roofs, path, self.output, make_valid=True, drop_empty=True, dropna=True, explode=True,
                       remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)



class GenerateRoofs(Brick):
    """Generate roofs from given footprints (or ZKH points)

    Note:
        Footprints features should contain "sat_azimuth", "sat_elevation" and "building_height" attributes

    Args:
        footprints: filename of feature collection with footprints geometries
        output: filename of output footprints feature collection
        sat_azimuth_tag: tag name for satellite azimuth
        sat_elevation_tag: tag name for satellite elevation
        building_height_tag: tag name for building height
    """
    footprints: str
    output: Optional[str] = Field(None)
    meta: Optional[str] = Field(None)
    sat_azimuth_tag: str = Field(defaults.SAT_AZIMUTH_TAG)
    sat_elevation_tag: str = Field(defaults.SAT_ELEVATION_TAG)
    building_height_tag: str = Field(defaults.DEFINITIVE_HEIGHT_TAG)

    def model_post_init(self, __context):
        self.output = self.output or self.footprints

    def __call__(self, path):
        footprints = io.read_fc(path, self.footprints).to_utm()

        if footprints.empty:
            io.save_fc(footprints, path, self.output, make_valid=False)
            return

        if defaults.SAT_AZIMUTH_TAG not in footprints.columns or defaults.SAT_ELEVATION_TAG not in footprints.columns:
            if self.meta:
                meta = io.read_fc(path, self.meta, make_valid=False).to_crs(footprints.crs)
                footprints = assign_meta_angles_by_aoi(footprints, meta)
            else:
                logger.warning('Satellite angles not found in footprints and meta not provided,'
                               ' skipping roofs generation')
                io.save_fc(footprints, path, self.output, make_valid=False)
                return

        def _generate_roof(footprint: pd.Series) -> shapely.Polygon:
            return generate_roof_from_footprint(footprint.geometry, footprint[self.building_height_tag],
                                                footprint[defaults.SAT_AZIMUTH_TAG],
                                                footprint[defaults.SAT_ELEVATION_TAG])

        footprints[:, 'geometry'] = footprints.apply(_generate_roof)
        io.save_fc(footprints, path, self.output, make_valid=True, drop_empty=True, dropna=True, explode=True,
                   remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)


class HeightsByArea(VectorProcessingBrick):
    """Predicts buildings heights based on area, having .csv file with area->height histogram

    Args:
        input: name of feature collection
        histogram_file: name of to .csv with histogram columns must be Min area,Max area,Median,IQR, 1st row header
        output: name of output feature collection
        crs: crs for area calculation
        height_tag: tag for height
        confidence_tag: tag for height confidence
    """
    input: str
    output: Optional[str] = Field(None)
    histogram_file: str = Field('russia')
    crs: Optional[str] = Field('epsg:3857')
    height_tag: str = Field(defaults.AREA_HEIGHT_TAG)
    confidence_tag: str = Field(defaults.AREA_HEIGHT_CONFIDENCE_TAG)
    _histogram = None


    def model_post_init(self, __context):
        self.output = self.output or self.input
        self._histogram = dict()
        with importlib.resources.open_text(height_histograms, f'{self.histogram_file}.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    self._histogram[int(row[1])] = (float(row[2]), float(row[3]))
                except Exception as e:
                    logger.warning(str(e))
                    continue

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        def height_by_area(polygon: shapely.Polygon) -> Tuple[float, float]:  # TODO: to functional
            a = polygon.area
            for bin in self._histogram.keys():
                if a < bin:
                    return self._histogram[bin]

        heights_and_confidences = fc.map(height_by_area, 'geometry')
        fc[:, self.confidence_tag] = heights_and_confidences.map(lambda x: x[1])  # TODO: what does it mean? IQR?
        fc[:, self.height_tag] = heights_and_confidences.map(lambda x: x[0])
        return fc
