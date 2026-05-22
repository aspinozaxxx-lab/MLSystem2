import numpy as np
import pandas as pd
import shapely
from typing import Tuple, Callable, Any
from pyproj import CRS
from gpdadapter import FeatureCollection
from ..base import defaults
from loguru import logger
from .utils.mathutils import softmax
from .utils import angleutils


def validate_azimuth(x: Any) -> bool:
    """For usage in check_meta_angles()"""
    return isinstance(x, (int, float))


def validate_elevation(x: Any) -> bool:
    """For usage in check_meta_angles()"""
    return isinstance(x, (int, float)) and (0 <= x <= 90)


def angles_from_relative_position(fp: shapely.Polygon, rt: shapely.Polygon, height: float) -> Tuple[float, float]:
    """
    Returns satellite angles flom building height and footprint-rooftop relative positions
    Args:
        fp: footprint geometry
        rt: rooftop geometry
        height: building height
    Returns:
        sat_azimuth in degrees from y-axis clockwise, sat_elevation
    """
    vec = angleutils.vector_from_relative_positions(fp, rt)
    return angleutils.azimuth_from_vector(vec), angleutils.elevation_from_vector(vec, height)


def height_from_relative_position(fp: shapely.Polygon, rt: shapely.Polygon, elevation: float) -> float:
    """
    Returns building height from footprint and rooftop relative positions
    Args:
        fp: footprint geometry
        rt: rooftop geometry
        elevation: sat elevation
    Returns:
        building height, float
    """
    vec = angleutils.vector_from_relative_positions(fp, rt)
    return angleutils.height_from_vector(vec, elevation)


def vector_from_triangle(triangle: shapely.geometry.Polygon) -> Tuple[float, float]:
    """Returns a vector along the triangle's longest side"""
    if isinstance(triangle, shapely.geometry.MultiPolygon):
        triangle = triangle.geoms[0]
    coords = triangle.exterior.coords
    assert len(coords) == 3 or (len(coords) == 4 and coords[0] == coords[-1])
    lines = [shapely.geometry.LineString(coords[i:i + 2]) for i in range(3)]
    lines = sorted(lines, key=lambda x: x.length)
    min_line = lines[0]
    med_line = lines[1]
    max_line = lines[2]
    start = min_line.intersection(max_line).coords[0]
    end = med_line.intersection(max_line).coords[0]
    return end[0] - start[0], end[1] - start[1]


def azimuth_from_triangle(polygon: shapely.geometry.Polygon) -> float:
    """Returns azimuth and elevation calculated from triangle"""
    return angleutils.azimuth_from_vector(vector_from_triangle(polygon))


def elevation_from_triangle(polygon: shapely.geometry.Polygon, height: float) -> float:
    """Returns azimuth and elevation calculated from triangle"""
    return angleutils.elevation_from_vector(vector_from_triangle(polygon), height)


def failsafe_wrapper(func, default_result, msg: str = ''):
    def wrapper(x):
        try:
            return func(x)
        except Exception as e:
            logger.warning(msg+str(e))
            return default_result

    return wrapper


def get_angles_from_collection(aoi: shapely.Polygon,
                               markup: FeatureCollection,
                               height_tag: str = defaults.DEFINITIVE_HEIGHT_TAG,
                               default_azimuth: float = 0,
                               default_elevation: float = 90):
    """Compute satellite and sun angles from walls and shadows markup

    Args:
        aoi: Area of interest
        markup: Walls and Shadows markup with specified "building_height" attribute
        height_tag: name of Feature property, containing height
        default_azimuth: value to return if azimuth calculation fails
        default_elevation: value to return if elevation calculation fails


    Returns:
        dict: {'azimuth': <list of all azimuth values>,
               'elevation': <list of all elevation values>,
               'azimuth_median': float,
               'elevation_median': float,
               'azimuth_std': float,
               'elevation_std': float}
    """
    markup = markup.overlay(aoi)

    az_calculator = failsafe_wrapper(azimuth_from_triangle, np.nan, 'Failed to calculate azimuth: ')
    el_calculator = failsafe_wrapper(lambda x: elevation_from_triangle(x['geometry'], x[height_tag]), np.nan,
                                     'Failed to calculate elevation: ')

    azimuths = markup.map(az_calculator, 'geometry').dropna().to_list()
    elevations = markup.apply(el_calculator).dropna().to_list()

    az_median = angleutils.median_angle(azimuths) if azimuths else default_azimuth
    el_median = angleutils.elevation_guard(np.median(elevations)) if elevations else default_elevation

    az_mean = angleutils.mean_angle(azimuths) if azimuths else default_azimuth
    # el_mean = angleutils.elevation_guard(np.mean(elevations)) if elevations else default_elevation

    az_std = np.sqrt(np.sum([(az_mean - a)**2 for a in azimuths])) if azimuths else 0
    el_std = np.std(elevations) if elevations else 0

    # TODO: do we really need all this?
    return dict(
        azimuth=azimuths,
        elevation=elevations,
        azimuth_median=az_median,
        elevation_median=el_median,
        azimuth_std=az_std,
        elevation_std=el_std
    )


def assign_meta_angles_by_aoi(fc: FeatureCollection, meta: FeatureCollection) -> FeatureCollection:
    """For each AOI in meta assigns meta angles to features in fc by intersection"""
    if fc.empty:
        return fc

    for tag in set(defaults.angle_tags).difference(set(fc.columns)):
        fc[:, tag] = pd.Series(dtype=float)

    if meta.empty or not all(t in meta.columns for t in defaults.angle_tags):
        logger.warning('No angles in meta or meta is empty!')
        return fc

    if not fc.crs == meta.crs:
        logger.warning(f'CRS mismatch in assign_meta_angles_by_aoi: {fc.crs, meta.crs}')

    indexes = fc.query(meta.geometry).T  # shape (N, 2)
    for idx_pair in indexes:
        for tag in defaults.angle_tags:
            fc[idx_pair[1], tag] = meta[idx_pair[0], tag]
    return fc


def check_meta_angles(fc: FeatureCollection,
                      sun_azimuth_tag: str = defaults.SUN_AZIMUTH_TAG,
                      sun_elevation_tag: str = defaults.SUN_ELEVATION_TAG,
                      sat_azimuth_tag: str = defaults.SAT_AZIMUTH_TAG,
                      sat_elevation_tag: str = defaults.SAT_ELEVATION_TAG) -> bool:
    """Returns True if every feature in fc has valid sun and satellite angles"""
    try:
        return fc.loc[:, (sat_azimuth_tag, sun_azimuth_tag)].map(validate_azimuth).all().all() and \
               fc.loc[:, (sat_elevation_tag, sun_elevation_tag)].map(validate_elevation).all().all()
    except KeyError:
        return False  # There is no such columns in meta -> continue


def compute_meta_angles(meta: FeatureCollection,
                        walls_labels: FeatureCollection,
                        shadows_labels: FeatureCollection,
                        sun_azimuth_tag: str = defaults.SUN_AZIMUTH_TAG,
                        sun_elevation_tag: str = defaults.SUN_ELEVATION_TAG,
                        sat_azimuth_tag: str = defaults.SAT_AZIMUTH_TAG,
                        sat_elevation_tag: str = defaults.SAT_ELEVATION_TAG,
                        height_tag: str = defaults.DEFINITIVE_HEIGHT_TAG
                        ) -> FeatureCollection:
    if meta.empty:
        return meta

    if not (walls_labels.crs == shadows_labels.crs == meta.crs):
        logger.warning(f'CRS mismatch in compute_meta_angles: {walls_labels.crs, shadows_labels.crs, meta.crs}')

    meta[:, sun_azimuth_tag] = pd.Series(dtype=float)
    meta[:, sun_elevation_tag] = pd.Series(dtype=float)
    meta[:, sat_azimuth_tag] = pd.Series(dtype=float)
    meta[:, sat_elevation_tag] = pd.Series(dtype=float)

    def compute_angles_rowwise(row: pd.Series) -> pd.Series:
        sun_angles = get_angles_from_collection(row.geometry, markup=shadows_labels, height_tag=height_tag)
        sat_angles = get_angles_from_collection(row.geometry, markup=walls_labels, height_tag=height_tag)
        row[sun_azimuth_tag] = sun_angles['azimuth_median']
        row[sun_elevation_tag] = sun_angles['elevation_median']
        row[sat_azimuth_tag] = sat_angles['azimuth_median']
        row[sat_elevation_tag] = sat_angles['elevation_median']
        return row

    return FeatureCollection(meta.apply(compute_angles_rowwise))


def _predict_angles(pred: np.array, coef: float = 1.) -> dict:
    """Predicts meta angles from model single output"""
    # pred  ~  (C, H, W)
    sat_weights = softmax(pred[0].reshape(-1) * coef)
    sun_weights = softmax(pred[3].reshape(-1) * coef)

    sat_vecs = pred[1:3].reshape(2, -1)
    sun_vecs = pred[4:].reshape(2, -1)

    sat_vec_avg = np.average(sat_vecs, axis=-1, weights=sat_weights)
    sun_vec_avg = np.average(sun_vecs, axis=-1, weights=sun_weights)

    return {'sat_azimuth': angleutils.azimuth_from_vector(sat_vec_avg),
            'sat_elevation': angleutils.elevation_from_vector(sat_vec_avg, 1),
            'sun_azimuth': angleutils.azimuth_from_vector(sun_vec_avg),
            'sun_elevation': angleutils.elevation_from_vector(sun_vec_avg, 1)}


def predict_angles(bc, model: Callable[[np.array], np.array],
                   stride: Tuple[int, int] = (256, 256),
                   sample_size: Tuple[int, int] = (512, 512),
                   coef: float = 1.) -> FeatureCollection:
    """Predicts meta angles with model"""
    height, width = bc.height, bc.width
    ny, nx = (height - sample_size[0]) // stride[0], (width - sample_size[1]) // stride[1]
    pred = np.zeros((6, ny, nx))
    for y in range(ny):
        for x in range(nx):
            crop = np.stack(
                [band._band.read(window=((y*stride[0], y*stride[0] + sample_size[0]),
                                         (x*stride[1], x*stride[1] + sample_size[1])))[0].astype(np.uint8)
                 for band in bc],
                axis=0)
            pred[:, y, x] = model(crop)
    angles = _predict_angles(pred, coef)
    angles['geometry'] = shapely.geometry.box(*bc[0]._band.bounds)

    return FeatureCollection({k: [v] for k, v in angles.items()}, crs=bc[0]._band.crs)


def check_aoi_meta_intersection(aoi: FeatureCollection, meta: FeatureCollection, buffer: float = 201):
    """
    Raises ValueError if meta does not cover aoi.
    Buffer in Web Mercator units (meters)
    Buffer value is based on WorkflowEngine buffer of 200 meters plus 1 meter for implementation differences
    """
    # check if meta covers AOI
    # reprojecting to Web Mercator and setting segments to 8
    # for correspondence with buffer of AOI in Workflow Engine
    meta = meta.to_crs(CRS.from_epsg(3857))
    aoi = aoi.to_crs(CRS.from_epsg(3857))
    if len(aoi) != 1 or not isinstance(aoi[0, 'geometry'], (shapely.Polygon, shapely.MultiPolygon)):
        raise ValueError('AOI is empty or invalid')
    if not aoi[0, 'geometry'].within(meta.geometry.union_all().buffer(buffer, quad_segs=8)):
        raise ValueError('Meta does not cover AOI')

