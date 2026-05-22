import pandas as pd
import scipy
import numpy as np
import shapely.geometry
from gpdadapter import FeatureCollection
from tqdm import tqdm
from scipy.optimize import minimize_scalar, differential_evolution
from typing import Tuple, List, Final
from functools import wraps
from ..base import defaults
from . import generate
from pyproj import CRS
from .utils.geomutils import calculate_iou, intersection, to_multipolygon
from .metaangles import assign_meta_angles_by_aoi
import warnings
from loguru import logger

# ------------------------------ Helpers -----------------------------------

HOOK_HEIGHT: Final[float] = 20  # height of initial 'hook' when selecting wall for roof
CROP_HEIGHT: Final[float] = 200  # maximum height when selecting wall for roof


# ----------------------------------- Main Logic -----------------------------------

def select_walls_and_shadows_for_roof_old(
    roof: shapely.Polygon,
    walls: FeatureCollection,
    shadows: FeatureCollection,
    sun_azimuth: float,
    sat_azimuth: float,
    sun_elevation: float,
    sat_elevation: float,
    hook_height: float = HOOK_HEIGHT,
    crop_height: float = CROP_HEIGHT
) -> Tuple[shapely.MultiPolygon, shapely.MultiPolygon]:
    """Select shadows and walls for provided roof feature with following scheme:
        1. Create small wall
        2. Intersect all walls with created small wall (to choose nearest)
        3. Create big wall and intersect selected nearest walls
        4. Repeat with shadows
    Args:
        roof: roof Polygon
        walls: walls FeatureCollection
        shadows: shadows FeatureCollection
        sun_azimuth: sun_azimuth
        sat_azimuth: sat_azimuth
        sun_elevation: sun_elevation
        sat_elevation: sat_elevation
        hook_height: height of initial 'hook' when selecting wall for roof
        crop_height: maximum height when selecting wall for roof
    Returns:
        walls, shadows as MultiPolygons
    """

    # TODO: subtract neighbour rooftops and (if possible) walls and shadows from detected walls and shadows
    # ------ selecting walls ------
    hook_wall = generate.generate_wall_from_roof(roof, hook_height, sat_azimuth, sat_elevation, closing=0, simplify=0.2)
    detected_walls = walls[walls.query(hook_wall)].geometry

    if len(detected_walls) > 0:
        crop_wall = generate.generate_wall_from_roof(roof, crop_height, sat_azimuth, sat_elevation, closing=0, simplify=0.2)
        detected_walls = [intersection(w, crop_wall) for w in detected_walls]

    detected_walls = to_multipolygon(detected_walls).buffer(0)

    # ------ selecting shadows ------

    # this is not an actual footprint, but will be useful
    # for shadow cropping, as we don't know where the actual footprint is located
    footprint = detected_walls.union(roof).convex_hull.simplify(1.)

    hook_shadow = generate.generate_shadow_from_footprint(footprint, hook_height, sun_azimuth, sun_elevation,
                                                 sat_azimuth, sat_elevation, closing=0, simplify=0.2).buffer(0)

    detected_shadows = shadows[shadows.query(hook_shadow)].geometry

    if len(detected_shadows) > 0:
        crop_shadow = generate.generate_shadow_from_footprint(footprint, crop_height, sun_azimuth, sun_elevation,
                                                     sat_azimuth, sat_elevation, closing=0, simplify=0.2).buffer(0)
        detected_shadows = [intersection(s, crop_shadow) for s in detected_shadows]

    detected_shadows = to_multipolygon(detected_shadows).buffer(0)

    return detected_walls, detected_shadows


def estimate_height_old(
    roof: shapely.geometry.Polygon,
    walls: shapely.geometry.MultiPolygon,
    shadows: shapely.geometry.MultiPolygon,
    sun_azimuth: float,
    sat_azimuth: float,
    sun_elevation: float,
    sat_elevation: float,
    max_iterations: int = 30,
    height_range: Tuple[float, float] = (3.0, 50.0),
):
    """Optimize building height with given shadows and walls"""

    if walls.is_empty and shadows.is_empty:
        return scipy.optimize.OptimizeResult(
            fun=1.0,
            message="Empty shadows and walls",
            nfev=0,
            status=0,
            success=False,
            x=height_range[0],
        )

    def objective(height):
        gen_shadows, gen_walls = generate.generate_shadow_and_wall_from_roof(
            roof,
            height,
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            sat_azimuth=sat_azimuth,
            sat_elevation=sat_elevation,
            closing=0.01,
            return_empty_on_error=True,
        )
        iou_shadows = calculate_iou(shadows, gen_shadows)
        iou_walls = calculate_iou(walls, gen_walls)
        iou = (iou_shadows + iou_walls) / 2
        return 1 - iou

    # here we will make dynamic height range scaling
    # minimize_scalar not always converge for wide ranges
    # so, we will divide our range by 4, 2, 1 to start with smaller ranges
    # and go to wider range only if we converge to upper bound
    h_min, h_max = height_range
    hs = [max(h_max / 2 ** i, h_min) for i in range(3)]

    for h in reversed(hs):
        res = minimize_scalar(
            objective,
            method="bounded",
            bounds=(h_min, h),
            options=dict(maxiter=max_iterations),
            tol=1.,
        )
        if not(res.fun < 0.95 and res.x / h > 0.95):
            break

    return res


class HeightEstimator:

    def __init__(
        self,
        roofs: FeatureCollection,
        walls: FeatureCollection,
        shadows: FeatureCollection,
        angles: dict,
        sw_height_tag: str = defaults.SW_HEIGHT_TAG,
        sw_confidence_tag: str = defaults.SW_CONFIDENCE_TAG,
        simplification: float = 1.,
        height_range: tuple = (3, 50),
        default_height: float = 6.0,
        max_iterations: int = 30,
        presimplify: float = 1,
        verbose: bool = True,
    ):
        self.roofs = roofs.simplify(presimplify)
        self.walls = walls.simplify(presimplify)
        self.shadows = shadows.simplify(presimplify)
        self.angles = angles
        self.sw_height_tag = sw_height_tag
        self.sw_confidence_tag = sw_confidence_tag
        self.simplification = simplification
        self.height_range = height_range
        self.default_height = default_height
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _estimate_height(self, i):
        """Estimate height for one roof"""
        roof = self.roofs.geometry[i]

        walls, shadows = select_walls_and_shadows_for_roof_old(
            roof=roof,
            walls=self.walls,
            shadows=self.shadows,
            sat_azimuth=self.angles[defaults.SAT_AZIMUTH_TAG],
            sun_azimuth=self.angles[defaults.SUN_AZIMUTH_TAG],
            sat_elevation=self.angles[defaults.SAT_ELEVATION_TAG],
            sun_elevation=self.angles[defaults.SUN_ELEVATION_TAG]
        )

        result = estimate_height_old(
            roof=roof,
            walls=walls.simplify(self.simplification),
            shadows=shadows.simplify(self.simplification),
            sat_azimuth=self.angles[defaults.SAT_AZIMUTH_TAG],
            sun_azimuth=self.angles[defaults.SUN_AZIMUTH_TAG],
            sat_elevation=self.angles[defaults.SAT_ELEVATION_TAG],
            sun_elevation=self.angles[defaults.SUN_ELEVATION_TAG],
            max_iterations=self.max_iterations,
            height_range=self.height_range,
        )

        return result

    def compute_heights(self) -> Tuple[np.array, np.array]:
        results = []
        with tqdm(total=len(self.roofs), disable=not self.verbose) as pbar:
            for roof_idx in range(len(self.roofs)):
                pbar.update()
                results.append(self._estimate_height(roof_idx))

        heights = np.round(np.array([r.x if r.fun < 0.85 else self.default_height for r in results]),
                           defaults.HEIGHT_DECIMALS)
        confs = np.round(np.array([1 - r.fun for r in results]), defaults.CONFIDENCE_DECIMALS)
        return heights, confs


def split_to_regions_by_intersection(aois: FeatureCollection,
                                     fc: FeatureCollection) -> List[FeatureCollection]:
    # TODO: not used
    assert aois.crs == fc.crs, f"{aois.crs} != {fc.crs}"
    return [fc.overlay(aoi) for aoi in aois.geometry]


def utm_zone(fc: FeatureCollection, idx: int) -> CRS:
    """
    gets appropriate UTM zone CRS (rasterio.crs.CRS object) for certain AOI at idx
    raises ValueError if the FC is too wide
    """
    f: FeatureCollection = fc.geometry[idx]
    lon1, lat1, lon2, lat2 = f.total_bounds
    if abs(lat1 - lat2) > 12:
        # TODO: split invalid AOI to make it valid
        raise ValueError(f"FeatureCollection has too wide latitude range "
                         f"({lat1} to {lat2}) to calculate UTM zone")
    return FeatureCollection({'geometry': [f]}, crs=fc.crs).estimate_utm_crs()


def measure_heights_old(
        roofs_fc: FeatureCollection,
        walls_fc: FeatureCollection,
        shadows_fc: FeatureCollection,
        meta: FeatureCollection,
        sw_height_tag: str = defaults.SW_HEIGHT_TAG,
        sw_confidence_tag: str = defaults.SW_CONFIDENCE_TAG,
        simplification: float = 1.,
        height_range: tuple = (3, 50),
        default_height: float = 6.0,
        max_iterations: int = 30,
        height_addition: float = 0,
        verbose: bool = True) -> FeatureCollection:
    if not roofs_fc:
        return roofs_fc
    # We do not want to reproject roofs before division into AOIs because Meta geometry may be too big for UTM,
    # so we freeze the utm zone for all reprojections and handle walls&shadows first
    assert roofs_fc.crs == walls_fc.crs == shadows_fc.crs  # must be epsg4326 at this point
    result = FeatureCollection()  # empty container

    for aoi_idx in range(len(meta)):
        crs = utm_zone(meta, aoi_idx)
        aoi_roofs_fc = roofs_fc.overlay(meta.geometry[aoi_idx]).to_crs(crs)
        if len(aoi_roofs_fc) == 0:
            continue
        aoi_walls_fc = walls_fc.overlay(meta.geometry[aoi_idx]).to_crs(crs)
        aoi_shadows_fc = shadows_fc.overlay(meta.geometry[aoi_idx]).to_crs(crs)

        aoi_angles = meta[aoi_idx, list(defaults.angle_tags)].to_dict()
        estimator = HeightEstimator(
            roofs=aoi_roofs_fc,
            walls=aoi_walls_fc,
            shadows=aoi_shadows_fc,
            angles=aoi_angles,
            sw_height_tag=sw_height_tag,
            sw_confidence_tag=sw_confidence_tag,
            simplification=simplification,
            height_range=height_range,
            default_height=default_height,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        heights, confs = estimator.compute_heights()
        aoi_roofs_fc[:, sw_height_tag] = heights
        aoi_roofs_fc[:, sw_confidence_tag] = confs
        result.append(aoi_roofs_fc)

    # We calculated height in the web mercator projection.
    # To get the real height we must multiply it by cos(latitude) of the object
    if height_addition:
        roofs_fc[:, sw_height_tag] = roofs_fc[:, sw_height_tag].map(
            lambda x: x if x == default_height else x + height_addition)

    return result


# NEW ------------------------------------------------------------------------------------------------------------------

def select_walls_and_shadows_for_roof(row: pd.Series,
                                      walls: FeatureCollection,
                                      shadows: FeatureCollection,
                                      hook_height: float = HOOK_HEIGHT,
                                      crop_height: float = CROP_HEIGHT) -> pd.Series:
    """Select shadows and walls for provided roof feature with following scheme:
        1. Create small wall
        2. Intersect all walls with created small wall (to choose nearest)
        3. Create big wall and intersect selected nearest walls
        4. Repeat with shadows
    Args:
        row: roof fc row
        walls: walls FeatureCollection
        shadows: shadows FeatureCollection
        hook_height: height of initial 'hook' when selecting wall for roof
        crop_height: maximum height when selecting wall for roof
    Returns:
        walls, shadows as MultiPolygons
    """
    # ------ selecting walls ------
    sat_azimuth, sat_elevation, sun_azimuth, sun_elevation = row['sat_azimuth'], row['sat_elevation'],\
        row['sun_azimuth'], row['sun_elevation']
    try:
        hook_wall = generate.generate_wall_from_roof(row['geometry'], hook_height, sat_azimuth,
                                            sat_elevation, closing=0, simplify=0.2)
        detected_walls = walls.geometry[walls.query(hook_wall)]

        if len(detected_walls) > 0:
            crop_wall = generate.generate_wall_from_roof(row['geometry'], crop_height, sat_azimuth, sat_elevation,
                                                closing=0, simplify=0.2)
            detected_walls = detected_walls.clip(crop_wall)
        detected_walls = detected_walls.unary_union
    except:
        detected_walls = None

    # ------ selecting shadows ------

    # this is not an actual footprint, but will be useful
    # for shadow cropping, as we don't know where the actual footprint is located
    try:
        if detected_walls:
            footprint = detected_walls.union(row['geometry']).convex_hull.simplify(1.)
        else:
            footprint = row['geometry'].convex_hull.simplify(1.)

        hook_shadow = generate.generate_shadow_from_footprint(footprint, hook_height, sun_azimuth, sun_elevation,
                                                     sat_azimuth, sat_elevation, closing=0, simplify=0.2)

        detected_shadows = shadows.geometry[shadows.query(hook_shadow)]

        if len(detected_shadows) > 0:
            crop_shadow = generate.generate_shadow_from_footprint(footprint, crop_height, sun_azimuth, sun_elevation,
                                                         sat_azimuth, sat_elevation, closing=0, simplify=0.2)
            detected_shadows = detected_shadows.intersection(crop_shadow)

        detected_shadows = detected_shadows.unary_union
    except:
        detected_shadows = None
    row['walls'] = detected_walls
    row['shadows'] = detected_shadows
    return row


def select_walls_for_roof(row: pd.Series,
                          walls: FeatureCollection,
                          rt_buffer: float = 1.0) -> pd.Series:
    try:
        detected_walls = walls.geometry[walls.query(row['geometry'].buffer(rt_buffer))].unary_union
    except:
        detected_walls = None

    row['walls'] = detected_walls
    return row


def select_walls_for_footprint(row: pd.Series,
                               walls: FeatureCollection,
                               x_shift_tag: str,
                               y_shift_tag: str,
                               rt_buffer: float = 1.0) -> pd.Series:
    # translate footprint back to rooftop
    rooftop = generate.translate_polygon(row['geometry'], -row[x_shift_tag], -row[y_shift_tag])
    try:
        detected_walls = walls.geometry[walls.query(rooftop.buffer(rt_buffer))].unary_union
    except:
        detected_walls = None

    row['walls'] = detected_walls
    return row


def estimate_height(row: pd.Series,
                    max_iterations: int = 30,
                    height_range: Tuple[float, float] = (3.0, 50.0),
                    height_tag: str = defaults.SW_HEIGHT_TAG,
                    confidence_tag: str = defaults.SW_CONFIDENCE_TAG,
                    default_height: float = -1) -> pd.Series:
    if not(row['walls']) and not(row['shadows']):
        row[confidence_tag] = 0
        row[height_tag] = -1
        return row

    def objective(height):
        gen_shadows, gen_walls = generate.generate_shadow_and_wall_from_roof(
            row['geometry'],
            height,
            sun_azimuth=row['sun_azimuth'],
            sun_elevation=row['sun_elevation'],
            sat_azimuth=row['sat_azimuth'],
            sat_elevation=row['sat_elevation'],
            closing=0.01,
            return_empty_on_error=True,
        )
        iou_shadows = calculate_iou(row['shadows'], gen_shadows) if row['shadows'] else 0
        iou_walls = calculate_iou(row['walls'], gen_walls) if row['walls'] else 0
        iou = (iou_shadows + iou_walls) / 2
        return 1 - iou

    # here we will make dynamic height range scaling
    # minimize_scalar not always converge for wide ranges
    # so, we will divide our range by 4, 2, 1 to start with smaller ranges
    # and go to wider range only if we converge to upper bound
    h_min, h_max = height_range
    hs = [max(h_max / 2 ** i, h_min) for i in range(3)]

    for h in reversed(hs):
        res = minimize_scalar(
            objective,
            method="bounded",
            bounds=(h_min, h),
            options=dict(maxiter=max_iterations),
            tol=1.,
        )
        if not (res.fun < 0.95 and res.x / h > 0.95):
            break

    row[confidence_tag] = np.round(1 - res.fun, defaults.CONFIDENCE_DECIMALS)
    row[height_tag] = np.round(res.x, defaults.HEIGHT_DECIMALS) if res.fun < 0.85 else default_height
    return row


def estimate_shift(row: pd.Series,
                   max_iterations: int = 30,
                   closing: float = 0.0,
                   simplify: float = 0.0,
                   max_shift: float = 15.0,
                   x_shift_tag: str = '_x_shift',
                   y_shift_tag: str = '_y_shift',
                   confidence_shift_tag: str = '_confidence_shift') -> pd.Series:
    if not row['walls']:
        row[x_shift_tag] = 0
        row[y_shift_tag] = 0
        row[confidence_shift_tag] = 1
        return row

    def objective(shift_coords):
        x, y = shift_coords[0], shift_coords[1]
        gen_walls = generate.generate_wall_from_roof_by_x_y(row['geometry'], x, y,
                                                            closing=closing,
                                                            simplify=simplify)
        iou_walls = calculate_iou(row['walls'], gen_walls)
        return 1 - iou_walls

    roof_bbox = np.array(tuple(shapely.geometry.box(*row['geometry'].bounds).exterior.coords)[:-1])
    wall_bbox = np.array(tuple(shapely.geometry.box(*row['walls'].bounds).exterior.coords)[:-1])
    left_bound = max(min(0, wall_bbox[:, 0].min() - roof_bbox[:, 0].min()), -max_shift)
    right_bound = min(max(0, wall_bbox[:, 0].max() - roof_bbox[:, 0].max()), max_shift)
    top_bound = min(max(0, wall_bbox[:, 1].max() - roof_bbox[:, 1].max()), max_shift)
    bottom_bound = max(min(0, wall_bbox[:, 1].min() - roof_bbox[:, 1].min()), -max_shift)
    bounds = [(left_bound, right_bound), (bottom_bound, top_bound)]
    res = differential_evolution(objective, bounds, tol=1.0, maxiter=max_iterations)

    row[confidence_shift_tag] = np.round(1 - res.fun, defaults.CONFIDENCE_DECIMALS)
    row[x_shift_tag] = res.x[0]
    row[y_shift_tag] = res.x[1]
    return row


def measure_heights(roofs_fc: FeatureCollection,
                    walls_fc: FeatureCollection,
                    shadows_fc: FeatureCollection,
                    meta: FeatureCollection,
                    sw_height_tag: str = defaults.SW_HEIGHT_TAG,
                    sw_confidence_tag: str = defaults.SW_CONFIDENCE_TAG,
                    simplification: float = 1.,
                    height_range: tuple = (3, 50),
                    default_height: float = 6.0,
                    max_iterations: int = 30,
                    height_addition: float = 0,
                    verbose: bool = True) -> FeatureCollection:
    if roofs_fc.empty:
        return roofs_fc
    if walls_fc.empty and shadows_fc.empty:
        roofs_fc[:, sw_height_tag] = -1
        roofs_fc[:, sw_confidence_tag] = 0
        return roofs_fc
    if meta.empty:
        warnings.warn("Got empty meta, returning")
        roofs_fc[:, sw_height_tag] = -1
        roofs_fc[:, sw_confidence_tag] = 0
        return roofs_fc

    crs = meta.estimate_utm_crs()
    roofs_fc.to_crs(crs, inplace=True)
    walls_fc.to_crs(crs, inplace=True)
    shadows_fc.to_crs(crs, inplace=True)
    meta.to_crs(crs, inplace=True)

    roofs_fc = assign_meta_angles_by_aoi(roofs_fc, meta)

    roofs_fc_simplified = roofs_fc.simplify(simplification)
    walls_fc.simplify(simplification, inplace=True)
    shadows_fc.simplify(simplification, inplace=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FeatureWarning from gpd.unary_union()
        roofs_fc_simplified = FeatureCollection(roofs_fc_simplified.apply(select_walls_and_shadows_for_roof,
                                                                          walls=walls_fc,
                                                                          shadows=shadows_fc))

    roofs_fc_simplified = FeatureCollection(roofs_fc_simplified.apply(estimate_height,
                                                                      max_iterations=max_iterations,
                                                                      height_range=height_range,
                                                                      default_height=default_height,
                                                                      height_tag=sw_height_tag,
                                                                      confidence_tag=sw_confidence_tag))

    roofs_fc[:, sw_height_tag] = roofs_fc_simplified[:, sw_height_tag]
    roofs_fc[:, sw_confidence_tag] = roofs_fc_simplified[:, sw_confidence_tag]
    if height_addition:
        roofs_fc[:, sw_height_tag] = roofs_fc[:, sw_height_tag].map(
            lambda x: x if x == default_height else x + height_addition)

    return roofs_fc


def measure_shifts(roofs_fc: FeatureCollection,
                   walls_fc: FeatureCollection,
                   max_iterations: int = 30,
                   closing: float = 0.0,
                   simplify: float = 0.0,
                   rt_buffer: float = 1.0,
                   max_shift: float = 15.0,
                   x_shift_tag: str = '_x_shift',
                   y_shift_tag: str = '_y_shift',
                   confidence_shift_tag: str = '_confidence_shift'
                   ) -> FeatureCollection:
    if roofs_fc.empty or walls_fc.empty:
        return roofs_fc
    roofs_fc.to_utm(inplace=True)
    walls_fc.to_utm(inplace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FeatureWarning from gpd.unary_union()
        roofs_fc1 = FeatureCollection(roofs_fc.apply(select_walls_for_roof, walls=walls_fc, rt_buffer=rt_buffer))

    roofs_fc2 = FeatureCollection(roofs_fc1.apply(estimate_shift,
                                                  max_iterations=max_iterations,
                                                  closing=closing,
                                                  simplify=simplify,
                                                  max_shift=max_shift,
                                                  x_shift_tag=x_shift_tag,
                                                  y_shift_tag=y_shift_tag,
                                                  confidence_shift_tag=confidence_shift_tag))

    roofs_fc[:, confidence_shift_tag] = roofs_fc2[:, confidence_shift_tag]
    roofs_fc[:, x_shift_tag] = roofs_fc2[:, x_shift_tag]
    roofs_fc[:, y_shift_tag] = roofs_fc2[:, y_shift_tag]

    return roofs_fc

def estimate_correct_shift(row: pd.Series,
                           fp_fc: FeatureCollection,
                           x_shift_tag: str,
                           y_shift_tag: str,
                           confidence_shift_tag: str,
                           corr_x_shift_tag: str,
                           corr_y_shift_tag: str,
                           corr_confidence_shift_tag: str,
                           corr_threshold: float,
                           closing: float = 0.0,
                           simplify: float = 0.0):

    if not row['walls']:
        row[corr_x_shift_tag] = 0
        row[corr_y_shift_tag] = 0
        row[corr_confidence_shift_tag] = -1
        return row

    fc = fp_fc.overlay(row.geometry)
    if fc.empty:
        row[corr_x_shift_tag] = 0
        row[corr_y_shift_tag] = 0
        row[corr_confidence_shift_tag] = -2
        return row

    x_shift = 0
    y_shift = 0
    max_confidence_shift = row[confidence_shift_tag]
    for i in range(len(fc)):
        if fc[i].properties[confidence_shift_tag] > max_confidence_shift:
            max_confidence_shift = fc[i].properties[confidence_shift_tag]
            x_shift = fc[i].properties[x_shift_tag] - row[x_shift_tag]
            y_shift = fc[i].properties[y_shift_tag] - row[y_shift_tag]

    if x_shift == 0 and y_shift == 0:
        row[corr_x_shift_tag] = 0
        row[corr_y_shift_tag] = 0
        row[corr_confidence_shift_tag] = -3
        return row

    gen_walls = generate.generate_wall_from_roof_by_x_y(row['geometry'],
                                                        x_shift,
                                                        y_shift,
                                                        closing=closing,
                                                        simplify=simplify)
    iou_walls = calculate_iou(row['walls'], gen_walls)
    if iou_walls > corr_threshold:
        row[corr_x_shift_tag] = x_shift
        row[corr_y_shift_tag] = y_shift
        row[corr_confidence_shift_tag] = iou_walls
    else:
        row[corr_x_shift_tag] = 0
        row[corr_y_shift_tag] = 0
        row[corr_confidence_shift_tag] = -4

    return row


def correct_shifts(fp_fc: FeatureCollection,
                   walls_fc:FeatureCollection,
                   x_shift_tag: str = '_x_shift',
                   y_shift_tag: str = '_y_shift',
                   confidence_shift_tag: str = '_confidence_shift',
                   corr_x_shift_tag: str = '_corr_x_shift',
                   corr_y_shift_tag: str = '_corr_y_shift',
                   corr_confidence_shift_tag: str = '_corr_confidence_shift',
                   corr_threshold: float = 0.5,
                   rt_buffer: float = 1.0,
                   closing: float = 0.0,
                   simplify: float = 0.0) -> FeatureCollection:
    if fp_fc.empty or walls_fc.empty:
        return fp_fc

    fp_fc.to_utm(inplace=True)
    walls_fc.to_utm(inplace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FeatureWarning from gpd.unary_union()
        fp_fc1 = FeatureCollection(fp_fc.apply(select_walls_for_footprint,
                                               walls=walls_fc,
                                               x_shift_tag=x_shift_tag,
                                               y_shift_tag=y_shift_tag,
                                               rt_buffer=rt_buffer))

    fp_fc2 = FeatureCollection(fp_fc1.apply(estimate_correct_shift,
                                            fp_fc=fp_fc,
                                            x_shift_tag=x_shift_tag,
                                            y_shift_tag=y_shift_tag,
                                            confidence_shift_tag=confidence_shift_tag,
                                            corr_x_shift_tag=corr_x_shift_tag,
                                            corr_y_shift_tag=corr_y_shift_tag,
                                            corr_confidence_shift_tag=corr_confidence_shift_tag,
                                            corr_threshold=corr_threshold,
                                            closing=closing,
                                            simplify=simplify))

    fp_fc[:, corr_confidence_shift_tag] = fp_fc2[:, corr_confidence_shift_tag]
    fp_fc[:, corr_x_shift_tag] = fp_fc2[:, corr_x_shift_tag]
    fp_fc[:, corr_y_shift_tag] = fp_fc2[:, corr_y_shift_tag]

    return fp_fc