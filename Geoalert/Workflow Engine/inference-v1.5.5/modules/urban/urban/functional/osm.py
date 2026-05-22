import overpass
import shapely
from typing import Dict, Hashable, List, Tuple
from gpdadapter import FeatureCollection
from loguru import logger
from pathlib import Path

ENDPOINTS = ["https://z.overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.openstreetmap.fr/api/interpreter",
             "https://overpass.nchc.org.tw",
             "https://lz4.overpass-api.de/api/interpreter",
             "https://1.overpass.kumi.systems/api/interpreter",
             "https://2.overpass.kumi.systems/api/interpreter",
             "https://3.overpass.kumi.systems/api/interpreter",
             "https://4.overpass.kumi.systems/api/interpreter"]

ENDPOINTS_FILE = Path('./osm_endpoints')


def ls_to_polygon(ls: dict):
    """
    Convert LineString objects to polygon
    Args:
        ls: LineString GeoJSON - like object

    Returns:
        polygon: GeoJSON-like object
    """
    if ls['type'] == 'LineString':
        polygon = {
            'type': 'Polygon',
            'coordinates': [ls['coordinates']]
        }
    else:
        raise ValueError('Non-supported geometry type')
    return polygon


def update_endpoints(endpoints: list, n_failed_endpoints: int):
    assert 0 <= n_failed_endpoints <= len(endpoints)
    if not n_failed_endpoints:
        # everything is fine, no need to update
        logger.trace("Everything is fine, no need to update")
        return
    elif len(endpoints) == n_failed_endpoints:
        # does not work at all!
        logger.error("Every OSM endpoint failed!")
        return
    new_endpoints = endpoints[n_failed_endpoints:] + endpoints[:n_failed_endpoints]
    logger.trace(f"Moving {endpoints[:n_failed_endpoints]} down the list. New list is {new_endpoints}")
    return [endpoint + '\n' for endpoint in new_endpoints]


def query_osm(query: str, timeout: int) -> dict:
    # We create the file to store actual list of endpoints
    if ENDPOINTS_FILE.exists():
        endpoints = [file.strip() for file in open(ENDPOINTS_FILE).readlines()]
    else:
        endpoints = ENDPOINTS

    n_failed_endpoints = 0

    # return empty gj in case everything failed
    result = {"features": []}

    for endpoint in endpoints:
        try:
            logger.debug(f"Querying OSM at {endpoint}")
            api = overpass.API(endpoint=endpoint, timeout=timeout)
            result = api.get(query, verbosity='geom')
        except Exception as e:
            logger.warning(f'Loading OSM features with {endpoint} endpoint failed : {e}.')
            n_failed_endpoints += 1
        else:
            logger.debug(f"Got results from OSM {endpoint}")
            break

    new_endpoints = update_endpoints(endpoints, n_failed_endpoints)
    if new_endpoints:
        with open(ENDPOINTS_FILE, 'w') as dst:
            dst.writelines(new_endpoints)

    return result


def load_osm_buildings(bbox: Tuple[float, float, float, float], timeout: int = 600) -> FeatureCollection:
    """

    Args:
        bbox: 4 points, Left-Bot format (left, bottom, right, top)
        timeout: time to wait server response, seconds

    Returns:
        fc: Feature collection of Buildings footprints
    """

    query = 'way["building"]' + str(bbox) + ';'
    gj = query_osm(query, timeout=timeout)

    # format features
    features = []
    for f in gj['features']:
        try:
            f['geometry'] = shapely.geometry.shape(ls_to_polygon(f['geometry'])).simplify(0)
            features.append(f)
        except Exception as e:
            logger.debug("Skipping geometry with error `{}`.".format(e))

    logger.debug(f"Got {len(features)} features from OSM")
    return FeatureCollection(features)


def load_osm_roads(bbox: Tuple[float, float, float, float], timeout: int = 600) -> FeatureCollection:
    """

    Args:
        bbox: 4 points, Left-Bot format (left, bottom, right, top)
        timeout: time to wait server response, seconds

    Returns:
        fc: Feature collection of osm HighWay
    """

    query = 'way["highway"]' + str(bbox) + ';'
    gj = query_osm(query, timeout=timeout)

    # format features
    features = []
    for f in gj['features']:
        geom = f['geometry']
        if geom['type'] == 'LineString' and len(geom['coordinates']) > 1:
            f['geometry'] = shapely.geometry.shape(geom).simplify(0)  # remove unnecessary points
            features.append(f)

    return FeatureCollection(features)


def load_osm_landuse(bbox: Tuple[float, float, float, float], timeout: int = 600) -> FeatureCollection:
    """

    Args:
        bbox: 4 points, Left-Bot format (left, bottom, right, top)
        timeout: time to wait server response, seconds

    Returns:
        fc: Feature collection of landuse features
    """

    query = 'way["landuse"]' + str(bbox) + ';'
    gj = query_osm(query, timeout=timeout)

    # format features
    features = []
    for f in gj['features']:
        try:
            f['geometry'] = shapely.geometry.shape(ls_to_polygon(f['geometry'])).simplify(0)
            features.append(f)
        except Exception as e:
            logger.debug("Skipping geometry with error `{}`.".format(e))

    return FeatureCollection(features)


def map_osm_landuse_(
        fc: FeatureCollection,
        osm_landuse_fc: FeatureCollection,
        class_mapping: Dict[Hashable, List[Hashable]],
        tag: str = "class_id",
):
    """Correct classes in fc with OSM data
    
        class_mapping: dict with classes correspondence to OSM. 
            e.g. {101: ["residential"], 103: ["commercial"]}
    """
    if 'landuse' not in osm_landuse_fc.columns or osm_landuse_fc.empty or fc.empty:
        return fc
    for target_cls, osm_classes in class_mapping.items():
        for osm_cls in osm_classes:
            cls_osm_subset = osm_landuse_fc[osm_landuse_fc[:, 'landuse'] == osm_cls]
            for osm_cls_feature_idx in cls_osm_subset:
                intersection_indexes = fc.query(cls_osm_subset[osm_cls_feature_idx, 'geometry'])
                for intersection_index in intersection_indexes:
                    coverage = fc[intersection_index, 'geometry'].intersection(
                        cls_osm_subset[osm_cls_feature_idx, 'geometry']).area / fc[intersection_index, 'geometry'].area
                    if coverage > 0.6:
                        fc[intersection_index, tag] = target_cls
                        fc[intersection_index, "osm_landuse_class"] = osm_cls
    return fc
