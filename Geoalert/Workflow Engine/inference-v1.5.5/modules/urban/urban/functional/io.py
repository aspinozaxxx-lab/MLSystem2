import os
from gpdadapter import FeatureCollection
from aeronet_raster import BandCollection
from aeronet_raster.utils.utils import parse_directory
from ..base.defaults import GEOM_TYPES
from pyproj import CRS
from typing import Sequence, Union
from loguru import logger

def read_bc(path: str, bands: Sequence[str]) -> BandCollection:
    bands = parse_directory(path, bands)  # TODO: fix annotation in parse_directory() Tuple[str] to Sequence[str]
    bc = BandCollection(bands)
    logger.debug(f"Read BandCollection of shape {bc.shape}")
    return bc

def read_fc(path: str,
            name: str,
            crs: Union[None, CRS, str] = None,
            make_valid: bool = True,
            dropna: bool = True,
            drop_empty: bool = True,
            explode: bool = False,
            remove_repeated_points: bool = True,
            keep_only_geometry_types: Union[type, str, Sequence[str], Sequence[type], None] = None) -> FeatureCollection:
    """Postprocess and save fc as geojson
    Args:
        path - path to directory
        name - filename (without extension)
        crs - if not None, force reproject to CRS
        make_valid - if True, runs make_valid() before saving
        dropna - drop NaN or None values (only if make_valid=True)
        drop_empty - drop empty geometries (only if make_valid=True)
        explode - explode GeometryCollections (only if make_valid=True)
        remove_repeated_points - remove repeated points with 0 tolerance (only if make_valid=True)
        keep_only_geometry_types - types of geometry to save (Polygon, LineString, etc.). (TYPES, NOT STRINGS!)
    Returns:
        FeatureCollection
    """
    fp = os.path.join(path, '{}.geojson'.format(name))
    fc = FeatureCollection.from_file(fp)
    if not fc.empty:
        if make_valid:
            fc = fc.make_valid(dropna=dropna, drop_empty=drop_empty, explode=explode,
                               remove_repeated_points=remove_repeated_points)
        if keep_only_geometry_types:
            if not isinstance(keep_only_geometry_types, (list, tuple)):
                keep_only_geometry_types = [keep_only_geometry_types, ]
            keep_only_geometry_types = [GEOM_TYPES[x.lower()] if isinstance(x, str) else x for x in
                                        keep_only_geometry_types]
            fc = fc.filter(lambda x: isinstance(x.geometry, tuple(keep_only_geometry_types)))
        if crs is not None:
            fc.to_crs(crs, inplace=True)
    return fc


def save_fc(fc: FeatureCollection,
            path: str,
            name: str,
            hold_crs: bool = False,
            make_valid: bool = True,
            dropna: bool = True,
            drop_empty: bool = True,
            explode: bool = False,
            remove_repeated_points: bool = True,
            keep_only_geometry_types: Union[type, str, Sequence[str], Sequence[type], None] = None):
    """Postprocess and save fc as geojson
    Args:
        fc - FeatureCollection
        path - path to directory
        name - filename (without extension)
        hold_crs - if False, force reproject to latlon
        make_valid - if True, runs make_valid() before saving
        dropna - drop NaN or None values (only if make_valid=True)
        drop_empty - drop empty geometries (only if make_valid=True)
        explode - explode GeometryCollections (only if make_valid=True)
        remove_repeated_points - remove repeated points with 0 tolerance (only if make_valid=True)
        keep_only_geometry_types - types of geometry to save (Polygon, LineString, etc.). (TYPES, NOT STRINGS!)
    """
    fp = os.path.join(path, '{}.geojson'.format(name))
    if not fc.empty:
        if not hold_crs:
            fc.to_crs('EPSG:4326', inplace=True)
        if make_valid:
            fc = fc.make_valid(dropna=dropna, drop_empty=drop_empty, explode=explode,
                               remove_repeated_points=remove_repeated_points)
        if keep_only_geometry_types:
            if not isinstance(keep_only_geometry_types, (list, tuple)):
                keep_only_geometry_types = [keep_only_geometry_types, ]
            keep_only_geometry_types = [GEOM_TYPES[x.lower()] if isinstance(x, str) else x for x in
                                        keep_only_geometry_types]
            fc = fc.filter(lambda x: isinstance(x.geometry, tuple(keep_only_geometry_types)))
    fc.to_file(fp, hold_crs=True)
