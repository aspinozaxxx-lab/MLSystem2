from shapely.geometry import Polygon, shape
import numpy as np


def _get_mgs_poly(cell: str,
                  metadata_csv_path) -> Polygon:
    """Get mgs polygon

    Args:
        cell (str): Militay Grid System string
        metadata_csv_path (str, optional): Path to MGS csv. Defaults to ''.

    Returns:
        Polygon: shapely polygon EPSG:4326 of corresponging MGS cell.
    """
    mgs_names = np.loadtxt(metadata_csv_path, delimiter=',', skiprows=1, dtype=str, usecols=0)
    mgs_polygons = np.loadtxt(metadata_csv_path, delimiter=',', skiprows=1, dtype=np.float64, usecols=np.arange(1, 11))
    try:
        name_index = np.where(mgs_names == cell)[0][0]
    except IndexError:
        raise ValueError(f"Cell {cell} is not found in MGRS index file")
    mgs_poly = Polygon(mgs_polygons[name_index].reshape((-1, 2)))
    return mgs_poly


def aoi_intersects_cell(cell: str, aoi: dict, metadata_csv_path: str):
    cell_polygon = _get_mgs_poly(cell, metadata_csv_path)
    return cell_polygon.intersects(shape(aoi))
