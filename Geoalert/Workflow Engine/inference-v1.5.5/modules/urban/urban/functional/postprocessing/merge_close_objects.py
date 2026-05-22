from .flatten_multipolygons import merge_connected_polygons
from gpdadapter import FeatureCollection


def merge_close_objects(fc: FeatureCollection, max_distance: float) -> FeatureCollection:
    if max_distance <= 0:
        return fc
    # make features wider by max_distance/2 so that the close features will touch each other

    buffered_fc = fc.buffer(max_distance/2)
    # merge overlapping objects
    merged_fc = merge_connected_polygons(buffered_fc).buffer(-max_distance/2)
    return merged_fc
