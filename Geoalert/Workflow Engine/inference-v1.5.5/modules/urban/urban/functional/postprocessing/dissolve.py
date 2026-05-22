from gpdadapter import FeatureCollection
import numpy as np


def dissolve_fc(small_polygons: FeatureCollection,
                big_polygons: FeatureCollection,
                distance: float,
                delete_detached: bool = False) -> FeatureCollection:
    """
    dissolves features from input to the closest features of the anchor
    Args:
        small_polygons: FeatureCollection
        big_polygons: FeatureCollection
        distance:
        delete_detached:

    Returns:
        FeatureCollection
    """
    # The first subarray contains input geometry integer indices.
    # The second subarray contains tree geometry integer indices.
    neighbors_idxs = big_polygons.query(small_polygons.geometry.buffer(distance))
    detached = small_polygons[list(set(range(len(small_polygons))).difference(set(neighbors_idxs[0])))]
    for small_idx in set(neighbors_idxs[0]):
        big_indexes = neighbors_idxs[1][neighbors_idxs[0] == small_idx]
        if len(big_indexes) > 1:
            areas = [small_polygons[small_idx, 'geometry'].intersection(
                big_polygons[big_idx, 'geometry']).area for big_idx in big_indexes]
            fuse_idx = big_indexes[np.argmax(areas)]
        else:
            fuse_idx = big_indexes[0]
        big_polygons[fuse_idx, 'geometry'] = big_polygons[fuse_idx, 'geometry'].union(
            small_polygons[small_idx, 'geometry'])
    if not delete_detached:
        big_polygons.append(detached)
    return big_polygons
