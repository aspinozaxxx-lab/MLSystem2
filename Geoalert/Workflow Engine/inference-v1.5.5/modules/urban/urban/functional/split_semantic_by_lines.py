from sknw import build_sknw
from skimage.morphology import skeletonize
from gpdadapter import FeatureCollection
from scipy.ndimage import binary_erosion
import networkx as nx
import shapely
from shapely.ops import unary_union, polygonize
import numpy as np
from .remove_diagonal_connectivity import remove_diagonal_connectivity
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects


def _extend_linestring(coords: np.ndarray, extend: float = 30):
    """ Extends the first segment of a LineString (represented by its coords as np array) in the
     direction of its angle by 'extend' value.
     Returns the coordinates of a point - new first point of the linestring.
     In order to extend the last segment of a LineString flip the coords"""
    # TODO: calc average weighted angle
    angle = np.arctan2(coords[0, 1] - coords[1, 1], coords[0, 0] - coords[1, 0])
    dx, dy = extend*np.cos(angle), extend*np.sin(angle)
    return np.array((coords[0, 0] + dx, coords[0, 1] + dy))


def skeletonize_pred(pred: np.ndarray,
                     semantic_raster: np.ndarray,
                     thr: float = 70,
                     erode: int = 5,
                     min_edge_len: float = 20,
                     straighten_buffer: float = 4,
                     simplify: float = 2,
                     min_tail_length: float = 15,
                     extend_terminal: float = 15):
    """Turns raster lines into a list of simplified LineStrings
    Args:
        pred: 2d 8bit numpy array with raster lines.
        semantic_raster: semantic mask (2d binary array) which is meant to be split by the lines.
                         Is used here for optimization purpose to crop all 'pred' outside the semantic mask.
        thr: threshold for 'pred'
        erode: erosion iterations for semantic mask to make it smaller before cropping 'pred',
               which helps to reduce border artifacts
        min_edge_len: all non-connected or circular edges shorter than this will be removed
        straighten_buffer: simplification param. Edges which lie completely within this buffer will be replaced with
                           straight lines
        simplify: simplification param for those edges that don't satisfy 'straighten_buffer' condition
        min_tail_length: post-simplification that should remove short 'tails' on the terminal nodes
        extend_terminal: extension length for terminal nodes after simplification
    Returns:
        list of LineStrings - simplified vector representation of 'pred'"""

    semantic_eroded = binary_erosion(semantic_raster, iterations=erode)
    pred = skeletonize((pred * semantic_raster > thr).astype(np.uint8)) * semantic_eroded

    G = build_sknw(pred, iso=False)

    for u, v in list(G.edges()):
        # filter short non-connected edges
        if G.edges[u, v]['weight'] < min_edge_len and G.degree[u] == G.degree[v] == 1:
            G.remove_edge(u, v)
            continue
        # filter short circular edges
        if G.edges[u, v]['weight'] < min_edge_len and u == v:
            G.remove_edge(u, v)
            continue

        coords = G.edges[u, v]['pts']
        # simplify edge as single line if the whole edge lies within 'straighten_buffer'
        if shapely.within(shapely.LineString(coords),
                          shapely.LineString((coords[0], coords[-1])).buffer(straighten_buffer)):
            coords = np.stack((coords[0], coords[-1]))
        # else regular simplification
        else:
            coords = np.array(shapely.LineString(coords).simplify(simplify).coords)

        # replace an edge with a simplified one
        G.add_edge(u, v, pts=coords, weight=shapely.LineString(coords).length)

        # additional filtering after simplification (removes short 'tails')
        if (G.degree[u] == 1 or G.degree[v] == 1) and G.edges[u, v]['weight'] < min_tail_length:
            G.remove_edge(u, v)

    G.remove_nodes_from(list(nx.isolates(G)))  # remove isolated nodes

    # extend terminal nodes
    for u, v in list(G.edges()):
        if G.degree[u] == 1:
            G.add_node(u, o=_extend_linestring(G.edges[u, v]['pts'], extend_terminal))
            coords = np.stack((G.nodes[u]['o'], G.nodes[v]['o']))
            G.add_edge(u, v, pts=coords, weight=shapely.distance(shapely.Point(coords[0]), shapely.Point(coords[-1])))
        if G.degree[v] == 1:
            G.add_node(v, o=_extend_linestring(np.flip(G.edges[u, v]['pts'], 0), extend_terminal))
            coords = np.stack((G.nodes[u]['o'], G.nodes[v]['o']))
            G.add_edge(u, v, pts=coords, weight=shapely.distance(shapely.Point(coords[0]), shapely.Point(coords[-1])))

    return [shapely.LineString(np.flip(G.edges[u, v]['pts'], 1)) for u, v in G.edges]


def polygonize_split(lines: FeatureCollection) -> FeatureCollection:
    merged_lines = unary_union(lines.geometry)
    polygons = list(polygonize(merged_lines))
    return FeatureCollection(polygons, crs=lines.crs)


def dissolve_small_polygons(fc: FeatureCollection, min_area: float = 7, buffer: float = 0.1):
    """ Dissolve polygons smaller than 'min_area' by merging them with neighbour big polygons"""
    idxs = fc.geometry.area < min_area
    small_polygons = fc[idxs].buffer(buffer)
    big_polygons = fc[~idxs]

    # The first subarray contains input geometry integer indices.
    # The second subarray contains tree geometry integer indices.
    idxs = big_polygons.query(small_polygons.geometry.buffer(buffer))

    for small_idx in set(idxs[0]):
        big_indexes = idxs[1][idxs[0] == small_idx]  # all 'big' polygons touching this one
        # if more than one touching polygon, select the one with the biggest intersection area
        if len(big_indexes) > 1:
            areas = [small_polygons[small_idx, 'geometry'].intersection(
                     big_polygons[big_idx, 'geometry']).area for big_idx in big_indexes]
            fuse_idx = big_indexes[np.argmax(areas)]
        else:
            fuse_idx = big_indexes[0]

        # fuse
        big_polygons[fuse_idx, 'geometry'] = big_polygons[fuse_idx, 'geometry'].union(
                     small_polygons[small_idx, 'geometry'])

    return big_polygons


def erosion(x: np.ndarray, iterations: int) -> np.ndarray:
    if iterations > 0:
        return ndi.binary_erosion(x, iterations=iterations)
    return x


def dilation(x: np.ndarray, iterations: int) -> np.ndarray:
    if iterations > 0:
        return ndi.binary_dilation(x, iterations=iterations)
    return x


def watershed_split(semantic: np.ndarray, lines: np.ndarray,
                    threshold: float = 0, semantic_erosion: int = 0,
                    lines_dilation: int = 0, min_size: int=32) -> np.ndarray:
    # extract roof markers from semantic and split lines
    markers = np.clip(
        np.logical_and(erosion(semantic, semantic_erosion),
            np.logical_not(dilation((lines > threshold).astype(np.uint8), iterations=lines_dilation))
        ).astype(np.uint8), 0, 1)

    markers = ndi.label(markers)[0]
    markers = remove_small_objects(markers, min_size=min_size)
    labels = watershed(- semantic, markers=markers, mask=semantic, watershed_line=True)
    return remove_diagonal_connectivity((labels > 0).astype('uint8'))
