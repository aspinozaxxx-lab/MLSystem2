import cv2
import numpy as np
from itertools import tee

import shapely
from sknw import build_sknw
from networkx import MultiGraph, Graph
from shapely.ops import unary_union
from shapely.geometry import LineString, Polygon
from skimage.draw import disk
from typing import Union, List

# ================  workflow steps=================== #


def skelet2linestrings(skelet_band, approx, min_terminal_length):
    """
    from skeletonized band to linestrings following the skeleton
    Args:
        skelet_band:
        approx:
        min_terminal_length:

    Returns:

    """
    skelet = skelet_band.numpy()
    G = skelet2graph(skelet, min_terminal_length)
    node_lines = graph2lines(G)
    arcs = extract_arcs(node_lines, G, approx)
    linestrings = [LineString(arc) for arc in arcs]
    return linestrings


def linestrings2polygons(linestrings: List[shapely.LineString], mask_band,
                         min_width, max_width, merge) -> List[shapely.Polygon]:
    
    """
    make polygons from linestrings, inflating each string to width estimated from mask_band
    Args:
        linestrings:
        mask_band:
        min_width:
        max_width:
        merge:

    Returns:

    """
    if min_width == max_width:
        width = min_width
    else:
        mask = mask_band.numpy()
        # The widths are estimated also in raster coordinates
        # However the max and min width are in projected CRS
        # average over x and y resolution (units per pixel). Normally, they are equal, but not necessary
        
        avg_res = ((mask_band.res[0] + mask_band.res[1]) / 2)
        pixel_min_w = min_width / avg_res
        pixel_max_w = max_width / avg_res
        width = _find_roads_width(mask, linestrings, pixel_min_w, pixel_max_w)
        
        # transform width in pixels back to the projection units as we need it later
        width = [w * avg_res for w in width]

    return inflate_lines(linestrings, width, merge=merge)


# ====================================================================== #

# =================== utils ======================= #
def _pairwise(iterable):
    # "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _flatten(l):
    return [item for sublist in l for item in sublist]


# ==================== graph normalization ============= #
def _add_direction_change_nodes(pts, s, e, s_coord, e_coord, approx: float = 2):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord

    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)

    return ps


def _remove_small_terminal(G, threshold=10):
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = _flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
        vals = _flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < threshold:
                G.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < threshold:
                G.remove_node(e)
    return G

# ========================= substeps ========================== #


def skelet2graph(skelet, min_terminal_length=10) -> Union[MultiGraph, Graph]:
    """
    makes shapely lines from the skeleton of the mask

    Args:
        skelet:
        min_terminal_length:

    Returns:

    """
    G = build_sknw(skelet, multi=True)
    G = _remove_small_terminal(G, min_terminal_length)
    return G


def graph2lines(G: Union[MultiGraph, Graph]) -> List:
    node_lines = []
    edges = list(G.edges())

    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def extract_arcs(node_lines: List, G: Union[MultiGraph, Graph], approx: float = 2):
    terminal_points = [i for i, d in dict(G.degree()).items() if d == 1]
    terminal_lines = {}
    vertices = []
    for w in node_lines:
        coord_list = []
        additional_paths = []
        for s, e in _pairwise(w):
            vals = _flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = G.nodes[s]['o'], G.nodes[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = _add_direction_change_nodes(pts, s, e, s_coord, e_coord, approx)
                # ps = pts
                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)
    return vertices


def inflate_lines(linestrings: List[LineString], width=5, merge=False, resolution=1, cap_style=1, join_style=1):
    if not linestrings:
        return []
    if isinstance(width, list):
        assert len(width) == len(linestrings), 'If the widths is a list, the width must be set for every linestring'
        polys = [ls.buffer(w / 2, resolution=resolution, cap_style=cap_style, join_style=join_style) for ls, w in
                 zip(linestrings, width)]
    else:
        # Check for the width to be numeric?
        polys = [ls.buffer(width / 2, resolution=resolution, cap_style=cap_style, join_style=join_style) for ls in
                 linestrings]

    if merge:
        polys = unary_union(polys)
    if isinstance(polys, Polygon):
        polys = [polys]
    return polys

# ========================= lines2fc internals ========================= #


def _estimate_width_around_point(mask, point, radius):
    """
    We estimate the width:
        - if the width is less than radius of the circle, the good estimation is N/2R where N is area of road pixels
        - if the width is more, then we assume it to be equal to radius.
        So, the radius must be not less than maximum allowed road width
    """
    area = disk(point, radius, shape=mask.shape)
    width = np.count_nonzero(mask[area]) / (2 * radius)

    return min(width, radius)


def _estimate_segment_width(mask: np.ndarray, line: np.ndarray, radius):
    """
    We try to estimate the real road width in the mask, counting white pixels in circumference of the points
    throughout the segment.
    We do not want to measure the width at the ends of the segment because crossroads can be segmented not very
    carefully, and the road is influenced by the neighbours.
    So:
        If the segment consists of more than 2 points, we use the internal direction-change-points to estimate the width
        If the segment has only the start and the end, we find the center and estimate the width around this point
    """

    if len(line) == 2:
        p = (np.array(line[0]) + np.array(line[1])) / 2
        return _estimate_width_around_point(mask, p, radius)
    else:
        estimates = []
        for p in line[1:-1]:
            estimates.append(_estimate_width_around_point(mask, p, radius))
        return np.mean(estimates)


def _find_roads_width(mask: np.ndarray, linestrings: List[shapely.LineString], min_width, max_width) -> List[float]:
    radius = max_width
    width = []
    for ls in linestrings:
        w = _estimate_segment_width(mask, ls.coords, radius)
        width.append(max(w, min_width))
    return width
# =========================   ========================= #
