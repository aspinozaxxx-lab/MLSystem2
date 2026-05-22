import shapely
import shapely.geometry
from gpdadapter import FeatureCollection

EPSILON = 0.00001


def simple_fc_difference(fc1: FeatureCollection, fc2: FeatureCollection, flat_multipolygons=False, allow_fails=False):
    """
    Simplified version of fc_difference without merge-back and flattening
    """
    if fc1.empty:
        return fc2
    if fc2.empty:
        return fc1
    assert fc1.crs == fc2.crs
    return fc1.overlay(fc2, how='difference', make_valid=True, keep_geom_type=True).make_valid(
        explode=flat_multipolygons)


def fc_difference(fc1: FeatureCollection, fc2: FeatureCollection, verbose=False, flat_multipolygons=False,
                  area_threshold=25, compactness_threshold=0.2, rectangle_ratio_threshold=5,
                  allow_fails=False, simplify=EPSILON):
    """Subtract features of fc2 from features of fc1. If flat_multipolygon is enabled -
    resulted MultiPolygon features become Polygons with same properties

    Also, the new polygons that meet the merge_criteria are either merged to the nearest features back
    or removed if they cannot be merged
    """
    if fc1.empty:
        return fc2
    if fc2.empty:
        return fc1
    assert fc1.crs == fc2.crs
    return fc1.overlay(fc2, how='difference', make_valid=True, keep_geom_type=True).make_valid(
        explode=flat_multipolygons)


def merge_parts_back(f_diff, f_inter, f_raw, merge_criteria):
    # we will apply the difference operation only if the result is suitable
    # What are we doing: if some feature does not meet the requirements, we find the intersection part
    # that was used to cut it off the closest part, and glue them back together
    diff = f_diff.geometry
    inter = f_inter.geometry
    assert isinstance(diff, shapely.geometry.MultiPolygon) or isinstance(diff, shapely.geometry.Polygon)

    if isinstance(diff, shapely.geometry.Polygon):
        # only one part and it is bad
        if merge_criteria(diff):
            return f_raw
        else:
            merged = diff

    else:
        new_diff = shapely.geometry.MultiPolygon([])
        for part in diff:
            if merge_criteria(part):
                new_diff = new_diff.union(merge_part(part, diff, inter, merge_criteria))
            else:
                new_diff = new_diff.union(part)

        merged = new_diff
    return merged


def merge_part(part, diff, inter, merge_criteria):
    # The small part is merged to the adjacent intersection parts,
    # and then to the adjacent difference parts
    assert isinstance(diff, shapely.geometry.MultiPolygon)
    if isinstance(inter, shapely.geometry.Polygon):
        new_part = part.union(inter)
    else:
        new_part = shapely.geometry.Polygon(part)
        for inter_part in inter:
            if inter_part.intersects(part):
                new_part = new_part.union(inter_part)

    # If there are more than two good parts to be merged together (which we do not want to do) we cannot decide
    # to which part should we attach the selected one, so then we will just remove it.
    #
    part_counter = 0
    res = shapely.geometry.Polygon()
    for diff_part in diff:
        if diff_part.intersects(new_part) and diff_part != part:
            res = new_part.union(diff_part)
            if not merge_criteria(diff_part):
                part_counter += 1

    if part_counter <= 1:
        return res
    else:
        return shapely.geometry.Polygon()
