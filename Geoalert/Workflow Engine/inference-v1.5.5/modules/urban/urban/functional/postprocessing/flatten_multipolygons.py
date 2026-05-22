from gpdadapter import FeatureCollection


def merge_connected_polygons(fc: FeatureCollection):
    """
    Args:
        fc: input FeatureCollection with polygons

    Returns:
        FeatureCollection, where all intersected polygons are merged and non-polygon features are skipped
        THERE ARE NO PROPERTIES IN THE NEW FEATURES, ONLY GEOMETRY!
    """
    """
    polygons_list = [i.geometry[0] for i in fc if type(i.geometry[0]) == shapely.geometry.polygon.Polygon]
    gc = shapely.ops.unary_union(polygons_list)

    if type(gc) == shapely.geometry.polygon.Polygon:  # If there are one result -> unary_union returns Polygon
        unary_union_geometries = [gc, ]
    elif len(gc.geoms) > 1:  # If there are many results -> unary_union returns geometry collection
        unary_union_geometries = list(gc.geoms)
    else:  # If we have nor one result and nor many results -> zero results
        unary_union_geometries = []
        
    features = [g for g in unary_union_geometries]
    return FeatureCollection(features, crs=fc.crs)"""
    return FeatureCollection(fc.merge_connected_geometries())


def flatten_multipolygons(fc: FeatureCollection) -> FeatureCollection:
    """
    Args:
        fc: input FeatureCollection
    Returns:
        FeatureCollection with the same geometry but all MultiPolygons split to separate Polygon features
        Properties are not affected (!)
        Non-polygon/MultiPolygon features are removed
    """
    """# TODO: use geopandas explode() !
    new_features = []
    for feat in fc:
        # split the multipolygons if they appear in results, also deleting too small areas
        if isinstance(feat.geometry[0], shapely.geometry.Polygon):
            new_features.append(feat)
        elif isinstance(feat.geometry[0], shapely.geometry.MultiPolygon):
            new_features += [FeatureCollection(p, crs=fc.crs) for p in feat.geometry[0].geoms]
        # else skip the feature entirely
    return FeatureCollection(new_features, crs=fc.crs)"""
    return fc.explode()
