import json
import requests
from gpdadapter import FeatureCollection
from collections import defaultdict
from typing import Dict, List, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from .postprocessing.shapely_ext import intersection_over_union


def process_fc_with_josm(fc: FeatureCollection, endpoint_url: str, timeout: int = 120) -> FeatureCollection:
    
    # prepare data
    fc[:, "_josm_id"] = [str(i) for i in range(len(fc))]
    properties_dict = {f.properties["_josm_id"]: f.properties for f in fc}

    data = fc.to_json_dict()
    data.pop("crs", None)  # remove CRS because josm expect CRS object
    data = json.dumps(data).encode()
    
    # make request to server
    n_attempts = 3
    for attempt in range(n_attempts): # 3 retries
        try:
            response = requests.post(
                endpoint_url, 
                headers={'cache-control': "no-cache"}, 
                files={"file": data}, 
                timeout=timeout,
            )
            break
        except Exception as e:
            if attempt == n_attempts - 1:
                raise e 
    
    # parse response
    gj = json.loads(response.content.decode(encoding="utf-8"))
    crs = "epsg:4326"
    features = gj.get("features", [])

    filtered_features = []
    for f in features:
        if f['properties'] is not None and f['geometry'].is_valid:
            id = f['properties']["_josm_id"]
            f['properties'] = properties_dict[id]
            f['properties'].pop("_josm_id", None)
            filtered_features.append(f)

    return FeatureCollection(filtered_features, crs=crs)


def make_fc_dict(fc: FeatureCollection, tag: str) -> Dict[str, list]:  # TODO: Not used
    """Group feature collection by provided tag in Dict"""
    data = defaultdict(list)
    for f in fc:
        id_ = f.properties.get(tag, None)
        if id_ is not None:
            data[id_].append(f)
    return data


def union_features(features: List[Polygon]) -> Union[Polygon, MultiPolygon]:    # TODO: Not used
    """Stick list of features together"""
    if len(features) == 1:
        result = features[0]
    else:
        result = unary_union(features)
    return result


def replace_blocks_by_josm(
    fc: FeatureCollection,
    josm_fc: FeatureCollection,
    union_tag: str, 
    iou_threshold: float, 
    hausdorff_threshold: float
) -> FeatureCollection:
    """Replace original features by features simplified with JOSM if criteria are satisfied:
    iou > iou_threshold and hausdorff_distance < hausdorff_threshold"""
    
    if union_tag is None:
        union_tag = "_block_id"
    fc[:, union_tag] = range(len(fc))

    all_init_features = fc.groupby(union_tag)
    all_josm_features = josm_fc.groupby(union_tag)
    
    features = FeatureCollection()
    
    for k in all_init_features.keys():
            
        init_features = all_init_features[k]
        josm_features = all_josm_features.get(k, None)
        
        if josm_features is None:
            features.append(init_features)
            continue
        
        # union split shapes to compare "blocks"
        init_shape = unary_union(init_features.geometry)
        josm_shape = unary_union(josm_features.geometry)
        
        # compute metrics over "blocks"
        iou = intersection_over_union(init_shape, josm_shape, ignore_errors=True)
        hausdorff = init_shape.hausdorff_distance(josm_shape)
        
        if iou > iou_threshold and hausdorff < hausdorff_threshold:
            josm_features[:, "_josm"] = True
            features.append(josm_features)
        else:
            features.append(init_features)

    return features
