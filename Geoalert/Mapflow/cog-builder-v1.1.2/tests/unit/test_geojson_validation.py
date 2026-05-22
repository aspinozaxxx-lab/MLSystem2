import pytest
import geojson
from cog_builder.functional.geometry import maybe_valid_geometry
from cog_builder.functional.errors import CogInvalidAOI
from shapely.geometry import mapping

def test_valid_polygon_is_accepted():
    valid_geojson = {"type": "Polygon", "coordinates":(((1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0,1.0), (1.0,1.0)),)}
    assert mapping(maybe_valid_geometry(valid_geojson)) == valid_geojson #geojson.Polygon([[[1, 1], [1, 0], [0, 0], [0, 1], [1, 1]]])


def test_valid_multipolygon_is_accepted():
    valid_geojson = {"type": "MultiPolygon", "coordinates":[(((1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0,1.0), (1.0,1.0)),)]}
    assert mapping(maybe_valid_geometry(valid_geojson)) == valid_geojson #geojson.MultiPolygon([[[[1, 1], [1, 0], [0, 0], [0, 1], [1, 1]]]])

# when shapely is present
def test_invalid_polygon_is_validated():
    invalid_geojson = {"type": "Polygon", "coordinates":(((1.0, 1.0),
                                                          (1.0, 0.0),
                                                             (0.0, 0.0),
                                                             (0.0,2.0),
                                                             (0.0, 1.0),
                                                             (1.0,1.0)),)}
    valid_geojson = {"type": "Polygon", "coordinates": (((0.0, 1.0),
                                                           (1.0, 1.0),
                                                           (1.0, 0.0),
                                                           (0.0, 0.0),
                                                           (0.0, 1.0)),)}
    assert mapping(maybe_valid_geometry(invalid_geojson)) == valid_geojson

def test_valid_multipoint_is_rejected():
    invalid_geojson = {"type": "MultiPoint", "coordinates":[[1,1],[1,0],[0,0]]}
    with pytest.raises(CogInvalidAOI):
        maybe_valid_geometry(invalid_geojson)


def test_invalid_json_is_rejected():
    # Todo: test for invalid jsons
    # problem is that the geojson lib returns exception with input included and makes the error message to fail
    #invalid_json = {"total": "bullshit"}
    invalid_json = "Not A Json"
    with pytest.raises(CogInvalidAOI):
        maybe_valid_geometry(invalid_json)


def test_invalid_polygon_is_rejected():
    # not enough levels of nesting
    invalid_json = {"type": "Polygon","coordinates": [[1,1], [1,0], [0,0], [0,1], [1,1]]}
    with pytest.raises(CogInvalidAOI) as e:
        maybe_valid_geometry(invalid_json)


def test_non_geometry_is_rejected():
    # feature, not geometry
    feature_geojson = {"type":"Feature", "geometry":{"type":"Polygon","coordinates":[[[1,1],[1,0],[0,0],[0,1],[1,1]]]}}
    assert geojson.Feature(feature_geojson).is_valid  # We check that it _is_ valid geojson (but not geometry)
    with pytest.raises(CogInvalidAOI) as e:
        maybe_valid_geometry(feature_geojson)
