import os
import json
import pytest
from urban import Compose
from urban.bricks import Merge
from gpdadapter import FeatureCollection

DATA_PATH = 'tests/test_data/tmp'


def write_to_geojson(folder, name, coordinates, geometry_type="Polygon"):
    collection = {"type": "FeatureCollection",
                  "features":
                      [
                          {
                              "type": "Feature",
                              "properties": {},
                              "geometry":
                                  {
                                      "type": geometry_type,
                                      "coordinates": coordinates
                                  }
                          }
                      ]
                  }
    with open(os.path.join(folder, name+'.geojson'), 'w') as dst:
        dst.write(json.dumps(collection))


@pytest.fixture
def generate_intersecting_geometries():
    os.makedirs(DATA_PATH, exist_ok=True)

    input_vectors = ["1", "2", "3"]
    output_vector = "merged"

    geometries = [[[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]],  # Big square
                  [[[0, 0], [0, 1], [20, 1], [20, 0], [0, 0]]],  # Horizontal stripe - longer
                  [[[0, 0], [0, 10], [1, 10], [1, 0], [0, 0]]]]  # Vertical stripe - shorter
    for file, geom in zip(input_vectors, geometries):
        write_to_geojson(DATA_PATH, file, geom)

    yield input_vectors, output_vector, DATA_PATH

    for file in input_vectors:  # + output_vector
        os.remove(os.path.join(DATA_PATH, file+'.geojson'))


def test_merge_without_difference(generate_intersecting_geometries):
    input_vectors, output_vector, folder = generate_intersecting_geometries

    pipeline = Compose(bricks=[Merge(input_vectors=input_vectors,
                                     output_vector=output_vector,
                              )])
    pipeline(folder)

    fc = FeatureCollection.from_file(os.path.join(folder, output_vector+'.geojson'))
    assert len(fc) == 3
    assert fc[0].geometry[0].area == 100
    assert fc[1].geometry[0].area == 20
    assert fc[2].geometry[0].area == 10


def test_merge_with_difference(generate_intersecting_geometries):
    input_vectors, output_vector, folder = generate_intersecting_geometries

    pipeline = Compose(bricks=[Merge(input_vectors=input_vectors,
                                     output_vector=output_vector,
                                     subtract={"1": ["2", "3"], "2": ["3"]})])
    pipeline(folder)

    fc = FeatureCollection.from_file(os.path.join(folder, output_vector+'.geojson'))
    assert len(fc) == 3
    assert fc[0].geometry[0].area == pytest.approx(81, abs=0.1)
    assert fc[1].geometry[0].area == pytest.approx(19, abs=0.1)
    assert fc[2].geometry[0].area == 10


def test_validate_subtraction():
    assert Merge.validate_subtraction(["1", "2", "3"], []) == {}
    assert Merge.validate_subtraction(["1", "2", "3"], None) == {}

    assert Merge.validate_subtraction(["1", "2", "3"], {"1": ["2", "3"], "2": ["3"]}) == {"1": ["2", "3"], "2": ["3"]}

    with pytest.raises(TypeError):
        # subtract 'value' must be a list
        Merge.validate_subtraction(["1", "2", "3"], {"1": "2"})
    with pytest.raises(ValueError):
        # Key not in input
        Merge.validate_subtraction(["1", "2", "3"], {"4": ["2", "3"], "2": ["3"]})
    with pytest.raises(ValueError):
        # Value not in input
        Merge.validate_subtraction(["1", "2", "3"], {"1": ["5", "3"], "2": ["3"]})
    with pytest.raises(ValueError):
        # subtract from itself
        Merge.validate_subtraction(["1", "2", "3"], {"1": ["1", "3"], "2": ["3"]})
    with pytest.raises(ValueError):
        # circular
        Merge.validate_subtraction(["1", "2", "3"], {"1": ["2", "3"], "2": ["1"]})