from urban.bricks.buildings_postprocessing import CorrectTopology
from urban import Compose
from gpdadapter import FeatureCollection
from pathlib import Path
import pytest


@pytest.fixture
def filepaths():
    output = "two_corrected_polygons"
    folder = Path("./tests/test_data/synthetic/topology")
    output_file = folder/(output+".geojson")
    yield output, folder, output_file
    output_file.unlink()


def test_correct_topology_polygons(filepaths):
    """
    Take two intersecting Polygons and see if they will not intersect after CorrectTopology
    """
    output, folder, output_file = filepaths
    pipeline = Compose(bricks=[CorrectTopology(input="two_polygons", output=output)])

    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    fc = FeatureCollection.from_file(output_file)
    intersects = False
    for idx1 in range(len(fc)):
        feature = fc.iloc[idx1].geometry
        for idx2 in range(idx1+1, len(fc)):
            other = fc.iloc[idx2].geometry
            if feature != other:
                intersects = intersects or feature.intersects(other)

    # see that the result is expected
    assert not intersects


def test_correct_topology_multipolygons(filepaths):
    """
    Take intersecting Polygon and Multipolygon and see if they will not intersect after CorrectTopology
    """
    output, folder, output_file = filepaths
    pipeline = Compose(bricks = [CorrectTopology(input="poly_and_multipoly", output=output)])

    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    fc = FeatureCollection.from_file(output_file)
    intersects = False
    for idx1 in range(len(fc)):
        feature = fc.iloc[idx1].geometry
        for idx2 in range(idx1+1, len(fc)):
            other = fc.iloc[idx2].geometry
            if feature != other:
                intersects = intersects or feature.intersects(other)

    # see that the result is expected
    assert not intersects


"""def test_correct_topology_not_works_multipolygons(filepaths):
    
    Take intersecting Polygon and Multipolygon and see if they will intersect
    after CorrectTopology without MultiPolygon split
    

    output, folder, output_file = filepaths
    pipeline = Compose([CorrectTopology("poly_and_multipoly", output=output, flatten_multipolygons=False)])

    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    fc = FeatureCollection.from_file(output_file)
    intersects = False
    for idx1 in range(len(fc)):
        feature = fc.iloc[idx1].geometry
        for idx2 in range(idx1+1, len(fc)):
            other = fc.iloc[idx2].geometry
            if feature != other:
                intersects = intersects or feature.intersects(other)

    # see that the result is expected
    assert intersects"""


def test_topology_correction_by_subtraction_of_smaller(filepaths):
    """

    """

    output, folder, output_file = filepaths
    input = "two_polygons"
    input_file = folder/(input+".geojson")
    pipeline = Compose(bricks = [CorrectTopology(input=input,
                                                 output=output,
                                                 distance_step=0.0,
                                                 flatten_multipolygons=True,
                                                 correct_by_subtraction=1,
                                                 buffer=0.1)])

    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    fc = FeatureCollection.from_file(output_file)
    intersects = False
    for idx1 in range(len(fc)):
        feature = fc.iloc[idx1].geometry
        for idx2 in range(idx1+1, len(fc)):
            other = fc.iloc[idx2].geometry
            if feature != other:
                intersects = intersects or feature.buffer(-0.001).intersects(other.buffer(-0.001))

    # see that the result is expected
    assert len(fc) == 2
    assert not intersects

    # see that the BIGGER area became less AND THE SMALLER ONE DID NOT CHANGE
    input_fc = FeatureCollection.from_file(input_file)
    input_areas = [feat.geometry[0].area for feat in input_fc]
    output_areas = [feat.geometry[0].area for feat in fc]
    assert max(input_areas) > max(output_areas)
    assert min(input_areas) == pytest.approx(min(output_areas))


def test_topology_correction_by_subtraction_of_bigger(filepaths):
    """

    """

    output, folder, output_file = filepaths
    input = "two_polygons"
    input_file = folder/(input+".geojson")
    pipeline = Compose(bricks = [CorrectTopology(input=input,
                                                 output=output,
                                                 distance_step=0.0,
                                                 flatten_multipolygons=True,
                                                 correct_by_subtraction=-1,
                                                 buffer=0.1)])

    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    fc = FeatureCollection.from_file(output_file)
    intersects = False
    for idx1 in range(len(fc)):
        feature = fc.iloc[idx1].geometry
        for idx2 in range(idx1+1, len(fc)):
            other = fc.iloc[idx2].geometry
            if feature != other:
                intersects = intersects or feature.buffer(-0.001).intersects(other.buffer(-0.001))

    # see that the result is expected
    assert len(fc) == 2
    assert not intersects

    # see that the smaller area became less AND THE BIGGER ONE DID NOT CHANGE
    input_fc = FeatureCollection.from_file(input_file)
    input_areas = [feat.geometry[0].area for feat in input_fc]
    output_areas = [feat.geometry[0].area for feat in fc]
    assert max(input_areas) == pytest.approx(max(output_areas))
    assert min(input_areas) > min(output_areas)
