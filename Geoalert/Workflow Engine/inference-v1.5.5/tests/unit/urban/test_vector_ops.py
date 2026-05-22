import pytest
from pathlib import Path
from urban import Compose
from urban.bricks.vector_ops import FilterNarrowObjects, RemoveOverlappingObjects
from gpdadapter import FeatureCollection


"""================= test FilterNarrowObjects ===================="""


@pytest.fixture
def filter_narrow_filepaths():
    input = "polygons_different_compactness"
    output = "output"
    folder = Path("./tests/test_data/synthetic/vector_ops")
    input_file = folder/(input + ".geojson")
    output_file = folder/(output+".geojson")
    yield input, output, folder, input_file, output_file
    if output_file.exists():
        output_file.unlink()


def test_filter_narrow_objects_removes_nothing(filter_narrow_filepaths):
    input, output, folder, input_file, output_file = filter_narrow_filepaths

    pipeline = Compose(bricks = [FilterNarrowObjects(input=input,
                                                     output=output,
                                                     min_width=0.0,
                                                     min_isoperimetric_quotient=0.0)])
    # see that brick produces result
    assert not output_file.exists()
    pipeline(str(folder))
    assert output_file.exists()

    input_fc = FeatureCollection.from_file(input_file)
    output_fc = FeatureCollection.from_file(output_file)
    assert len(input_fc) == 4
    assert len(output_fc) == 4


def test_filter_narrow_objects_removes_low_width(filter_narrow_filepaths):
    input, output, folder, input_file, output_file = filter_narrow_filepaths
    # width less than 2 meters
    pipeline = Compose(bricks = [FilterNarrowObjects(input=input,
                                                     output=output,
                                                     min_width=2.0,
                                                     min_isoperimetric_quotient=0.0)])
    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    input_fc = FeatureCollection.from_file(input_file)
    output_fc = FeatureCollection.from_file(output_file)
    assert len(input_fc) == 4
    assert len(output_fc) == 3


def test_filter_narrow_objects_removes_low_quotient(filter_narrow_filepaths):
    input, output, folder, input_file, output_file = filter_narrow_filepaths
    # isoperimetric quotient is between rectangles with 1:2 and 1:3 ratio
    pipeline = Compose(bricks = [FilterNarrowObjects(input=input,
                                                     output=output,
                                                     min_width=0.0,
                                                     min_isoperimetric_quotient=0.6)])
    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    input_fc = FeatureCollection.from_file(input_file)
    output_fc = FeatureCollection.from_file(output_file)
    # todo: think about better way to check than the quantity?
    assert len(input_fc) == 4
    assert len(output_fc) == 3


def test_filter_narrow_removes_up_to_squares(filter_narrow_filepaths):
    input, output, folder, input_file, output_file = filter_narrow_filepaths
    # isoperimetric quotient higher than in square
    pipeline = Compose(bricks = [FilterNarrowObjects(input=input,
                                                     output=output,
                                                     min_width=0.0,
                                                     min_isoperimetric_quotient=0.8)])
    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    input_fc = FeatureCollection.from_file(input_file)
    output_fc = FeatureCollection.from_file(output_file)
    assert len(input_fc) == 4
    assert len(output_fc) == 1
    # only round-shape-like polygon must stay


def test_filter_narrow_removes_everything_by_width(filter_narrow_filepaths):
    input, output, folder, input_file, output_file = filter_narrow_filepaths
    # isoperimetric quotient higher than in square
    pipeline = Compose(bricks = [FilterNarrowObjects(input=input,
                                                     output=output,
                                                     min_width=20.0,
                                                     min_isoperimetric_quotient=0.0)])
    # see that brick produces result
    assert not output_file.exists()
    pipeline(folder)
    assert output_file.exists()

    input_fc = FeatureCollection.from_file(input_file)
    output_fc = FeatureCollection.from_file(output_file)
    assert len(input_fc) == 4
    assert len(output_fc) == 0
    # everything is removed


"""================= test RemoveOverlappingObjects ===================="""


@pytest.fixture
def remove_overlapping_filepaths():
    input = "input"
    output = "output"
    folder_1 = Path("./tests/test_data/synthetic/fully_overlapped_polygon")
    folder_2 = Path("./tests/test_data/synthetic/partially_overlapped_polygons")
    input_file_1 = folder_1/(input + ".geojson")
    input_file_2 = folder_2/(input + ".geojson")
    output_file_1 = folder_1/(output+".geojson")
    output_file_2 = folder_2/(output+".geojson")
    yield input, output, folder_1, folder_2, input_file_1, input_file_2, output_file_1, output_file_2
    if output_file_1.exists():
        output_file_1.unlink()
    if output_file_2.exists():
        output_file_2.unlink()


def test_remove_fully_overlapping_objects(remove_overlapping_filepaths):
    # fully overlapping objects removed but partly overlapping not removed
    input, output, folder_1, folder_2, input_file_1, \
        input_file_2, output_file_1, output_file_2 = remove_overlapping_filepaths
    pipeline = Compose(bricks = [RemoveOverlappingObjects(input=input,
                                                          output=output,
                                                          max_area=0.0,
                                                          area_fraction_threshold=1.0)])

    # see that brick produces result
    pipeline(str(folder_1))

    input_fc = FeatureCollection.from_file(input_file_1)
    output_fc = FeatureCollection.from_file(output_file_1)
    assert len(input_fc) == 2
    assert input_fc.iloc[0].geometry.contains(input_fc.iloc[1].geometry) or\
           input_fc.iloc[1].geometry.contains(input_fc.iloc[0].geometry)
    assert len(output_fc) == 1
    # fully overlapped object must be deleted

    pipeline(str(folder_2))

    input_fc = FeatureCollection.from_file(input_file_2)
    output_fc = FeatureCollection.from_file(output_file_2)
    assert len(input_fc) == 3
    assert len(output_fc) == 3
    # partially overlapped object must not be deleted


def test_remove_partially_overlapping_objects_05(remove_overlapping_filepaths):
    # fully overlapping objects removed but partly overlapping not removed
    input, output, folder_1, folder_2, input_file_1,\
        input_file_2, output_file_1, output_file_2 = remove_overlapping_filepaths
    pipeline = Compose(bricks = [RemoveOverlappingObjects(input=input,
                                                          output=output,
                                                          max_area=0.0,
                                                          area_fraction_threshold=0.5)])

    # see that brick produces result
    pipeline(folder_1)

    input_fc = FeatureCollection.from_file(input_file_1)
    output_fc = FeatureCollection.from_file(output_file_1)
    assert len(input_fc) == 2
    assert len(output_fc) == 1
    # fully overlapped object must be deleted

    pipeline(folder_2)

    input_fc = FeatureCollection.from_file(input_file_2)
    output_fc = FeatureCollection.from_file(output_file_2)
    assert len(input_fc) == 3
    assert len(output_fc) == 2
    # one of 2 partially overlapped objects must be deleted


def test_remove_partially_overlapping_objects_01(remove_overlapping_filepaths):
    # fully overlapping objects removed but partly overlapping not removed
    input, output, folder_1, folder_2, input_file_1,\
        input_file_2, output_file_1, output_file_2 = remove_overlapping_filepaths
    pipeline = Compose(bricks = [RemoveOverlappingObjects(input=input,
                                                          output=output,
                                                          max_area=0.0,
                                                          area_fraction_threshold=0.1)])
    pipeline(str(folder_1))

    input_fc = FeatureCollection.from_file(input_file_1)
    output_fc = FeatureCollection.from_file(output_file_1)
    assert len(input_fc) == 2
    assert len(output_fc) == 1
    # fully overlapped object must be deleted

    pipeline(str(folder_2))
    input_fc = FeatureCollection.from_file(input_file_2)
    output_fc = FeatureCollection.from_file(output_file_2)
    assert len(input_fc) == 3
    assert len(output_fc) == 1
    # partially overlapped objects must be deleted


def test_remove_partially_overlapping_objects_max_area(remove_overlapping_filepaths):
    # fully overlapping objects removed but partly overlapping not removed
    input, output, folder_1, folder_2, input_file_1,\
        input_file_2, output_file_1, output_file_2 = remove_overlapping_filepaths
    pipeline = Compose(bricks = [RemoveOverlappingObjects(input=input,
                                                          output=output,
                                                          max_area=20.0,
                                                          area_fraction_threshold=0.1)])
    pipeline(str(folder_2))
    input_fc = FeatureCollection.from_file(input_file_2)
    output_fc = FeatureCollection.from_file(output_file_2)
    assert len(input_fc) == 3
    assert len(output_fc) == 2
    # one of two partially overlapped object must not be deleted
