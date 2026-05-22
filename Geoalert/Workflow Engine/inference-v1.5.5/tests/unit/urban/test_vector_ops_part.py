import shapely
import os
import shutil
import pytest
from pathlib import Path
from urban import Compose, NMS, FilterByProperty
from urban.functional import io
from gpdadapter import FeatureCollection


@pytest.fixture(scope='module')
def get_empty_fc():
    folder = './tests/test_data/tmp1/'
    input_name = 'empty_fc_in'

    os.makedirs(folder, exist_ok=True)
    fc = FeatureCollection()
    io.save_fc(fc, path=folder, name=input_name)
    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_single_fc():
    folder = './tests/test_data/tmp1/'
    input_name = 'single_fc_in'

    os.makedirs(folder, exist_ok=True)
    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])],
                            'confidence': [1]})
    io.save_fc(fc, path=folder, name=input_name)
    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_two_fc():
    folder = './tests/test_data/tmp1/'
    input_name = 'two_fc_in'

    os.makedirs(folder, exist_ok=True)
    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]]),
                                         shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])],
                            'confidence': [1, 2]})
    io.save_fc(fc, path=folder, name=input_name)
    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_no_confidence_fc():
    folder = './tests/test_data/tmp1/'
    input_name = 'no_confidence_fc_in'

    os.makedirs(folder, exist_ok=True)
    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]]),
                                         shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])]})
    io.save_fc(fc, path=folder, name=input_name)
    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_iou_thr_fc():
    folder = './tests/test_data/tmp1/'
    input_name = 'iou_thr_fc_in'

    os.makedirs(folder, exist_ok=True)
    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 2], [1, 2], [1, 0]]),
                                         shapely.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])],
                            'confidence': [1, 2]})
    io.save_fc(fc, path=folder, name=input_name)
    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


"""================= test NMS ===================="""


def test_nms_on_empty_fc(get_empty_fc):
    folder, input_name = get_empty_fc
    folder = Path(folder)
    output_name = 'empty_fc_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert fc.empty


def test_nms_on_a_single_feature_fc(get_single_fc):
    folder, input_name = get_single_fc
    folder = Path(folder)
    output_name = 'single_fc_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert len(fc) == 1


def test_nms_on_two_features(get_two_fc):
    folder, input_name = get_two_fc
    folder = Path(folder)
    output_name = 'two_fc_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert (len(fc) == 1 and fc[0, 'confidence'] == 2)


def test_nms_on_two_features_none(get_two_fc):
    folder, input_name = get_two_fc
    folder = Path(folder)
    output_name = 'two_fc_none_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name, confidence_tag=None)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert len(fc) == 1


def test_nms_without_confidence(get_no_confidence_fc):
    folder, input_name = get_no_confidence_fc
    folder = Path(folder)
    output_name = 'no_confidence_fc_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert len(fc) == 2


def test_nms_with_iou_threshold(get_iou_thr_fc):
    folder, input_name = get_iou_thr_fc
    folder = Path(folder)
    output_name = 'iou_thr_fc_out'

    pipeline = Compose(bricks=[NMS(input=input_name, output=output_name, iou_threshold=0.75)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    # iou_thr here is greater than actual IoU, so nms should not work
    assert (len(fc) == 2)


"""================= test FilterByProperty ===================="""


def test_filter_by_property_on_empty_fc(get_empty_fc):
    folder, input_name = get_empty_fc
    folder = Path(folder)
    output_name = 'empty_fc_out_filter'

    pipeline = Compose(bricks=[FilterByProperty(input=input_name, property_tag='confidence', output=output_name)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert fc.empty


def test_filter_by_property_no_property(get_single_fc):
    folder, input_name = get_single_fc
    folder = Path(folder)
    output_name = 'single_fc_out_filter_no_property'

    pipeline = Compose(bricks=[FilterByProperty(input=input_name, property_tag='class_id', output=output_name,
                                                return_no_property='empty')])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert fc.empty


def test_filter_by_property_no_value(get_single_fc):
    folder, input_name = get_single_fc
    output_name = 'single_fc_out_filter_no_value'

    with pytest.raises(ValueError, match='FilterByProperty `value` must be set if `predicate` is not "not_none"'):
        pipeline = Compose(bricks=[FilterByProperty(input=input_name, property_tag='confidence',
                                                    predicate='more_equal_than', output=output_name)])


def test_filter_by_property(get_single_fc):
    folder, input_name = get_single_fc
    folder = Path(folder)
    output_name = 'single_fc_out_filter'

    pipeline = Compose(bricks=[FilterByProperty(input=input_name, property_tag='confidence', output=output_name,
                                                value=1)])
    pipeline(folder)

    fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))
    assert len(fc) == 1