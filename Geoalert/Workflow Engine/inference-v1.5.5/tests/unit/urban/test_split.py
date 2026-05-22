import pytest
import os
import shutil
import rasterio
import numpy as np
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window

from urban.functional.raster_ops.split import generate_windows, split
SIZE_MULTIPLIER = 1


def generate_data():
    """
    We generate image with 10 channels where ea
    """
    shape = (11, 10*SIZE_MULTIPLIER, 10*SIZE_MULTIPLIER)
    stripe_shape = (1*SIZE_MULTIPLIER, 10*SIZE_MULTIPLIER)
    max_stripe = np.ones(shape=stripe_shape, dtype='uint8')*255
    ones_stripe = np.ones(shape=stripe_shape, dtype='uint8')

    image = np.zeros(shape=shape, dtype='uint8')

    for ch in range(10):
        stack = [ones_stripe]*10
        stack[ch] = max_stripe
        image[ch] = np.concatenate(stack, axis=0)

    # add horizontal split as alpha
    max_stripe = np.ones(shape=(10*SIZE_MULTIPLIER, 5*SIZE_MULTIPLIER), dtype='uint8')*255
    ones_stripe = np.zeros(shape=(10*SIZE_MULTIPLIER, 5*SIZE_MULTIPLIER), dtype='uint8')

    image[10] = np.concatenate([max_stripe, ones_stripe], axis=1)
    return image


def save_image(name, profile, data):

    colorinterp = profile.pop('colorinterp', None)
    mask = profile.pop('mask', None)
    # mask can be:
    # - none - then there is no additional mask info;
    # - np array
    # - if there is colorinterp with alpha in the profile, the mask will be written as alpha\
    # if not, it will be a separate bitmask
    data = np.array(data)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(name, 'w', **profile) as dst:
            if mask is not None:
                if colorinterp and ColorInterp.alpha == colorinterp[-1]:
                    # if mask is alpha channel - let's add it to the alpha
                    data[-1] = mask
                elif 'nodata' in profile.keys():
                    data = np.where(mask > 0, data, profile['nodata'])
                else:
                    # if mask is given, but there is no alpha channel - write is as a mask
                    dst.write_mask(mask != 0)
            if colorinterp:
                assert data.shape[0] == len(colorinterp)
                dst.colorinterp = colorinterp
            dst.write(data.astype(profile['dtype']))


def _full_path(tmpdir, name):
    return os.path.join(tmpdir, name + '.tif')


@pytest.fixture(scope='module')
def generate_input_images():
    TMPDIR = './tests/test_data/tmp'

    data = generate_data()

    base = {'driver': 'GTiff',
            'crs': 'EPSG:3857',
            'transform': Affine(0.5, 0, 10000, 0, -0.5, 10000),
            'width': data.shape[2], 'height': data.shape[1]}

    rgb = {'count': 3}  #
    pan = {'count': 1}
    pana = {'count': 2}
    rgba = {'count': 4}  #
    multi = {'count': 10}

    color_pan_alpha = {'colorinterp': [ColorInterp.gray, ColorInterp.alpha]}
    color_rgb = {'colorinterp': [ColorInterp.red, ColorInterp.green, ColorInterp.blue]}
    color_alpha = {'colorinterp': [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]}

    webp = {'compress': 'WEBP'}
    jpeg = {'compress': 'JPEG', 'photometric': 'YCbCr'}

    byte = {'dtype': 'uint8'}
    d_16bit = {'dtype': 'uint16'}
    d_float = {'dtype': 'float32'}

    nodata = {'nodata': 0}
    mask = {'mask': data[-1]}

    profiles = {
        # panchrome 8 bit
        # without mask
        'pan': {**base, **byte, **pan},
        # with mask
        'pan_nodata': {**base, **byte, **pan, **nodata, **mask},
        'pan_alpha': {**base, **byte, **pana, **color_pan_alpha, **mask},
        'pan_bitmask': {**base, **byte, **pan, **mask},
        # rgb 8 bit
        # without mask
        'rgb': {**base, **byte, **rgb},
        'rgb_color': {**base, **byte, **rgb, **color_rgb},
        # with mask
        'rgb_nodata': {**base, **byte, **rgb, **nodata, **mask},
        'rgb_bitmask': {**base, **byte, **rgb, **mask},
        'rgba': {**base, **byte, **rgba, **color_alpha, **mask},
        # Lossy compression!
        'rgb_jpeg_lossy': {**base, **byte, **rgb, **jpeg},
        'rgb_webp_lossy': {**base, **byte, **rgb, **webp, **mask},
        # multispectral 16 bit
        # without mask
        'multi': {**base, **d_16bit, **multi},
        # with mask (no alpha channel variant here)
        'multi_nodata': {**base, **d_16bit, **multi, **nodata, **mask},
        'multi_bitmask': {**base, **d_16bit, **multi, **mask},
    }

    os.makedirs(TMPDIR, exist_ok=True)
    for name, profile in profiles.items():
        save_image(_full_path(TMPDIR, name), profile, data[:profile['count'], :, :])
    yield data, TMPDIR, list(profiles.keys())

    # remove the files
    # shutil.rmtree(TMPDIR, ignore_errors=True)


@pytest.fixture()
def generate_big_file():
    tmpdir = './tests/test_data/tmp'
    profile = {'driver': 'GTiff',
               'crs': 'EPSG:3857',
               'transform': Affine(0.5, 0, 10000, 0, -0.5, 10000),
               'width': 50000, 'height': 50000,
               'count': 3, 'compress': 'ZSTD', 'predictor': 2, 'ZSTD_LEVEL': 2,
               'dtype': 'uint8'}
    name = tmpdir + '/big.tif'
    save_image(name,
               profile,
               np.ones(dtype='uint8', shape=(3, 50000, 50000)))
    yield tmpdir, name
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_generate_windows():
    windows = generate_windows(dataset_height=1000, dataset_width=1000, window_width=1500, window_height=1500)
    assert set(windows) == {Window(0, 0, 1000, 1000)}

    windows = generate_windows(dataset_height=2000, dataset_width=3000, window_width=1500, window_height=1500)
    assert set(windows) == {Window(0, 0, 1500, 1500), Window(1500, 0, 1500, 1500), Window(0, 1500, 1500, 500),
                            Window(1500, 1500, 1500, 500)}


def test_split_extracts_rgb(generate_input_images):
    _, tmp_dir, files = generate_input_images
    rgb = ['RED', 'GRN', 'BLU']
    # must be able to extract RGB from any of the files, and keep the raster exactly the same as in the original
    for file in files:
        path = _full_path(tmp_dir, file)
        split(path, tmp_dir, rgb)
        with rasterio.open(path) as original:
            orig_data = np.where(original.read_masks(), original.read(), 0)
        for ch, ch_name in enumerate(rgb):
            with rasterio.open(_full_path(tmp_dir, ch_name)) as src:
                if "pan" not in file:
                    assert np.all(src.read(1) == orig_data[ch])
                else:
                    assert np.all(src.read(1) == orig_data[0])


def test_split_multi(generate_input_images):
    _, tmp_dir, files = generate_input_images
    multi = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10']
    for file in files:
        path = _full_path(tmp_dir, file)
        if 'multi' in file:
            split(path, tmp_dir, multi)


def test_split_fails_if_too_many_channels(generate_input_images):
    _, tmp_dir, files = generate_input_images
    rgb = ['RED', 'GRN', 'BLU']
    multi = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10']
    for file in files:
        path = _full_path(tmp_dir, file)
        if 'rgb' in file:
            with pytest.raises(ValueError):
                split(path, tmp_dir, multi)
        if 'pan' in file:
            with pytest.raises(ValueError):
                split(path, tmp_dir, rgb, allow_singleband=False)


def test_nodata_is_preserved(generate_input_images):
    data, tmp_dir, files = generate_input_images
    rgb = ['RED', 'GRN', 'BLU']

    for file in files:
        path = _full_path(tmp_dir, file)
        if 'nodata' in file and 'pan' not in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[ch], 0))

        elif 'nodata' in file and 'pan' in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[0], 0))


def test_bitmask_to_nodata(generate_input_images):
    data, tmp_dir, files = generate_input_images
    rgb = ['RED', 'GRN', 'BLU']

    for file in files:
        path = _full_path(tmp_dir, file)
        if 'bitmask' in file and 'pan' not in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[ch], 0))

        elif 'bitmask' in file and 'pan' in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[0], 0))


def test_alpha_to_nodata(generate_input_images):
    data, tmp_dir, files = generate_input_images
    rgb = ['RED', 'GRN', 'BLU']

    for file in files:
        path = _full_path(tmp_dir, file)
        if 'alpha' in file and 'pan' not in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[ch], 0))

        elif 'alpha' in file and 'pan' in file:
            split(path, tmp_dir, rgb)
            for ch, name in enumerate(rgb):
                with rasterio.open(_full_path(tmp_dir, name)) as src:
                    assert src.nodata == 0
                    assert np.all(src.read(1) == np.where(data[-1] > 0, data[0], 0))


@pytest.mark.skip
def test_big_file(generate_big_file):
    tmpdir, file = generate_big_file
    rgb = ['RED', 'GRN', 'BLU']
    split(file, tmpdir, rgb)
