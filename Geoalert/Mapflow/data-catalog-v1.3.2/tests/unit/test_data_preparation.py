import os
from PIL import Image
from app.functional.data import generate_preview, get_file_description


def test_generate_preview(get_files):
    for (input_file, preview_file, _, preview_shape, _) in get_files:
        if not preview_shape or not preview_file:
            continue
        assert not os.path.exists(preview_file)
        generate_preview(input_file=input_file, preview_file=preview_file, size=1024)
        assert os.path.exists(preview_file)
        with Image.open(preview_file) as image:
            assert image.size == (preview_shape[1], preview_shape[0])
        # todo: add image with size smaller than 1024, with non-rgb channels, with 16-bit data


def test_get_file_description(get_files):
    for (input_file, _, input_profile, _, checksum) in get_files:
        if not checksum:
            continue
        expected = (
            os.path.split(input_file)[-1],
            checksum,
            {
                'dtypes': (input_profile['dtype'], input_profile['dtype'], input_profile['dtype']),
                           'width': input_profile['width'],
                           'height': input_profile['height'],
                           'nodata': input_profile['nodata'],
                           'count': input_profile['count'],
                           'crs': input_profile['crs'],
                           'pixel_size': (input_profile['transform'][0], - input_profile['transform'][4])
            }
        )
        result = get_file_description(input_file, os.path.split(input_file)[-1])
        assert result == expected

