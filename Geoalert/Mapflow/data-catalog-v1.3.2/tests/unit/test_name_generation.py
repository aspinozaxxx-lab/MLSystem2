from app.functional.file_handling import get_minio_paths, get_preview_paths


def test_get_minio_paths():
    file, preview_l, preview_s = get_minio_paths('s3://users-data/test-user/mosaic1', 1024,256,
                                                 '3a2ce002-9818-4bcc-96e7-c44730501145')
    assert file == 's3://users-data/test-user/mosaic1/3a2ce002-9818-4bcc-96e7-c44730501145.tif'
    assert preview_l == 's3://users-data/test-user/mosaic1/3a2ce002-9818-4bcc-96e7-c44730501145_1024.jpg'
    assert preview_s == 's3://users-data/test-user/mosaic1/3a2ce002-9818-4bcc-96e7-c44730501145_256.jpg'


def test_get_preview_paths():
    preview_path_l, preview_path_s = get_preview_paths('raster.tif', '/tmp/data/012345', [1024, 64])
    assert str(preview_path_l) == '/tmp/data/012345/raster_1024.jpg'
    assert str(preview_path_s) == '/tmp/data/012345/raster_64.jpg'