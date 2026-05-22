import os

import rasterio
from shapely.geometry import shape
import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from loguru import logger

from cog_builder.app.message import CogBuilderArtifact
from cog_builder.app.storage import CogBuilderStorage
from cog_builder.config import CogBuilderConfig

minio_host = os.getenv("MINIO_HOST")
minio_port = os.getenv("MINIO_PORT")
minio_url = ':'.join((minio_host, minio_port)) if minio_port else minio_host
minio_secret_key = 
minio_access_key = 
BUCKET_NAME = os.getenv("BUCKET_NAME")


@pytest.fixture
def create_test_files():
    config = CogBuilderConfig()
    config.set_gdal_env()

    # First raster
    tempdir = TemporaryDirectory()
    workdir = Path(tempdir.name)
    filename1 = workdir /"part_1.tif"
    size = 10
    profile = {'width': size,
               'height': size,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(50, 0, 10000, 0, -50, 20000),
               'crs': 'EPSG:3857',
               'nodata': 0}
    raster = np.ones(dtype='uint8',
                     shape=(3, size, size))
    with rasterio.open(filename1, 'w', **profile) as dst:
        dst.write(raster)

    # Second raster - shifted 450 meters left and 450 meters up
    filename2 = workdir /"part_2.tif"
    size = 10
    profile = {'width': size,
               'height': size,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(50, 0, 9550, 0, -50, 20450),
               'crs': 'EPSG:3857',
               'nodata': 0}
    raster = np.ones(dtype='uint8',
                     shape=(3, size, size))*2
    with rasterio.open(filename2, 'w', **profile) as dst:
        dst.write(raster)

    # AOI to cover part of the files
    # Triangle 100*100 meters
    aoi = shape({"type": "Polygon", "coordinates": [[[9900, 20100],
                                                     [10100, 20100],
                                                     [10100, 19900],
                                                     [9900, 20100]]]})
    aoi_latlon = shape(rasterio.warp.transform_geom(src_crs="EPSG:3857",
                                                    dst_crs="EPSG:4326",
                                                    geom=aoi))
    yield workdir, filename1, filename2, aoi_latlon

    tempdir.cleanup()


def upload_create_bucket(storage, artifact, workdir, bucket_name):
    logger.debug(f"Storage: {storage.endpoint}, {storage.session.endpoint_url}")
    try:
        storage.s3_resource.create_bucket(Bucket=bucket_name)
    except Exception as e:
        logger.debug(f"Bucket not created: {e}")
        pass
    logger.debug(f"Uploading artifact {artifact.name} to {artifact.path}")
    storage.upload(artifact=artifact, workdir=workdir)
    logger.debug("Successfully uploaded")


def test_download_folder(create_test_files):
    folder_bucket_name = "folder-artifact"

    logger.debug("TESTING DOWNLOAD & CUT FROM FOLDER")
    workdir, filename1, filename2, aoi = create_test_files
    # first file
    s3_path = f's3://{folder_bucket_name}/file1.tif'
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename1.name)
    storage = CogBuilderStorage(minio_url=minio_url,
                               minio_secret_key=,
                               minio_access_key=
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=folder_bucket_name)
    # second file
    s3_path = f's3://{folder_bucket_name}/file2.tif'
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename2.name)
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=folder_bucket_name)

    # Upload non-tiff file: it should not be used in processing
    with open(workdir/'tmp.jpg', 'w') as dst:
        dst.write("Sample content")
    s3_path = f's3://{folder_bucket_name}/tmp.jpg'
    upload_artifact = CogBuilderArtifact(path=s3_path, name="tmp.jpg")
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=folder_bucket_name)
    # Upload file to SUBFOLDER. It should not be listed
    s3_path = f's3://{folder_bucket_name}/subfolder/file.tif'
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename1.name)
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=folder_bucket_name)


    download_artifact = CogBuilderArtifact(path=f's3://{folder_bucket_name}/', name="out5.tif")
    # check listed files. File in subfolder and jpg must not be listed
    files = storage.list_artifacts(download_artifact)
    assert len(files) == 2
    assert set(str(artifact.path) for artifact in files) == {f's3://{folder_bucket_name}/file1.tif',
                                                             f's3://{folder_bucket_name}/file2.tif'}

    # download
    storage.get_artifact(artifact=download_artifact, workdir=workdir, aoi=aoi)
    out_path = workdir / download_artifact.name
    assert out_path.exists
    assert out_path.is_file
    with rasterio.open(workdir/'out5.tif') as src:
        data = src.read(1)
        assert np.all(data == np.array([[2, 2, 2, 0],
                                 [2, 2, 2, 0],
                                 [2, 2, 2, 1],
                                 [0, 0, 1, 1],
                                 [0, 0, 1, 1]], dtype='uint8'))
        mask = src.read_masks(1)
        assert np.all(mask == np.array([[255, 255, 255, 0],
                                        [255, 255, 255, 0],
                                        [255, 255, 255, 255],
                                        [0, 0, 255, 255],
                                        [0, 0, 255, 255]], dtype='uint8'))

