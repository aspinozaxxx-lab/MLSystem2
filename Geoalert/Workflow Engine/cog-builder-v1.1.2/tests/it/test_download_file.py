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

    aoi = shape({"type": "Polygon", "coordinates": [[[100000, 20000],
                                                     [100250, 19800],
                                                     [100250, 19750.5],
                                                     [100000, 19950],
                                                     [100000, 20000]]]})
    aoi_latlon = shape(rasterio.warp.transform_geom(src_crs="EPSG:3857",
                                                    dst_crs="EPSG:4326",
                                                    geom=aoi))
    tempdir = TemporaryDirectory()
    workdir = Path(tempdir.name)
    filename = "input.tif"
    size = 1000
    profile = {'width': size,
               'height': size,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(0.5, 0, 100000, 0, -0.5, 20000),
               'crs': 'EPSG:3857',
               'nodata': 0}

    raster = np.ones(dtype='uint8',
                     shape=(3, size, size))
    with rasterio.open(workdir / filename, 'w', **profile) as dst:
        dst.write(raster)

    yield workdir, filename, aoi_latlon

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


def test_download_whole_nocut(create_test_files):
    logger.debug("TESTING DOWNLOAD WHOLE FILE")

    workdir, filename, aoi = create_test_files
    s3_path = 's3://healthcheck/healthcheck.tif'
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename)
    storage = CogBuilderStorage(minio_url=minio_url,
                               minio_secret_key=,
                               minio_access_key=
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=BUCKET_NAME)
    download_artifact = CogBuilderArtifact(path=s3_path, name='out.tif')
    storage.get_artifact(artifact=download_artifact, workdir=workdir)

    out_path = workdir / download_artifact.name
    assert out_path.exists
    assert out_path.is_file
    assert open(out_path, 'rb').read() == open(workdir / filename, 'rb').read()
    profile = rasterio.open(out_path).profile
    assert (profile['count'], profile['height'], profile['width']) == (3, 1000, 1000)


def test_download_partial(create_test_files):
    logger.debug("TESTING DOWNLOAD & CUT")
    workdir, filename, aoi = create_test_files
    s3_path = 's3://healthcheck/healthcheck.tif'
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename)
    storage = CogBuilderStorage(minio_url=minio_url,
                               minio_secret_key=,
                               minio_access_key=

    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=BUCKET_NAME)

    download_artifact = CogBuilderArtifact(path=s3_path, name="out5.tif")
    storage.get_raster_artifact(artifact=download_artifact, workdir=workdir, aoi=aoi)

    out_path = workdir / download_artifact.name

    assert out_path.exists
    assert out_path.is_file

    profile = rasterio.open(out_path).profile
    assert (profile['count'], profile['height'], profile['width']) == (3, 500, 500)

