import os

import rasterio
from shapely.geometry import shape
import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from loguru import logger

from cog_builder.app.message import CogBuilderArtifact, CogBuilderInputMessage, CogBuilderOutputMessage
from cog_builder.app.storage import CogBuilderStorage
from cog_builder.app.message_handler import CogBuilderMessageHandlder
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

    aoi = {"type": "Polygon", "coordinates": [[[100000, 20000],
                                               [100250, 19800],
                                               [100250, 19750.5],
                                               [100000, 19950],
                                               [100000, 20000]]]}
    aoi_latlon = rasterio.warp.transform_geom(src_crs="EPSG:3857",
                                dst_crs="EPSG:4326",
                                geom=aoi)
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


def test_handle_message(create_test_files):
    logger.debug("TESTING E2E HANDLE MESSAGE: NOT DEFAULT VALUES")

    workdir, filename, aoi = create_test_files
    s3_path = 's3://healthcheck/healthcheck.tif'
    out_path = "s3://healthcheck/output.tif"
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename)
    storage = CogBuilderStorage(minio_url=minio_url,
                               minio_secret_key=,
                               minio_access_key=
    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=BUCKET_NAME)
    config = CogBuilderConfig()

    message_handler = CogBuilderMessageHandlder(config = config)
    input = {"task_id": 42,
             "runcheck_url": "https://runcheck",
             "input": {"raster_source": s3_path,
                       "compress": "JPEG",
                       "channels": "1,2,3",
                       "aoi": aoi},
             "output": {"target_uri": out_path}}
    input_message = CogBuilderInputMessage(**input)

    expected_output_message= CogBuilderOutputMessage(task_id=42,
                                                     status=0,
                                                     messages=[])

    output_message = message_handler.handle_message(input_message)
    assert output_message == expected_output_message

    with storage.open(out_path) as src:
        profile = src.profile
        mask = src.read_masks(1)
    assert (profile['count'], profile['height'], profile['width']) == (3, 512, 512)
    assert profile['compress'] == 'jpeg'
    # actual pixel number without AOI would be about 175000; so we want to make sure that mask is applied
    assert np.abs(np.count_nonzero(mask) - 34880) < 100


def test_handle_message_default(create_test_files):
    logger.debug("TESTING E2E HANDLE MESSAGE: DEFAULT VALUES")

    workdir, filename, _ = create_test_files
    s3_path = 's3://healthcheck/healthcheck.tif'
    out_path = "s3://healthcheck/output2.tif"
    upload_artifact = CogBuilderArtifact(path=s3_path, name=filename)
    storage = CogBuilderStorage(minio_url=minio_url,
                               minio_secret_key=,
                               minio_access_key=

    upload_create_bucket(storage=storage, artifact=upload_artifact, workdir=workdir, bucket_name=BUCKET_NAME)
    config = CogBuilderConfig()
    message_handler = CogBuilderMessageHandlder(config = config)
    input = {"task_id": 42,
             "runcheck_url": "https://runcheck",
             "input": {"raster_source": s3_path},
             "output": {"target_uri": out_path}}
    input_message = CogBuilderInputMessage(**input)

    expected_output_message= CogBuilderOutputMessage(task_id=42,
                                                     status=0,
                                                     messages=[])

    output_message = message_handler.handle_message(input_message)
    assert output_message == expected_output_message

    with storage.open(out_path) as src:
        profile = src.profile
        mask = src.read_masks(1)
    # With webp compression, the alpha channel is added
    # todo: make dstAlpha=False and use internal mask?
    assert (profile['count'], profile['height'], profile['width']) == (4, 1024, 1024)
    assert profile['compress'] == 'webp'
    # there were 1000*1000 pixels, but the image was rescaled to 1024*1024 with resolution change from 0.5 to 0.59....
    # Actual values are about 7e5, but we allow for some error
    assert np.abs(np.count_nonzero(mask) - 1000*1000*((0.5/0.5971642834774684)**2)) < 1000
