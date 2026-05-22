import uuid
import os
from pathlib import Path
from typing import Union, List

from loguru import logger
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2.shape import from_shape

from functional.data import get_footprint, get_file_description, generate_preview, get_file_size
from functional.storage import minio_client
from crud.image import get_first_image_of_mosaic, update_images_table, get_image_by_id
from crud.workflow import add_workflow_with_image_id
from config import Config
from errors import (MemoryLimitExceeded,
                    FileTooBig,
                    PreviewNotFound)

from functional.validations import check_file
from functional.validations import validate_metadata, validate_footprint

PREVIEW_EXTENSION = ".jpg"
ALLOWED_FILE_EXTENSIONS = [".tif", ".tiff"]
MINIO_BUCKET = Config.MINIO_BUCKET
MAX_FILE_SIZE = Config.MAX_UPLOAD_FILE_SIZE
MAX_IMAGE_AREA_DEGREES = Config.MAX_IMAGE_AREA_DEGREES
MEMORY_LIMIT_SWITCH = Config.MEMORY_LIMIT
PREVIEW_SIZE_L = Config.PREVIEW_SIZE_L
PREVIEW_SIZE_S = Config.PREVIEW_SIZE_S


def handle_file_upload_to_mosaic(tmp_path: Union[str, Path], filename: str,
                                 s3_mosaic_path: str, mosaic_id: UUID,
                                 available_disk_space: int) -> None:

    """
    Function to handle uploaded files: uploads file to minio, generates preview, uploads preview to minio,
    gets footprint and metadata of file, updates images table of database.
    Args:
        tmp_path: temp directory where initial file should be, and where all the other files should be created
        filename: name of the initial file without folder
        s3_mosaic_path: where the mosaic should be in the minio
        mosaic_id: ID for database
        available_disk_space: available space on disk for user
    """
    image_id = uuid.uuid4()
    image_path = Path(tmp_path)/filename
    file_size = get_file_size(image_path)
    if file_size > MAX_FILE_SIZE:
        raise FileTooBig(actual_file_size=file_size)
    if (available_disk_space <= 0 or available_disk_space - file_size <= 0) and MEMORY_LIMIT_SWITCH:
        raise MemoryLimitExceeded(available_memory=available_disk_space if available_disk_space > 0 else 0,
                                  memory_requested=file_size)
    check_file(image_path)
    # prepare file metadata and preview
    filename, sha1_checksum, meta_data = get_file_description(file_path=image_path, filename=filename)

    # get base metadata for mosaic
    # for now, we will consider metadata of the first image as base for our mosaic
    try:
        mosaic_metadata = get_first_image_of_mosaic(mosaic_id=mosaic_id).meta_data
    except Exception:
        mosaic_metadata = None
    # skip mosaic validation if mosaic is empty
    if mosaic_metadata:
        validate_metadata(source_metadata=meta_data, target_metadata=mosaic_metadata,
                          mosaic_id=mosaic_id, filename=filename)

    # image_path = Path(tmp_path)/filename
    footprint = get_footprint(image_path)
    validate_footprint(footprint)
    footprint = from_shape(footprint)

    preview_file_l, preview_file_s = get_preview_paths(filename, tmp_path, [PREVIEW_SIZE_L, PREVIEW_SIZE_S])
    generate_preview(input_file=image_path, preview_file=preview_file_l, size=PREVIEW_SIZE_L)
    generate_preview(input_file=image_path, preview_file=preview_file_s, size=PREVIEW_SIZE_S)

    # upload preview and image to S3
    image_url, preview_url_l, preview_url_s = get_minio_paths(s3_mosaic_path=s3_mosaic_path,
                                                              preview_size_l=PREVIEW_SIZE_L,
                                                              preview_size_s=PREVIEW_SIZE_S,
                                                              image_id=image_id)
    minio_client.upload_object_by_url(url=image_url, local_path=image_path)
    minio_client.upload_object_by_url(url=preview_url_l, local_path=preview_file_l)
    minio_client.upload_object_by_url(url=preview_url_s, local_path=preview_file_s)

    update_images_table(image_id=image_id, mosaic_id=mosaic_id, image_url=image_url,
                        preview_url_l=preview_url_l, preview_url_s=preview_url_s,
                        footprint=footprint, filename=filename, sha1_checksum=sha1_checksum,
                        meta_data=meta_data, file_size=file_size)
    add_workflow_with_image_id(image_id=image_id, status='UNPROCESSED')


def generate_file_name(suffix: str) -> str:
    """
    Generate and return unique filename with given suffix
    """
    return f"{uuid.uuid4().hex}.{suffix}"


def delete_temp_files(*args: str):
    for arg in args:
        if os.path.isfile(arg):
            try:
                os.remove(arg)
            except Exception:
                logger.info(f"file: {arg} couldn't be deleted")


def get_preview_paths(filename: str,
                      folder: Union[str, Path],
                      preview_sizes: List[int]):
    file_path = Path(folder)/filename
    suffix = file_path.suffix

    if not suffix:
        logger.info(f'Adding tif suffix: workaround for mapflow web which does not send file names, got name {filename}')
        file_path = file_path.with_suffix('.tif')
    elif suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError(f"File must have one of allowed extensions: {ALLOWED_FILE_EXTENSIONS}, got {file_path.name = }")

    preview_paths = [file_path.with_name(f'{file_path.stem}_{preview_size}' + PREVIEW_EXTENSION) for preview_size in preview_sizes]
    return preview_paths


def get_minio_paths(s3_mosaic_path, preview_size_l, preview_size_s, image_id=None):
    image_url = f'{s3_mosaic_path}/{str(image_id)}.tif'
    preview_url_l = f'{s3_mosaic_path}/{str(image_id)}_{preview_size_l}' + PREVIEW_EXTENSION
    preview_url_s = f'{s3_mosaic_path}/{str(image_id)}_{preview_size_s}' + PREVIEW_EXTENSION
    return image_url, preview_url_l, preview_url_s


def generate_mosaic_path(username: str, owner_id: uuid, mosaic_id: uuid):
    return f"s3://{MINIO_BUCKET}/{username}_{owner_id}/{mosaic_id}"


def get_preview_url_by_type(image_id: UUID, preview_type: str) -> str:
    # get image record by id
    image = get_image_by_id(image_id=image_id)
    if preview_type == "l":
        return image.preview_url_l
    elif preview_type == "s":
        return image.preview_url_s
    else:
        raise PreviewNotFound(image_id=image_id)


