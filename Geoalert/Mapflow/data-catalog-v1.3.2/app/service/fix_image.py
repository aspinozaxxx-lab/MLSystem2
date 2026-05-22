from uuid import UUID
import tempfile
from pathlib import Path

from errors import ItemNotFound
from functional.data import generate_preview
from functional.storage import minio_client
from crud import get_image_by_id, update_preview_url_record, get_mosaic_by_id
from functional.file_handling import get_preview_paths, get_minio_paths
from functional.urlparse import parse_image_url
from config import Config


def generate_missing_preview(image_id: UUID, preview_type: str) -> str:
    """
    Generates missing preview, and returns its url at minio.
    """
    if preview_type == 's':
        prev_size = Config.PREVIEW_SIZE_S
    else:
        prev_size = Config.PREVIEW_SIZE_L
    image = get_image_by_id(image_id=image_id)
    if not image:
        raise ItemNotFound(uid=image_id)
    filename, minio_bucket_from_url, minio_object_from_url = parse_image_url(image.image_url)

    s3_mosaic_path = get_mosaic_by_id(uid=image.mosaic_id).mosaic_url

    with tempfile.TemporaryDirectory() as tmp_dir:
        # download image from minio to temporary dir
        image_path = Path(tmp_dir) / filename
        minio_client.download_object_by_url(url=image.image_url, file_path=str(image_path))

        # generate preview of proper size in temporary dir
        preview_file_path = get_preview_paths(filename, tmp_dir, [prev_size])[0]
        generate_preview(input_file=image_path, preview_file=preview_file_path, size=prev_size)

        # generate preview url
        _, _, preview_url = get_minio_paths(s3_mosaic_path=s3_mosaic_path,
                                            preview_size_l=prev_size,
                                            preview_size_s=prev_size)

        # upload preview to minio
        minio_client.upload_object_by_url(url=preview_url, local_path=preview_file_path)

        # update images table
        update_preview_url_record(image_id=image_id, preview_type=preview_type, preview_url=preview_url)
    return preview_url
