from uuid import UUID
from fastapi.responses import StreamingResponse

from errors import AccessDenied, ItemNotFound
from crud import (user_memory_info, get_user_id, check_image_exists, check_user_has_access_to_image,
                  add_coglink_to_images_table, get_image_by_id, get_mosaic_by_id)
from functional.file_handling import get_preview_url_by_type
from functional.urlparse import parse_image_url
from functional.storage import s3
from .fix_image import generate_missing_preview


def get_memory_info_of_user_service(login: str):
    memory_limit, memory_used, memory_free = user_memory_info(login=login)
    return {
        "memoryLimit": memory_limit,
        "memoryUsed": memory_used,
        "memoryFree": memory_free
    }


def get_image_preview_service(login: str, image_id: UUID, preview_type: str) -> StreamingResponse:
    if not check_image_exists(image_id=image_id):
        raise ItemNotFound(uid=image_id, instance_type='image')
    user_id = get_user_id(login=login)
    if not check_user_has_access_to_image(user_id=user_id, image_id=image_id):
        raise AccessDenied(uid=image_id, user=login, instance_type="image")
    preview_url = get_preview_url_by_type(image_id=image_id, preview_type=preview_type)
    if not preview_url:
        # generate preview and update images table
        preview_url = generate_missing_preview(image_id=image_id, preview_type=preview_type)
    _, bucket_name, object_name = parse_image_url(image_url=preview_url)
    result = s3.client.get_object(Bucket=bucket_name, Key=object_name)
    return StreamingResponse(content=result["Body"].iter_chunks(), media_type='image/jpeg')


def add_coglink_to_images_table_service(image_id: UUID, aoi_id: int) -> None:
    image_coglink = generate_image_coglink(image_id=image_id, aoi_id=aoi_id)
    add_coglink_to_images_table(image_id=image_id, coglink=image_coglink)


def generate_image_coglink(image_id: UUID, aoi_id: int) -> str:
    mosaic_id_of_image = get_image_by_id(image_id=image_id).mosaic_id
    mosaic_of_image = get_mosaic_by_id(uid=mosaic_id_of_image)
    mosaic_coglink = mosaic_of_image.cog_link
    if mosaic_coglink is None:
        mosaic_coglink = mosaic_of_image.mosaic_url + "/cog"
    return mosaic_coglink + "/area-" + str(aoi_id) + ".tif"
