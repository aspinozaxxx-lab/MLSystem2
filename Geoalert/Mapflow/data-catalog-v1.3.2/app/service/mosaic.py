import uuid
import tempfile
from typing import Union, List
from pathlib import Path
from fastapi import UploadFile
from loguru import logger
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID

from model.mosaic import Mosaic
from functional.storage import minio_client
from functional.urlparse import parse_image_url
from schemas.mosaic import MosaicUpdateRequestSchema
from crud.mosaic import (check_mosaic_owner,
                         update_mosaic_record,
                         create_mosaics_record,
                         get_mosaic_by_id,
                         check_mosaic_exists,
                         delete_mosaic_record_by_mosaic_id,
                         get_all_mosaics_of_user,
                         get_mosaics_of_user_by_tag)
from crud.image import (get_images_by_mosaic_id,
                        delete_image_records_by_uid,
                        get_mosaic_id_of_image,
                        get_image_by_id,
                        check_if_file_exists_by_image_url)
from crud.user import get_user_id, get_free_disk_space_of_user
from crud.misc import (create_usermosaic_record,
                       check_user_has_access_to_mosaic,
                       delete_usermosaic_record_of_user_by_mosaic_id)
from crud.workflow import delete_workflow_record_by_image_id
from errors import (MemoryLimitExceeded, FileTooBig, AccessDenied, FileCheckFailed, InternalError, ItemNotFound,
                    InvalidLinkToMinio, MinioObjectDoesntExist, FileAlreadyExists)

from functional.file_handling import handle_file_upload_to_mosaic, generate_file_name, generate_mosaic_path
from functional.data import save_upload_file_tmp, get_file_size
from config import Config

MEMORY_LIMIT_SWITCH = Config.MEMORY_LIMIT
RASTER_TILE_SERVER_URL = Config.RASTER_TILE_SERVER_URL


def mosaic_create_service(name: str, tags: Union[List[str], None], username: str):
    try:
        mosaic_id = uuid.uuid4()
        mosaic_owner = get_user_id(username)
        s3_mosaic_path = generate_mosaic_path(username=username, owner_id=mosaic_owner, mosaic_id=mosaic_id)
        cog_link = s3_mosaic_path + "/cog"
        mosaic = Mosaic(id=mosaic_id,
                        owner_id=mosaic_owner,
                        tags=tags,
                        mosaic_url=s3_mosaic_path,
                        name=name,
                        created_at=datetime.now(),
                        cog_link=cog_link)

        create_mosaics_record(mosaic)

        res = get_mosaic_by_id(mosaic_id)

        # also create a record on usermosaic table, which is the main table to look at in order to check if user has
        # access to a certain mosaic or not
        create_usermosaic_record(user_id=mosaic_owner, mosaic_id=mosaic_id)

    except Exception as e:
        logger.exception(f"Error on creating mosaic item in db: {str(e)}")
        raise e
    return res


def mosaic_update_service(username: str, mosaic_id: UUID, mosaic: MosaicUpdateRequestSchema):
    mosaic_exists = check_mosaic_exists(mosaic_id)
    if not mosaic_exists:
        raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
    user_id = get_user_id(username)
    # check if user is mosaic owner
    user_owns_mosaic = check_mosaic_owner(mosaic_id, user_id)
    # if user has an access, update mosaic
    if user_owns_mosaic:
        mosaic = Mosaic(
                id=mosaic_id,
                tags=mosaic.tags,
                name=mosaic.name
        )
        update_mosaic_record(mosaic)
        mosaic = get_mosaic_by_id(mosaic_id)

        return mosaic
    else:
        raise AccessDenied(uid=mosaic_id, user=username, instance_type="mosaic")


def mosaic_retrieve_service_by_id(username: str, mosaic_id: UUID):
    mosaic_exists = check_mosaic_exists(mosaic_id)
    if not mosaic_exists:
        raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
    # check if user has access to mosaic being retrieved
    user_id = get_user_id(username)
    user_has_access = check_user_has_access_to_mosaic(user_id=user_id, mosaic_id=mosaic_id)

    if user_has_access:
        # get mosaic and return results
        mosaic = get_mosaic_by_id(mosaic_id)
        tileUrl, tileJsonUrl = generate_xyz_link_for_mosaic(cog_link=mosaic.cog_link)
        rasterLayer = {
            'tileUrl': tileUrl,
            'tileJsonUrl': tileJsonUrl
        }
        return {"id": mosaic.id,
                "rasterLayer": rasterLayer,
                "tags": mosaic.tags,
                "name": mosaic.name,
                "created_at": mosaic.created_at}
    else:
        raise AccessDenied(uid=mosaic_id, user=username,  instance_type="mosaic")


def mosaic_delete_service(mosaic_id: UUID, username: str):
    try:
        mosaic_exists = check_mosaic_exists(mosaic_id)
        if not mosaic_exists:
            raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
        user_id = get_user_id(username)

        user_owns_mosaic = check_mosaic_owner(mosaic_id, user_id)
        if user_owns_mosaic:
            # get list of images of that mosaic
            images = get_images_by_mosaic_id(mosaic_id)

            for image in images:
                # delete images, previews and cogs from minio
                if image.cog_link is None:
                    minio_client.remove_objects_by_url([image.image_url, image.preview_url_l, image.preview_url_s])
                else:
                    minio_client.remove_objects_by_url([image.image_url, image.preview_url_l,
                                                        image.preview_url_s, image.cog_link])
                # delete workflow records from db
                delete_workflow_record_by_image_id(image_id=image.id)

            # delete database record: images that belong to mosaic
            uids_to_delete = [image.id for image in images]
            delete_image_records_by_uid(uids_to_delete)

            # delete usermosaic record
            delete_usermosaic_record_of_user_by_mosaic_id(user_id=user_id, mosaic_id=mosaic_id)

            # delete mosaic record
            delete_mosaic_record_by_mosaic_id(mosaic_id)

            return {"message": "Files deleted"}
        else:
            raise AccessDenied(uid=mosaic_id,  user=username, instance_type="mosaic")
    except Exception as e:
        raise e


def mosaic_retrieve_service(username: str, tags: list):
    """
    Retrieves all mosaics of a given user
    :param username: user login.
    :param tags: mosaic tags. List[str].
    :return: List[Mosaic]
    """
    user_id = get_user_id(username)
    if tags is None:
        mosaics = get_all_mosaics_of_user(user_id)
    else:
        mosaics = get_mosaics_of_user_by_tag(user_id, tags)
    response = []
    for mosaic in mosaics:
        tileUrl, tileJsonUrl = generate_xyz_link_for_mosaic(cog_link=mosaic.cog_link)
        rasterLayer = {
            'tileUrl': tileUrl,
            'tileJsonUrl': tileJsonUrl
        }
        response.append(
            {
                "id": mosaic.id,
                "rasterLayer": rasterLayer,
                "tags": mosaic.tags,
                "name": mosaic.name,
                "created_at": mosaic.created_at
            }
        )
    return response


def delete_image_by_id_service(username: str, image_id: UUID):
    # get mosaic_id using image_id
    mosaic_id = get_mosaic_id_of_image(image_id)
    if not mosaic_id:
        raise ItemNotFound(uid=image_id, instance_type='mosaic')
    user_id = get_user_id(username)
    # check if user is mosaic owner
    user_owns_mosaic = check_mosaic_owner(mosaic_id, user_id)
    if user_owns_mosaic:
        # delete image, preview_image and cogs from minio
        image = get_image_by_id(image_id)
        if not image.cog_link:
            minio_client.remove_objects_by_url([image.image_url, image.preview_url_l, image.preview_url_s])
        else:
            minio_client.remove_objects_by_url([image.image_url, image.preview_url_l,
                                                image.preview_url_s, image.cog_link])
        # delete image record from DB
        delete_image_records_by_uid([image_id])
        # delete workflow record from DB
        delete_workflow_record_by_image_id(image_id=image.id)
        return {"message": f"{image_id} deleted"}
    else:
        raise AccessDenied(uid=image_id, user=username, instance_type="image")


def create_mosaic_and_file_upload_service(username: str, files: List[UploadFile], tags: List[str], name: str) -> dict:
    # generate mosaic_id
    mosaic_id = uuid.uuid4()
    # get user id, who uploaded mosaic / owns mosaic
    mosaic_owner_id = get_user_id(username)
    # s3 mosaic path includes user_id (owner_id) and mosaic_id
    s3_mosaic_path = generate_mosaic_path(username=username, owner_id=mosaic_owner_id, mosaic_id=mosaic_id)
    if tags:
        mosaic_tags = tags
    else:
        mosaic_tags = []
    cog_link = s3_mosaic_path + '/cog'
    mosaic = Mosaic(id=mosaic_id, owner_id=mosaic_owner_id, mosaic_url=s3_mosaic_path,
                    tags=mosaic_tags, name=name, created_at=datetime.now(), cog_link=cog_link)

    try:
        create_mosaics_record(mosaic)
        # upload files, and create record on images table, for each file
        res = upload_images_to_existing_mosaic_service(username, mosaic_id, files)
        # update user_mosaic table
        create_usermosaic_record(user_id=mosaic_owner_id, mosaic_id=mosaic_id)
        return res
    except (MemoryLimitExceeded, FileTooBig) as e:
        # clear unsuccessful request artifacts: files from minio, db records
        logger.debug(f"Image cannot be uploaded. Removing created mosaic")
        mosaic_delete_service(mosaic_id=mosaic_id, username=username)
        raise e


def get_image_by_id_service(username: str, image_id: UUID):
    # get mosaic_id using image_id
    mosaic_id = get_mosaic_id_of_image(image_id)
    if not mosaic_id:
        raise ItemNotFound(uid=image_id, instance_type='mosaic')
    # check if user has access to mosaic
    user_id = get_user_id(username)
    user_has_access = check_user_has_access_to_mosaic(user_id=user_id, mosaic_id=mosaic_id)

    # if user has access, return image and status_code=200
    if user_has_access:
        res = get_image_by_id(image_id=image_id)
        return res
    else:
        raise AccessDenied(uid=image_id, user=username, instance_type="image")


def get_mosaic_images_service(username: str, mosaic_id: UUID):
    """
    Get all images of a given mosaic.
    :param username: user login.
    :param mosaic_id: mosaic id.
    :return: List[images] if user has access, else raise AccessDenied error. Or if mosaic doesn't exist raise
    ItemNotFound error.
    """
    mosaic_exists = check_mosaic_exists(mosaic_id)
    if not mosaic_exists:
        raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
    user_id = get_user_id(username)
    user_has_access = check_user_has_access_to_mosaic(user_id=user_id, mosaic_id=mosaic_id)
    if not user_has_access:
        raise AccessDenied(uid=mosaic_id, user=username, instance_type="mosaic")
    results = get_images_by_mosaic_id(mosaic_id=mosaic_id)
    return [res for res in results]


def legacy_whitemaps_upload_service(username: str, file: UploadFile, tags: List[str]):
    mosaic_name = uuid.uuid4()
    if file.filename == '':
        file.filename = generate_file_name(suffix='tif')
    res = create_mosaic_and_file_upload_service(username=username, files=[file], tags=tags, name=str(mosaic_name))
    try:
        created_mosaic_id = res.get("mosaic_id", None)
    except:
        created_mosaic_id = None
    if not created_mosaic_id:
        raise InternalError(username=username, service="legacy_whitemaps_upload_service")
    # get images of mosaic
    image = get_images_by_mosaic_id(mosaic_id=created_mosaic_id).first()
    return {"url": image.image_url}


def link_image_from_minio_to_mosaic_service(username: str, mosaic_id: UUID, image_url: str):
    """
    The purpose is to link the images that are uploaded to minio without catalog,
     to the users mosaics (and moving to user's buckets)
    """
    mosaic_exists = check_mosaic_exists(mosaic_id)
    if not mosaic_exists:
        raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
    user_id = get_user_id(username)
    user_owns_mosaic = check_mosaic_owner(mosaic_id=mosaic_id, user_id=user_id)
    if not user_owns_mosaic:
        raise AccessDenied(uid=mosaic_id, user=username, instance_type="mosaic")
    available_disk_space = get_free_disk_space_of_user(login=username)
    with tempfile.TemporaryDirectory() as tmp_dir:
        s3_mosaic_path = get_mosaic_by_id(mosaic_id).mosaic_url
        filename, minio_bucket_from_url, minio_object_from_url = parse_image_url(image_url)
        if minio_bucket_from_url == '' or minio_object_from_url == '':
            raise InvalidLinkToMinio(object_url=image_url)
        # check if object exists at storage
        object_exists = minio_client.check_if_object_exists(image_url)
        if not object_exists:
            raise MinioObjectDoesntExist(object_url=image_url)
        image_path = Path(tmp_dir)/filename
        minio_client.download_object_by_url(url=image_url, file_path=str(image_path))
        # we should not allow to link the images that are already in the catalog
        # if the image is in the catalog, we should use another service
        if check_if_file_exists_by_image_url(image_url=image_url):
            raise FileAlreadyExists(url=image_url)
        handle_file_upload_to_mosaic(tmp_path=tmp_dir, filename=filename, s3_mosaic_path=s3_mosaic_path,
                                     mosaic_id=mosaic_id,
                                     available_disk_space=available_disk_space)
        # after moving the file to mosaic folder, we remove the old file
        minio_client.remove_objects_by_url([image_url])
    return {"message": "File successfully linked to a mosaic",
            "mosaic_id": mosaic_id}


def upload_images_to_existing_mosaic_service(username: str, mosaic_id: UUID, files: List[UploadFile]) -> dict:
    mosaic_exists = check_mosaic_exists(mosaic_id)
    if not mosaic_exists:
        raise ItemNotFound(uid=mosaic_id, instance_type='mosaic')
    user_id = get_user_id(username)
    user_owns_mosaic = check_mosaic_owner(mosaic_id=mosaic_id, user_id=user_id)
    if not user_owns_mosaic:
        raise AccessDenied(uid=mosaic_id, user=username, instance_type="mosaic")
    s3_mosaic_path = get_mosaic_by_id(mosaic_id).mosaic_url

    available_disk_space = get_free_disk_space_of_user(login=username)
    if available_disk_space <= 0 and MEMORY_LIMIT_SWITCH:
        file_size = len(files[0].file.read())
        raise MemoryLimitExceeded(available_memory=available_disk_space, memory_requested=file_size)  # ?
    for file in files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir)/file.filename
            try:
                save_upload_file_tmp(file, image_path)
                handle_file_upload_to_mosaic(tmp_path=str(tmp_dir),
                                             filename=file.filename, s3_mosaic_path=s3_mosaic_path,
                                             mosaic_id=mosaic_id,
                                             available_disk_space=available_disk_space)
                available_disk_space = available_disk_space - get_file_size(str(image_path))
            except Exception as e:
                raise e
            finally:
                file.file.close()

    return {"message": "Files successfully uploaded",
            "mosaic_id": mosaic_id}


def generate_xyz_link_for_mosaic(cog_link: str):
    if cog_link is None:
        return "", ""
    tileUrl = RASTER_TILE_SERVER_URL + "/api/v0/cogs/tiles/{z}/{x}/{y}.png?uri=" + cog_link
    tileJsonUrl = RASTER_TILE_SERVER_URL + "/api/v0/cogs/tiles.json?uri=" + cog_link
    return tileUrl, tileJsonUrl
