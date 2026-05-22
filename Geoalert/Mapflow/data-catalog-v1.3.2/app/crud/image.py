import uuid
from datetime import datetime
from typing import List

from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy.dialects.postgresql import UUID

from functional.database import SessionLocal
from model import Data, Mosaic, User


def get_images_by_mosaic_id(mosaic_id: UUID) -> List[Data]:
    with SessionLocal.begin() as session:
        images = session.query(Data).where(Data.mosaic_id == mosaic_id)
    for image in images:
        # convert footprint record to json (dict)
        image.footprint = to_shape(image.footprint)
    return images


def get_all_images_of_user_by_login(login: str):
    with SessionLocal.begin() as session:
        images_of_user = session.query(
                Data
            ).join(
                Mosaic, Data.mosaic_id == Mosaic.id
            ).join(
                User, User.id == Mosaic.owner_id
            ).filter(
                User.login == login
            ).all()
    return images_of_user


def update_images_table(image_id: UUID, mosaic_id: uuid, image_url: str, preview_url_l: str, preview_url_s: str,
                        footprint: str, filename: str, sha1_checksum: str, meta_data: dict, file_size: int):
    with SessionLocal.begin() as session:
        data = Data(id=image_id, mosaic_id=mosaic_id, image_url=image_url,
                    preview_url_l=preview_url_l, preview_url_s=preview_url_s,
                    uploaded_at=datetime.now(),
                    footprint=footprint, filename=filename, checksum=sha1_checksum, meta_data=meta_data,
                    file_size=file_size)
        session.add(data)
        session.commit()
    return True


def update_preview_url_record(image_id: UUID, preview_type: str, preview_url: str) -> None:
    """
    Updates images table adding missing preview url
    """
    with SessionLocal.begin() as session:
        image = session.query(Data).filter(Data.id == image_id).first()
        if preview_type == 's':
            image.preview_url_s = preview_url
        else:
            image.preview_url_l = preview_url
        session.commit()


def delete_image_records_by_uid(uids: List[UUID]):
    with SessionLocal.begin() as session:
        for uid in uids:
            session.query(Data).filter(Data.id == uid).delete()
        session.commit()


def get_image_by_id(image_id: UUID) -> Data:
    with SessionLocal.begin() as session:
        image = session.query(Data).filter(Data.id == image_id).first()
    image.footprint = to_shape(image.footprint)
    return image


def get_first_image_of_mosaic(mosaic_id: UUID) -> Data:
    """
    Get first uploaded image of mosaic.
    """
    with SessionLocal.begin() as session:
        image = session.query(Data).filter(Data.mosaic_id == mosaic_id).order_by(Data.uploaded_at).first()
    image.footprint = to_shape(image.footprint)
    return image


def check_if_file_exists_by_image_url(image_url: str):
    with SessionLocal.begin() as session:
        image = session.query(Data).filter(Data.image_url == image_url).first()
    if not image:
        return False
    return True


def check_image_exists(image_id: UUID) -> bool:
    with SessionLocal.begin() as session:
        image = session.query(Data).where(Data.id == image_id)
    if image.first():
        return True
    return False


def get_mosaic_id_of_image(image_id: UUID):
    with SessionLocal.begin() as session:
        image = session.query(Data).where(Data.id == image_id).first()
    if image:
        return image.mosaic_id
    else:
        return False


def add_coglink_to_images_table(image_id: UUID, coglink: str):
    with SessionLocal.begin() as session:
        image = session.query(Data).where(Data.id == image_id).first()
        image.cog_link = coglink
        session.commit()
