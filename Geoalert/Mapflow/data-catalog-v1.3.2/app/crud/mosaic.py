from typing import List
from sqlalchemy.dialects.postgresql import UUID

from functional.database import SessionLocal
from model import Mosaic


def create_mosaics_record(mosaic: Mosaic):
    with SessionLocal.begin() as session:
        session.add(mosaic)
        session.commit()
    return True


def get_mosaic_by_id(uid: UUID) -> Mosaic:
    with SessionLocal.begin() as session:
        mosaic = session.query(Mosaic).filter(Mosaic.id == uid).first()
    return mosaic


def update_mosaic_record(mosaic: Mosaic):
    with SessionLocal.begin() as session:
        res = session.query(Mosaic).filter(Mosaic.id == mosaic.id).first()
        res.tags = mosaic.tags
        res.name = mosaic.name
        session.commit()


def check_mosaic_owner(mosaic_id: UUID, user_id: UUID) -> bool:
    """
    Check if given user owns mosaic or not
    :param mosaic_id: UUID
    :param user_id: UUID
    :return: True if given user owns mosaic, else False
    """
    with SessionLocal.begin() as session:
        mosaic = session.query(Mosaic).where(Mosaic.id == mosaic_id, Mosaic.owner_id == user_id)
        if mosaic.first():
            return True
    return False


def check_mosaic_exists(mosaic_id: UUID) -> bool:
    with SessionLocal.begin() as session:
        mosaic = session.query(Mosaic).where(Mosaic.id == mosaic_id)
    if mosaic.first():
        return True
    return False


def get_all_mosaics_of_user(user_id: UUID):
    with SessionLocal.begin() as session:
        mosaics = session.query(Mosaic).where(Mosaic.owner_id == user_id)
    res = []
    for mosaic in mosaics:
        res.append(mosaic)
    return res


def get_mosaics_of_user_by_tag(user_id: UUID, tags: List[str]):
    # mosaics_of_user = get_all_mosaics_of_user(user_id)
    # query = session.query(TestUser).filter(TestUser.numbers.contains([some_int])).all()
    with SessionLocal.begin() as session:
        mosaics = session.query(Mosaic).where(Mosaic.owner_id == user_id).filter(
            Mosaic.tags.contains(tags)).all()
    res = []
    for mosaic in mosaics:
        res.append(mosaic)
    return res


def delete_mosaic_record_by_mosaic_id(mosaic_id):
    with SessionLocal.begin() as session:
        session.query(Mosaic).filter(Mosaic.id == mosaic_id).delete()
        session.commit()
