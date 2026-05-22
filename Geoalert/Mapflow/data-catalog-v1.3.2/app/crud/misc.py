import uuid
from loguru import logger
from sqlalchemy.dialects.postgresql import UUID
from functional.database import SessionLocal
from model import UserMosaic, Data


# ---------------------------------------------- usermosaic management ---------------------------
def check_user_has_access_to_mosaic(user_id: uuid, mosaic_id: uuid) -> bool:
    with SessionLocal.begin() as session:
        usermosaic = session.query(UserMosaic).where(UserMosaic.user_id == user_id,
                                                     UserMosaic.mosaic_id == mosaic_id)
        if usermosaic.first():
            return True
    return False


def check_user_has_access_to_image(user_id: UUID, image_id: UUID) -> bool:
    # get mosaic_id to which image belongs
    with SessionLocal.begin() as session:
        image_record = session.query(Data).filter(Data.id == image_id).first()
        mosaic_id = image_record.mosaic_id
    # check if user has access to that mosaic:
    return check_user_has_access_to_mosaic(user_id=user_id, mosaic_id=mosaic_id)


def create_usermosaic_record(user_id: uuid, mosaic_id: uuid):
    with SessionLocal.begin() as session:
        usermosaic = UserMosaic(user_id=user_id, mosaic_id=mosaic_id)
        session.add(usermosaic)
        session.commit()
    return True


def delete_usermosaic_record_of_user_by_mosaic_id(user_id: UUID, mosaic_id: UUID):
    with SessionLocal.begin() as session:
        session.query(UserMosaic).filter(UserMosaic.mosaic_id == mosaic_id,
                                         UserMosaic.user_id == user_id).delete()
        session.commit()


# ------------------------------------------ db healthcheck -----------------------------------
def db_healthcheck() -> bool:
    try:
        with SessionLocal.begin() as session:
            session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.exception(e)
        return False
