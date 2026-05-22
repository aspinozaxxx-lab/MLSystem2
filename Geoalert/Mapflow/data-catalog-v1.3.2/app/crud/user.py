from typing import Optional
from model import User
from functional.database import SessionLocal
from crud.image import get_all_images_of_user_by_login


def user_update(isAdmin: bool, login: str, memoryLimit: int):
    """
    Update/create user record in db
    1. query for user
    2.1. if user exists - update user
    2.2. if user doesn't exist - create user with memory_used=0
    :return:
    """
    with SessionLocal.begin() as session:
        user = session.query(User).filter(User.login == login).first()
        if user:
            user.memory_limit = memoryLimit
            session.commit()
        else:
            user = User(is_admin=isAdmin, login=login, memory_used=0, memory_limit=memoryLimit)
            session.add(user)
            session.commit()


def add_or_update_user(is_admin: bool, user_id: str, email: str):
    with SessionLocal.begin() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id, is_admin=is_admin, login=email, memory_used=0, memory_limit=0)
            session.add(user)
            session.commit()


def get_user_id(login: str):
    """
    Get user id from database by login
    :param login:
    :return: user_id [uuid]
    """
    with SessionLocal.begin() as session:
        user = session.query(User).filter(User.login == login).first()
        return user.id


def get_free_disk_space_of_user(login: str) -> int:
    """
    Checks available disk space for a given user. \n
    :param login: user login
    :return: Free space available (in bytes). Could be negative or zero, in case no space available for the user.
    """
    images_of_user = get_all_images_of_user_by_login(login=login)
    total_space_used = sum((image.file_size for image in images_of_user))
    with SessionLocal.begin() as session:
        user = session.query(User).filter(User.login == login).first()
        memory_limit = user.memory_limit
    return memory_limit - total_space_used


def user_memory_info(login: str):
    with SessionLocal.begin() as session:
        user = session.query(User).filter(User.login == login).first()
        memory_limit = user.memory_limit
    free_space = get_free_disk_space_of_user(login=login)
    return memory_limit, memory_limit - free_space, free_space
