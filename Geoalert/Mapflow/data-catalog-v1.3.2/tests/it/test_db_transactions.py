import uuid
from tests.conftest import User, WorkflowDef

userdata = {
    "id": uuid.uuid4(),
    "is_admin": True,
    "login": "login",
    "memory_used": 225545,
    "memory_limit": 3322551
}


def test_write_to_db(db_session):
    user = User(**userdata)
    db_session.add(user)
    db_session.commit()
    some_user = db_session.query(User).filter(User.id == user.id).first()
    assert some_user.id == user.id
    assert some_user.is_admin == user.is_admin
    assert some_user.memory_limit == user.memory_limit


def test_wd_is_stored(db_session):
    wd = db_session.query(WorkflowDef).first()
    # from mock auth_server
    assert wd.id == 100
