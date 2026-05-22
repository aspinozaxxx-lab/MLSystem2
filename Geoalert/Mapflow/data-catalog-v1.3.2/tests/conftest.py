import os
import enum
import pytest
import uuid
from minio import Minio
from sqlalchemy import create_engine, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB, TEXT
from sqlalchemy import Column, ForeignKey, Integer, String, TIMESTAMP
from geoalchemy2 import Geometry

from .generate_files import create_tiff_file

# FIXME: import minio configuration and db configuration from config.py
# ================== Minio configuration ===========
minio_client = Minio("localhost:9000",
                     access_key=,
                     secret_key=,
                     secure=False)

# minio bucket
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'data-catalog-bucket')


# ================== Database configuration ========
DB_STRING = 'postgresql://{username}:{password}@{host}:{port}/{db_name}'
USERNAME = os.getenv('DB_USER', 'postgres')
PASSWORD = , '1234Qq')
HOST = os.getenv('DB_HOST', 'localhost')
PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'datacatalogdb')

DB_STRING = DB_STRING.format(username=USERNAME,
                             password=,
                             host=HOST,
                             port=PORT,
                             db_name=DB_NAME)


# ================== pytest fixtures =============================
@pytest.fixture(scope='session')
def get_files():
    # here we need to return checksum and full profile to check description
    description_inputs = [('./tests/test_files/aoi1.tif',
                           './tests/test_files/aoi1_thumb.tif',
                           {'height': 1281, 'width': 1025, 'count': 3,
                            'dtype': 'uint8', 'nodata': 0,
                            'crs': 'EPSG:3857', 'transform': (0.5, 0, 100000, 0, -0.5, 20000)},
                           (1024, 819),
                           '8860f22495a61b35e2756a5f2400d9e4abf6fda4')]

    preview_inputs = [('./tests/test_files/aoi2.tif',
                       './tests/test_files/aoi2_thumb.tif',
                       {'height': 3585, 'width': 4353},
                       (843, 1024),
                       None),
                      ('./tests/test_files/one_channel.tif',
                       './tests/test_files/one_channel_thumb.tif',
                       {'height': 3585, 'width': 4353, 'count': 1},
                       (843, 1024),
                       None)
                      ]

    all_inputs = description_inputs + preview_inputs
    for file in all_inputs:
        create_tiff_file(filename=file[0], **file[2])

    yield all_inputs

    for file in all_inputs:
        try:
            os.remove(file[0])
        except OSError:
            pass
        if file[1]:
            try:
                os.remove(file[1])
            except OSError:
                pass


@pytest.fixture(scope='session')
def large_tif():
    """
    Generate file with number of pixels greater than defined in configs.
    For tests using MAX_IMAGE_SIZE_PIXELS = 2000
    :return: yields created file.
    """
    width = height = 2001
    filename = './tests/test_files/large_tif.tif'
    create_tiff_file(filename=filename, width=width, height=height)
    yield filename

    try:
        os.remove(filename)
    except OSError:
        pass


@pytest.fixture(scope='session')
def urls():
    urls = {
        'root': 'http://localhost:8000/',
        'whitemaps-legacy-api': 'http://localhost:8000/rest/rasters',
        'upload-file-and-create-mosaic': 'http://localhost:8000/rest/rasters/mosaic/image?name={name}&tags={tag1}&tags={tag2}',
        'create-empty-mosaic': 'http://localhost:8000/rest/rasters/mosaic',
        'create-mosaic': 'http://localhost:8000/rest/rasters/mosaic?name={name}&tags={tag1}&tags={tag2}',
        'create-mosaic-without-tags': 'http://localhost:8000/rest/rasters/mosaic?name={name}',
        'get-mosaic': 'http://localhost:8000/rest/rasters/mosaic',
        'get-mosaic-by-id': 'http://localhost:8000/rest/rasters/mosaic/{mosaic_id}',
        'update-mosaic-by-id': 'http://localhost:8000/rest/rasters/mosaic/{mosaic_id}',
        'delete-mosaic-by-id': 'http://localhost:8000/rest/rasters/mosaic/{mosaic_id}',
        'get-mosaic-images-by-mosaic-id': 'http://localhost:8000/rest/rasters/mosaic/{mosaic_id}/image',
        'delete-image-from-mosaic': 'http://localhost:8000/rest/rasters/image/{image_id}',
    }
    yield urls


@pytest.fixture(scope='session')
def jsons():
    jsons = {
        'create_empty_mosaic': {'name': 'test_name', 'tags': ['tag1', 'tag2']},
        'update_mosaic': {'tags': ['new_tag1', 'new_tag2']}
    }
    yield jsons


@pytest.fixture(scope="session")
def connection():
    engine = create_engine(DB_STRING)
    return engine.connect()


@pytest.fixture
def db_session(connection):
    transaction = connection.begin()
    yield scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=connection)
    )


# FIXME: import db models from app package
# ======================= DB Models ==============
Base = declarative_base()


class Data(Base):
    __tablename__ = 'images'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mosaic_id = Column(UUID(as_uuid=True), ForeignKey('mosaics.id'))
    image_url = Column(String)
    preview_url_l = Column(String)
    preview_url_s = Column(String)
    uploaded_at = Column(TIMESTAMP(timezone=True))
    footprint = Column(Geometry('Polygon'))
    filename = Column(String)
    checksum = Column(String)
    meta_data = Column(JSONB)
    file_size = Column(Integer)
    cog_link = Column(String)


class Mosaic(Base):
    __tablename__ = 'mosaics'

    id = Column(UUID(as_uuid=True), primary_key=True)
    owner_id = Column(UUID(as_uuid=True))
    mosaic_url = Column(String)
    tags = Column(ARRAY(String))
    name = Column(String)
    created_at = Column(TIMESTAMP)
    cog_link = Column(String)


class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    is_admin = Column(Boolean)
    login = Column(String)
    memory_used = Column(Integer)
    memory_limit = Column(Integer)


class UserMosaic(Base):
    __tablename__ = 'usermosaic'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    mosaic_id = Column(UUID(as_uuid=True), ForeignKey('mosaics.id'))


class WorkflowDef(Base):
    __tablename__ = 'workflow_def'

    id = Column(Integer, primary_key=True)
    yaml = Column(TEXT)


class WorkflowStatus(enum.Enum):
    UNPROCESSED = 'UNPROCESSED'
    IN_PROGRESS = 'IN_PROGRESS'
    OK = 'OK'
    FAILED = 'FAILED'

    @classmethod
    def from_we_status(cls, we_status):
        if we_status == 'OK':
            return cls.OK
        elif we_status in ('FAILED', 'CANCELLED'):
            return cls.FAILED
        elif we_status in ('IN_PROGRESS', 'PAUSED'):
            return cls.IN_PROGRESS
        else:
            raise ValueError(f'Unknown WE status {we_status}')


class Workflow(Base):
    __tablename__ = 'workflow'

    image_id = Column(UUID(as_uuid=True), primary_key=True)
    we_id = Column(Integer)
    status = Column(Enum(WorkflowStatus), nullable=False)