import os
import uuid
import requests
from time import sleep
from loguru import logger
from geoalchemy2.shape import to_shape
from app.functional.data import get_footprint, get_file_description
from tests.conftest import Data, Mosaic, Workflow, WorkflowStatus


def test_multiple_file_upload_s3_unauthorized_request(get_files, urls):
    FILE1, FILE2, FILE3 = (file[0] for file in get_files)
    url = urls.get('upload-file-and-create-mosaic')

    with open(FILE1, 'rb') as f1:
        # files to post request
        files = [('files', f1)]
        response = requests.post(url=url, files=files)
    assert response.status_code == 401


def test_file_upload_s3_authorized_request(get_files, urls, db_session):
    mosaic_name = uuid.uuid4().hex
    url = urls.get('upload-file-and-create-mosaic').format(name=mosaic_name, tag1='tag1', tag2='tag2')
    filename = get_files[0][0]

    with open(filename, 'rb') as f1:
        # file to post request
        files = [('file', f1)]
        response = requests.post(url=url, files=files, auth=('another_login', 'another_password'))
    assert response.status_code == 200
    file = [filename]
    mosaic_id = response.json().get('mosaic_id')
    if not mosaic_id:
        assert False, 'Incorrect response!'

    image_records = db_session.query(Data).filter(Data.mosaic_id == mosaic_id).all()
    logger.info(f"Image records: {image_records}")
    for image_record, filename in zip(image_records, file):
        workflow_record = db_session.query(Workflow).filter(Workflow.image_id == image_record.id).first()
        db_session.refresh(workflow_record)
        logger.info(f"Workflow: {workflow_record.image_id}, {workflow_record.we_id}")
        image_record.footprint = to_shape(image_record.footprint)
        image_footprint = get_footprint(os.path.abspath(filename))
        image_filename, image_checksum, image_metadata = get_file_description(file_path=filename,
                                                                              filename=filename.split(sep='/')[-1])
        assert image_filename == image_record.filename
        assert image_checksum == image_record.checksum
        assert image_footprint.type == image_record.footprint.type
        assert image_record.footprint.almost_equals(image_footprint, 12)
        assert workflow_record.status.value == WorkflowStatus.UNPROCESSED.value
        # as sqlalchemy converts tuple to list type, before asserting, perform type conversion
        for key in image_metadata:
            if type(image_metadata[key]) == tuple:
                assert list(image_metadata[key]) == image_record.meta_data[key]
            else:
                assert image_metadata[key] == image_record.meta_data[key]
    # Test that COG processing is created.
    # It will be after the first polling request
    sleep(10)
    for record in image_records:
        db_session.refresh(record)
    db_session.refresh(workflow_record)
    image_records = db_session.query(Data).filter(Data.mosaic_id == mosaic_id).all()
    for image_record, filename in zip(image_records, file):
        workflow_record = db_session.query(Workflow).filter(Workflow.image_id == image_record.id).first()
        db_session.refresh(workflow_record)
        logger.info(f"Workflow: {workflow_record.image_id}, {workflow_record.we_id}")
        assert workflow_record.status.value == WorkflowStatus.IN_PROGRESS.value
        # as sqlalchemy converts tuple to list type, before asserting, perform type conversion
    # Test that COG is processed.
    # It will be after the first polling request
    sleep(12)
    for record in image_records:
        db_session.refresh(record)
    db_session.refresh(workflow_record)
    image_records = db_session.query(Data).filter(Data.mosaic_id == mosaic_id).all()
    for image_record, filename in zip(image_records, file):
        workflow_record = db_session.query(Workflow).filter(Workflow.image_id == image_record.id).first()
        db_session.refresh(workflow_record)
        logger.info(f"Workflow: {workflow_record.image_id}, {workflow_record.we_id}")
        assert workflow_record.status.value == WorkflowStatus.OK.value
        # as sqlalchemy converts tuple to list type, before asserting, perform type conversion


def test_whitemaps_legacy_api(get_files, urls, db_session):
    url = urls.get('whitemaps-legacy-api')
    filename = get_files[0][0]

    with open(filename, 'rb') as f:
        file = [('file', f)]
        resp = requests.post(url=url, files=file, auth=('another_login', 'another_password'))
        # response is in the form: {"url": "s3://data-catalog/user_id/mosaic_id/filename"}
    assert resp.status_code == 200
    assert "url" in resp.json().keys()
    mosaic_id = resp.json().get("url").split(sep='/')[-2]
    if not mosaic_id:
        assert False, 'Incorrect response!'
    image_record = db_session.query(Data).filter(Data.mosaic_id == mosaic_id).first()
    image_record.footprint = to_shape(image_record.footprint)
    image_footprint = get_footprint(filename)
    image_filename, image_checksum, image_metadata = get_file_description(file_path=filename,
                                                                          filename=filename.split(sep='/')[-1])
    assert image_filename == image_record.filename
    assert image_checksum == image_record.checksum
    assert image_footprint.type == image_record.footprint.type
    assert image_record.footprint.almost_equals(image_footprint, 12)

    # as sqlalchemy converts tuple to list type, before asserting, perform type conversion
    for key in image_metadata:
        if type(image_metadata[key]) == tuple:
            assert list(image_metadata[key]) == image_record.meta_data[key]
        else:
            assert image_metadata[key] == image_record.meta_data[key]


def test_upload_file_with_large_number_of_pixels(large_tif, urls):
    mosaic_name = uuid.uuid4().hex
    url = urls.get('upload-file-and-create-mosaic').format(name=mosaic_name, tag1='tag1', tag2='tag2')
    filename = large_tif

    with open(filename, 'rb') as f1:
        # file to post request
        files = [('file', f1)]
        response = requests.post(url=url, files=files, auth=('another_login', 'another_password'))
    assert response.status_code == 422
