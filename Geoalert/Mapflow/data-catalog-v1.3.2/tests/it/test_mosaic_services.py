import uuid
import requests
from tests.conftest import Mosaic, User, UserMosaic, Data
USERNAME = "another_login"
PASSWORD = 

json = {
    "tags": ["autumn", "winter"]
}

tags = ['autumn', 'winter']


def test_create_mosaic(urls, db_session, jsons):
    data = jsons.get('create_empty_mosaic')
    url = urls.get('create-empty-mosaic')
    res = requests.post(url=url, auth=(USERNAME, PASSWORD), json=data)
    assert res.status_code == 200
    for key in data.keys():
        assert key in res.json().keys()
        assert data.get(key) == res.json().get(key)
    mosaic_id = res.json().get('id')
    mosaic_tags = res.json().get('tags')
    if (not mosaic_id) or (not mosaic_tags):
        assert False, 'Invalid response'
    mosaic_record = db_session.query(Mosaic).filter(Mosaic.id == mosaic_id).first()
    assert mosaic_record.tags == mosaic_tags


def test_get_all_mosaics_of_user(urls, db_session):
    url = urls.get('get-mosaic')
    res = requests.get(url=url, auth=(USERNAME, PASSWORD))
    assert res.status_code == 200
    # get mosaics of a given user from db: get user id by login, query db by id
    user_id = db_session.query(User).filter(User.login == USERNAME).first().id

    mosaics_of_user = db_session.query(Mosaic).where(Mosaic.owner_id == user_id).all()
    for mosaic_from_response, mosaic_record in zip(res.json(), mosaics_of_user):
        assert mosaic_from_response.get('id') == str(mosaic_record.id)
        assert mosaic_from_response.get('tags') == mosaic_record.tags


def test_get_mosaic_by_id(urls, jsons):
    data = jsons.get('create_empty_mosaic')
    url = urls.get('create-empty-mosaic')
    created_mosaic = requests.post(url=url, auth=(USERNAME, PASSWORD), json=data)
    created_mosaic_id = created_mosaic.json().get('id')

    url = urls.get('get-mosaic-by-id').format(mosaic_id=created_mosaic_id)
    res = requests.get(url=url, auth=(USERNAME, PASSWORD))
    assert res.status_code == 200
    assert res.json().get('id') == created_mosaic_id


def test_update_mosaic_by_id(urls, db_session, jsons):
    data = jsons.get('create_empty_mosaic')
    url = urls.get('create-empty-mosaic')
    created_mosaic = requests.post(url=url, auth=(USERNAME, PASSWORD), json=data)
    created_mosaic_id = created_mosaic.json().get('id')
    new_json = jsons.get('update_mosaic')
    new_mosaic_name = uuid.uuid4().hex
    new_json['name'] = new_mosaic_name
    url = urls.get('update-mosaic-by-id').format(mosaic_id=created_mosaic_id)
    res = requests.put(url=url, auth=(USERNAME, PASSWORD), json=new_json)
    assert res.status_code == 200
    for key in new_json.keys():
        assert key in res.json().keys()
        assert new_json.get(key) == res.json().get(key)
    mosaic_id = res.json().get('id')
    mosaic_tags = res.json().get('tags')
    if (not mosaic_id) or (not mosaic_tags):
        assert False, 'Invalid response'
    mosaic_record = db_session.query(Mosaic).filter(Mosaic.id == mosaic_id).first()
    assert mosaic_record.tags == mosaic_tags


def test_delete_mosaic_by_id(urls, db_session, jsons):
    data = jsons.get('create_empty_mosaic')
    url = urls.get('create-empty-mosaic')
    created_mosaic = requests.post(url=url, auth=(USERNAME, PASSWORD), json=data)
    created_mosaic_id = created_mosaic.json().get('id')
    url = urls.get('delete-mosaic-by-id').format(mosaic_id=created_mosaic_id)
    res = requests.delete(url=url, auth=(USERNAME, PASSWORD))
    assert res.status_code == 200
    mosaic_record = db_session.query(Mosaic).filter(Mosaic.id == created_mosaic_id).first()
    assert not mosaic_record


def test_get_mosaic_images_by_mosaic_id(get_files, urls, db_session):
    FILE1 = get_files[0][0]
    url = urls.get('upload-file-and-create-mosaic')

    with open(FILE1, 'rb') as f1:
        # files to post request
        files = [('file', f1)]
        response = requests.post(url=url, files=files,
                                 auth=('another_login', 'another_password'))
    created_mosaic_id = response.json().get('mosaic_id')

    url = urls.get('get-mosaic-images-by-mosaic-id').format(mosaic_id=created_mosaic_id)
    res = requests.get(url=url, auth=(USERNAME, PASSWORD))
    image_records = db_session.query(Data).filter(Data.mosaic_id == created_mosaic_id)
    for image, image_record in zip(res.json(), image_records):
        assert image.get('id') == str(image_record.id)


def test_delete_image_from_mosaic(urls, db_session, get_files):
    # upload images to service
    FILE1, FILE2, FILE3 = (file[0] for file in get_files)
    mosaic_name = uuid.uuid4().hex
    url = urls.get('upload-file-and-create-mosaic').format(name=mosaic_name, tag1='tag1', tag2='tag2')

    with open(FILE1, 'rb') as f1:
        # files to post request
        files = [('file', f1)]
        response = requests.post(url=url, files=files,
                                 auth=('another_login', 'another_password'))
    created_mosaic_id = response.json().get('mosaic_id')
    assert response.status_code == 200
    # request recently created mosaic from service
    url = urls.get('get-mosaic-images-by-mosaic-id').format(mosaic_id=created_mosaic_id)
    res = requests.get(url=url, auth=(USERNAME, PASSWORD))

    # must report success
    assert res.status_code == 200  # , json.dumps(response.json())

    # take first image from response
    first_image = res.json()[0]
    image_id_to_delete = first_image.get('id')
    image_minio_path = first_image.get('image_url')
    if (not image_id_to_delete) or (not image_minio_path):
        assert False, 'Invalid response!'

    # request to delete image from mosaic
    url = urls.get('delete-image-from-mosaic').format(image_id=image_id_to_delete)
    res = requests.delete(url=url, auth=(USERNAME, PASSWORD))
    assert res.status_code == 200
    # check if image deleted from db and minio
    image_record = db_session.query(Data).filter(Data.mosaic_id == created_mosaic_id, Data.id == image_id_to_delete).\
        first()
    assert not image_record
