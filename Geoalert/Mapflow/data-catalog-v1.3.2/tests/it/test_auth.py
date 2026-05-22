import requests


# TODO maybe remove this test? or change so that, it uses real credentials for whitemaps?
def test_auth_backend_response(urls):
    url = urls.get('root')
    response = requests.get(url)
    assert response.status_code == 401

    response = requests.get(url=url, auth=("wrong-login", "wrong-password"))
    assert response.status_code == 401

    response = requests.get(url=url, auth=("another_login", "another_password"))
    assert response.status_code == 200

    response = requests.get(url=url, auth=("some_login", "1234567890"))
    assert response.status_code == 200
