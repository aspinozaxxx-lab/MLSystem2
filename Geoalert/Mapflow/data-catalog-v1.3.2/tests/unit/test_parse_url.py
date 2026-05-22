from app.functional.urlparse import parse_image_url


def test_parse_image_url():
    input = "s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/3bfe115d-b261-46a5-81dc-6d7e6d42cc33/aoi2.tif"
    filename = "aoi2.tif"
    bucket = "users-data"
    object = "a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/3bfe115d-b261-46a5-81dc-6d7e6d42cc33/aoi2.tif"

    assert parse_image_url(input) == (filename, bucket, object)