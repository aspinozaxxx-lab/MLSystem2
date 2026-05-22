import os

import pytest
import maploader
from maploader.errors import TileNotLoaded

URLS = [
    "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga"
]

MAXAR_URLS = ["https://securewatch.maxar.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService"
              "&FORMAT=image/png&TileRow={{y}}&TileCol={{x}}&TileMatrixSet=EPSG:3857"
              "&TileMatrix=EPSG:3857:{{z}}&CONNECTID={connectid}"]

REGION_1 = {
    "type": "Polygon",
    "coordinates": [
        [
            [
                41.71634316444397,
                36.04331291934319
            ],
            [
                41.71774059534073,
                36.04331291934319
            ],
            [
                41.71774059534073,
                36.04443849786634
            ],
            [
                41.71634316444397,
                36.04443849786634
            ],
            [
                41.71634316444397,
                36.04331291934319
            ]
        ]
    ]
}
REGION_2 = {
        "type": "Polygon",
        "coordinates": [
          [
            [
              59.89420741796493,
              75.65265641593838
            ],
            [
              59.8946526646614,
              75.65265641593838
            ],
            [
              59.8946526646614,
              75.65276275961564
            ],
            [
              59.89420741796493,
              75.65276275961564
            ],
            [
              59.89420741796493,
              75.65265641593838
            ]
          ]
        ]
      }


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("zoom", [16, 18])
def test_download(url, zoom, test_data):
    maploader.download(
        url,
        zoom=zoom,
        geometry=REGION_1,
        output_fp=os.path.join(test_data, "output_{}.tif".format(zoom)),
    )


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("zoom", [16])
def test_ignore_errors(url, zoom, test_data):
    maploader.download(
        url,
        zoom=zoom,
        geometry=REGION_2,
        output_fp=os.path.join(test_data, "output_empty_{}.tif".format(zoom)),
        ignore_errors=True,
    )


@pytest.mark.parametrize("url", MAXAR_URLS)
@pytest.mark.parametrize("zoom", [11])
def test_authorized_download(url, zoom, test_data):
    login = os.getenv('MAXAR_LOGIN')
    password = 
    connectid = os.getenv('SECURE_WATCH_CONNECT_ID')
    if not login or not password or not connectid:
        pytest.skip('Specify MAXAR_LOGIN, SECURE_WATCH_CONNECT_ID and MAXAR_PASSWORD env variables to run this test')
    url = url.format(connectid=connectid)
    maploader.download(
        url,
        zoom=zoom,
        geometry=REGION_2,
        output_fp=os.path.join(test_data, "output_maxar.tif"),
        ignore_errors=True,
        credentials=(login, password),
        retry_attempts=1,
        retry_delay=1
    )


@pytest.mark.parametrize("url", MAXAR_URLS)
@pytest.mark.parametrize("zoom", [11])
def test_bad_unauthorized_download(url, zoom, test_data):
    connectid = os.getenv('SECURE_WATCH_CONNECT_ID')
    if not connectid:
        pytest.skip('Specify SECURE_WATCH_CONNECT_ID env variable to run this test')
    url = url.format(connectid=connectid)
    with pytest.raises(TileNotLoaded) as e:
        maploader.download(
            url,
            zoom=zoom,
            geometry=REGION_2,
            output_fp=os.path.join(test_data, "output_maxar.tif"),
            ignore_errors=False,
            credentials=('wronglogin', 'wrongpassword'),
            retry_attempts=1,
            retry_delay=1
        )
    assert e.value.parameters['status'] in (401, 403)
