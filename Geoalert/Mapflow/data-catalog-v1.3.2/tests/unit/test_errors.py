import pytest

from errors import (DataCatalogError,
                        MemoryLimitExceeded,
                        FileTooBig,
                        FileOpenError,
                        FileCheckFailed,
                        ImageOutOfBounds,
                        ImageExtentTooBig,
                        ItemNotFound,
                        AccessDenied,
                        InternalError,
                        FileAlreadyExists,
                        PreviewNotFound,
                        FileValidationFailed)


def test_init_raises_assertion_error_on_insufficient_params():
    with pytest.raises(AssertionError):
        raise DataCatalogError(message="Message with placeholder {param}")
    with pytest.raises(AssertionError):
        raise DataCatalogError(message="Message with two placeholders {param1} {param2}", param1=0)


def test_all_errors_format():
    with pytest.raises(DataCatalogError):
        raise MemoryLimitExceeded(100, 200)
    with pytest.raises(DataCatalogError):
        raise FileTooBig(100)
    with pytest.raises(DataCatalogError):
        raise FileOpenError("input.tif")
    with pytest.raises(DataCatalogError):
        raise FileCheckFailed('input.tif', ('crs', 'width'), None, (1, 0, 0, 0, -1, 0), 300000, 1000)
    with pytest.raises(DataCatalogError):
        raise ImageOutOfBounds()
    with pytest.raises(DataCatalogError):
        raise ImageExtentTooBig()
    with pytest.raises(DataCatalogError):
        raise ItemNotFound('3a2ce002-9818-4bcc-96e7-c44730501145', 'mosaic')
    with pytest.raises(DataCatalogError):
        raise AccessDenied('3a2ce002-9818-4bcc-96e7-c44730501145', 'user@geoaert.io', 'mosaic')

    # Internal error can be with any of the parameters, so several different variants are tested
    with pytest.raises(DataCatalogError):
        raise InternalError()
    with pytest.raises(DataCatalogError):
        raise InternalError(username="user@geoalert.io")
    with pytest.raises(DataCatalogError):
        raise InternalError(service="test_service")
    with pytest.raises(DataCatalogError):
        raise InternalError(username="user@geoalert.io",
                            mosaic_id="3a2ce002-9818-4bcc-96e7-c44730501145",
                            image_id="3a2ce002-9818-4bcc-96e7-c44730501145",
                            service="test_service")

    with pytest.raises(DataCatalogError):
        raise FileAlreadyExists(url="s3:/healthcheck/area-113229.tif")
    with pytest.raises(DataCatalogError):
        raise PreviewNotFound(image_id="3a2ce002-9818-4bcc-96e7-c44730501145")
    with pytest.raises(DataCatalogError):
        raise FileValidationFailed(mosaic_id="3a2ce002-9818-4bcc-96e7-c44730501145",
                                   filename="file.tif",
                                   param_name="count",
                                   got_param=4,
                                   expected_param=3)


