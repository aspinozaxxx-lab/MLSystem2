from data_validator_lib.errors.local import *


def test_image_metadata_must_be_dict():
    err = ImageMetadataMustBeDict()
    assert err.log_message == "ERROR: Image metadata must be a dict (json)"

def test_image_metadata_key_error():
    err = ImageMetadataKeyError()
    assert err.log_message.startswith("ERROR: Image metadata must have keys:")
    assert all(key in err.log_message for key in ['crs', 'count', 'transform', 'dtype'])

def test_s3_url_error():
    err = S3URLError(actual_s3_link="s3://bucket/folder")
    assert err.log_message == "ERROR: URL of the image at s3 storage must be a string starting with s3://, " \
                              "and must be either folder ending with `/` or .tif/.tiff file " \
                              "got s3://bucket/folder"

def test_image_must_be_tiff():
    err = ImageMustBeTiff(actual_s3_link="s3://bucket/file.jpg")
    assert err.log_message == "ERROR: Image must be a tiff file, got url s3://bucket/file.jpg"

def test_local_request_key_error():
    err = LocalRequestKeyError()
    assert err.log_message == "ERROR: Request must contain either `profile` or `url` keys"

def test_read_from_s3_failed():
    err = ReadFromS3Failed(s3_link="s3://bucket/file.tif", message="some error")
    assert err.log_message == "ERROR: Failed to read file from s3://bucket/file.tif, error: some error"

def test_image_read_error():
    err = ImageReadError(message="some error")
    assert err.log_message == "ERROR: Failed to open the file or read metadata, error: some error"

def test_dtype_not_allowed():
    err = DtypeNotAllowed(required_dtypes=["uint8", "uint16"], request_dtype="uint32")
    assert err.log_message == "ERROR: Dtype must be one of ['uint8', 'uint16'], got uint32"

def test_pixel_size_too_low():
    err = PixelSizeTooLow(actual_res=0.5, min_res=1)
    assert err.log_message == "ERROR: Spatial resolution is too high. " \
                              "Got 0.5, minimum allowed pixel size is 1"

def test_pixel_size_too_high():
    err = PixelSizeTooHigh(actual_res=1, max_res=0.5)
    assert err.log_message == "ERROR: Spatial resolution is too low. " \
                                "Got 1, maximum allowed pixel size is 0.5"

def test_nchannels_not_allowed():
    err = NChannelsNotAllowed(required_nchannels=1, real_nchannels=3)
    assert err.log_message == "ERROR: Number of channels must be 1, got 3"

def test_bad_image_profile():
    err = BadImageProfile(profile={"crs": "EPSG:4326"}, required_keys=["crs", "transform"])
    assert err.log_message == "ERROR: Image profile (metadata) must have keys ['crs', 'transform']," \
                              "got profile: {'crs': 'EPSG:4326'}"

def test_image_check_error():
    err = ImageCheckError(checked_param="param", metadata="metadata", err_message="message")
    assert err.log_message == "ERROR: Error occurred during image param check: message. Image metadata = metadata"

def test_empty_folder():
    err = EmptyFolder(s3_link="s3://bucket/folder")
    assert err.log_message == "ERROR: S3 folder object `s3://bucket/folder` does not contain any images"