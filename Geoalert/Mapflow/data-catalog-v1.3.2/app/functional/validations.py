from pathlib import Path
from typing import Union

import rasterio
import shapely.geometry
from affine import Affine
from rasterio import RasterioIOError, CRS
from sqlalchemy.dialects.postgresql import UUID

from config import Config
from errors import (FileCheckFailed,
                    ImageOutOfBounds,
                    ImageExtentTooBig,
                    FileValidationFailed,
                    FileOpenError)


# =================== Validate file parametres by itself ====================== #

def check_file(filename: Union[str, Path]):
    """
    Preliminary check of the file.
    - file must open
    - crs must be valid
    - transform must be there and not Identity
    - maximum width and height

    Args:
        filename: str or path-like, path to file
    Returns: None if file is considered a valid dataset, raises FileCheckError of FileOpenError if the data is corrupted
    """
    try:
        with rasterio.open(filename) as src:
            profile = src.profile
            if src.driver != 'GTiff':
                raise RasterioIOError
    except RasterioIOError:
        # Raised when a dataset cannot be opened using one of the registered format drivers
        raise FileOpenError(filename=str(filename))

    crs = profile.get('crs', None)
    transform = profile.get('transform', None)
    width = profile.get('width', None)
    height = profile.get('height', None)
    bad_params = []
    if not isinstance(crs, CRS) or not crs.is_valid:
        bad_params.append('crs')
    if not transform or transform == Affine.identity():
        # identity can be generated in case there is no transfrom in the file. This should be treated as error
        bad_params.append('transform')
    if not width or width > Config.MAX_IMAGE_SIZE_PIXELS:
        bad_params.append('width')
    if not height or height > Config.MAX_IMAGE_SIZE_PIXELS:
        bad_params.append('height')
    if bad_params:
        raise FileCheckFailed(filename=str(filename), bad_parameters=bad_params,
                              crs=crs, transform=transform, width=width, height=height)


def validate_footprint(footprint: shapely.geometry.base.BaseGeometry):
    minx, miny, maxx, maxy = footprint.bounds
    if minx < - 360*3 or minx > 360*3 \
            or maxx < - 360*3 or maxx > 360*3 \
            or miny < - 360*3 or miny > 360*3 \
            or maxy < - 360*3 or maxy > 360*3:
        raise ImageOutOfBounds()
    elif footprint.area > Config.MAX_IMAGE_AREA_DEGREES:
        raise ImageExtentTooBig()


# ============= Validate file for fit to the mosaic ===================== #

def validate_metadata(source_metadata: dict, target_metadata: dict, mosaic_id: UUID, filename: str) -> None:
    """
    :param source_metadata: file metadata being uploaded.
    :param target_metadata: mosaic metadata where file is being uploaded to.
    :return: None, if everything is ok, else raise corresponding exception.
    """
    if not list(source_metadata.get("dtypes")) == list(target_metadata.get("dtypes")):
        raise FileValidationFailed(mosaic_id=mosaic_id, filename=filename,
                                   param_name="dtypes",
                                   got_param=source_metadata.get("dtypes"),
                                   expected_param=target_metadata.get("dtypes"))
    if not source_metadata.get("count") == target_metadata.get("count"):
        raise FileValidationFailed(mosaic_id=mosaic_id, filename=filename,
                                   param_name="count",
                                   got_param=source_metadata.get("count"),
                                   expected_param=target_metadata.get("count"))
    if not source_metadata.get("crs") == target_metadata.get("crs"):
        raise FileValidationFailed(mosaic_id=mosaic_id, filename=filename,
                                   param_name="crs",
                                   got_param=source_metadata.get("crs"),
                                   expected_param=target_metadata.get("crs"))
    if not resolution_validate(source_metadata.get("pixel_size"), target_metadata.get("pixel_size")):
        raise FileValidationFailed(mosaic_id=mosaic_id, filename=filename,
                                   param_name="pixel_size",
                                   got_param=source_metadata.get("pixel_size"),
                                   expected_param=target_metadata.get("pixel_size"))
    return


def resolution_validate(source_res, target_res) -> bool:
    x_target_res = target_res[0]
    y_target_res = target_res[1]

    x_source_res = source_res[0]
    y_source_res = source_res[1]

    if x_target_res == x_source_res and y_target_res == y_source_res:
        return True

    return True if (
        2 * abs((x_target_res - x_source_res)) / (x_target_res + x_source_res) < 0.1 and
        2 * abs((y_target_res - y_source_res)) / (y_target_res + y_source_res) < 0.1
    ) else False

# =========================================================== #