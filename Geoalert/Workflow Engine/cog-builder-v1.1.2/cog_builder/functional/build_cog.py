import os
import json
import tempfile
import rasterio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Mapping, Any, Optional
from osgeo import gdal
from loguru import logger
from tempfile import TemporaryDirectory

from .errors import CogReprojectionError, InputDataError
from .geometry import mapping

MAXIMUM_DATASOURCE_SIZE = 100_000 # in pixels
MINIMUM_REJECTED_PIXEL_SIZE = 0.01 # 10mm
MINIMUM_ALLOWED_PIXEL_SIZE = 0.0745 # 21th zoom

DEFAULT_CHANNELS = [1, 2, 3]


class BandStats:
    def __init__(self,
                 band: int,
                 type: str,
                 metadata: dict,
                 min: Optional[float] = None,
                 max: Optional[float] = None,
                 mean: Optional[float] = None,
                 stdDev: Optional[float] = None,
                 colorInterpretation: Optional[str] = None,
                 **kwargs):
        """
        Parsing statistics from gdal.Info about Band

        Presumably, it has min/max/mean/stdDev values, but sometimes they are inside ['metadata'][''] dict
        with different keys.
        This is not properly documented in gdal documentation and checked on a narrow subset of images,
        so it most probably will need correction for real data
        """
        metadata = metadata.get('', {})
        self.band = int(band)
        self.type = type
        if min is not None:
            self.min = float(min)
        elif "STATISTICS_MINIMUM" in metadata:
            self.min = float(metadata["STATISTICS_MINIMUM"])
        else:
            self.min = None #raise InputDataError("Minimum is not in image statistics!")

        if max is not None:
            self.max = float(max)
        elif "STATISTICS_MAXIMUM" in metadata:
            self.max = float(metadata["STATISTICS_MAXIMUM"])
        else:
            self.max = None #raise InputDataError("Maximum is not in image statistics!")

        if mean is not None:
            self.mean = float(mean)
        elif "STATISTICS_MEAN" in metadata:
            self.mean = float(metadata["STATISTICS_MEAN"])
        else:
            self.mean = None #raise InputDataError("Mean is not in image statistics!")

        if stdDev is not None:
            self.stdDev = float(stdDev)
        elif "STATISTICS_STDDEV" in metadata:
            self.stdDev = float(metadata["STATISTICS_STDDEV"])
        else:
            self.stdDev = None #raise InputDataError("Stddev is not in image statistics!")

        if colorInterpretation is not None:
            self.colorinterp = colorInterpretation
        else:
            self.colorinterp = metadata.get('colorinterp')

    @property
    def is_uint8(self):
        return self.type.lower() == 'byte'

    @property
    def statistics_valid(self):
        return self.min is not None \
            and self.max is not None \
            and self.mean is not None \
            and self.stdDev is not None

    def range_meanstd(self, std_width):
        """
        Returns range - tuple (min, max) for values scaling
        """
        if not self.statistics_valid:
            return None, None
        return max(self.min, (self.mean - self.stdDev * std_width)), \
               min(self.max, (self.mean + self.stdDev * std_width))


class CogBuilder:
    def __init__(self,
                 input_ds: Path,
                 cog_ds: Path,
                 workdir: Path,
                 channels: Optional[List[int]],
                 aoi: Mapping[str, Any],
                 compress: str = "WEBP",
                 std_width: float = 3.):
        """
        mask is a geometry object {"type": "Polygon", "coordinates":[[[...]]]}
        """
        gdal.UseExceptions()

        self.compress = compress
        self.workdir = workdir
        if channels is not None and len(channels) != 3:
            raise ValueError(f"Only 3 channels are supported, but found {len(channels)}")

        self.aoi = aoi

        logger.info(f"Building COG from {input_ds} to {cog_ds}. Parameters: " + \
                f"channels {channels if channels else 'not defined'}; " + \
                f"mask {'provided' if self.aoi else 'not provided'}")

        merc_ds = workdir/'merc.vrt'
        channel_opts = get_channel_opts(src=input_ds, std_width=std_width, channels=channels)
        self.reproject_to_3857(src=input_ds, dst=merc_ds)
        mask_opts = get_mask_opt(merc_ds)
        self.materialize_cog(src=workdir/merc_ds, dst=cog_ds, opts=" ".join((channel_opts, mask_opts)))

    def reproject_to_3857(self, src: Path, dst: Path):
        logger.info(f"Reprojecting { src } to { dst }")
        if self.aoi:
            features = {'type': 'FeatureCollection',
                        'features': [{'type': 'Feature', 'properties': {}, 'geometry': mapping(self.aoi)}]}

            cutline_file = str(self.workdir/'cutline.geojson')
            with open(cutline_file, 'w') as fh:
                json.dump(features, fh)
            logger.info(f"Cropping image using { cutline_file }")
        else:
            cutline_file = None

        gdal.Warp(str(dst), str(src), options=gdal.WarpOptions(dstSRS="EPSG:3857",
                                                     resampleAlg="bilinear",
                                                     dstAlpha=True,
                                                     cropToCutline=bool(self.aoi),
                                                     cutlineDSName=cutline_file))
        with rasterio.open(dst) as ds:
            x_res, y_res = ds.res
            if ds.width > MAXIMUM_DATASOURCE_SIZE or ds.height > MAXIMUM_DATASOURCE_SIZE:
                raise CogReprojectionError(f"Reprojected size ({ ds.width }, { ds.heig.ht })"
                                        f" is greater then { MAXIMUM_DATASOURCE_SIZE }")

        if x_res < MINIMUM_REJECTED_PIXEL_SIZE or y_res < MINIMUM_REJECTED_PIXEL_SIZE:
            raise CogReprojectionError(f"Reprojected pixel size xRes = {x_res} m/px, yRes = {y_res} m/px "
                                    f"is less than { MINIMUM_REJECTED_PIXEL_SIZE } m/px.")

        if x_res < MINIMUM_ALLOWED_PIXEL_SIZE or y_res < MINIMUM_ALLOWED_PIXEL_SIZE:
            logger.info(f"Pixel resolution xRes = {x_res} m/px, yRes = {y_res} m/px "
                        f"is less than {MINIMUM_ALLOWED_PIXEL_SIZE} m/px. "
                        f"Changing resolution to {MINIMUM_ALLOWED_PIXEL_SIZE} m/px")
            x_res = MINIMUM_ALLOWED_PIXEL_SIZE
            y_res = MINIMUM_ALLOWED_PIXEL_SIZE
            gdal.Warp(str(dst), str(src), options=gdal.WarpOptions(dstSRS="EPSG:3857",
                                                         resampleAlg="bilinear",
                                                         xRes=x_res,
                                                         yRes=y_res,
                                                         dstAlpha=True,
                                                         cropToCutline=bool(self.aoi),
                                                         cutlineDSName=cutline_file))

    def materialize_cog(self, src: Path, dst: Path, opts: str):
        logger.info(f"Materializing COG from { src } to { dst }")
        translate_opts = f"-of COG -ot Byte -co TILING_SCHEME=GoogleMapsCompatible" \
                         f" -co COMPRESS={self.compress} -co BIGTIFF=IF_SAFER -co NUM_THREADS=2 " \
                         f"{opts}"

        logger.info(f"GDAL option { translate_opts }")
        gdal.Translate(str(dst), str(src), options=translate_opts)

def get_channel_opts(src: str, std_width = 3, channels = None):

    try:
        info = gdal.Info(str(src), stats=True, format='json')
        logger.debug(f"Calculated GDALInfo {info}")
        bandstats = [BandStats(**band) for band in info['bands']]
    except RuntimeError as e:
        if "Failed to compute statistics, no valid pixels found in sampling" in str(e):
            logger.debug("Skipping statistics calculation, there are no valid pixels in image.")
            # This is when all the image is non-valid pixels, so no big deal if we fill all channels with the
            # data from the first one, it will be zeros anyway
            info = gdal.Info(str(src), stats=False, format='json')
            logger.debug(f"Calculated GDALInfo {info}")
            bandstats = [BandStats(**band) for band in info['bands']]
        else:
            raise e

    if len(bandstats) < 1:
        raise InputDataError("No bands found in input file!")
    if channels:
        options = _rgb_from_channels(bandstats=bandstats, channels=channels, std_width=std_width)
    # If channel positions are not set, we try to figure out how to prepare the image
    # If input channel has 1 or 2 channels, only the first channel is used, and copied into 3 output channels
    # If 3 and more are present, channels 1, 2, 3 are treated as RGB
    # If channel with COLORINTERP=ALPHA is present, we use it as nodata mask (option "-mask <bn>")
    # (in position 2 in case of 2-band image, or 4+ in case of multispectral)
    elif len(bandstats) <= 2:
        band = bandstats[0]
        options = "-b 1 -b 1 -b 1"
        if not band.is_uint8:
            if band.statistics_valid:
                m, M = band.range_meanstd(std_width=std_width)
                options += f"-scale {m} {M}"
            else:
                options += "-scale"
    else: # 3+ channels - make RGB
        options = _rgb_from_channels(bandstats=bandstats, channels=(1, 2, 3), std_width=std_width)
    return options

def _rgb_from_channels(bandstats, channels=(1,2,3), std_width=3):
    opts_list = []
    if len(bandstats) < max(channels):
        raise InputDataError(f"Input image has {len(bandstats)} bands, but bands {channels} are needed")
    # Channel positioins are explicitely set, so we use them
    # In this case, if the channel numbers do not correspond the expected, wo throw error
    for output_channel, input_channel in enumerate(channels, 1):
        band = bandstats[input_channel - 1]
        opts_list.append(f"-b {input_channel}")
        if not band.is_uint8:
            if band.statistics_valid:
                m, M = band.range_meanstd(std_width=std_width)
                opts_list.append(f"-scale_{output_channel} {m} {M}")
            else:
                opts_list.append(f"-scale_{output_channel}")
    options = " ".join(opts_list)
    return options

def get_mask_opt(src: str):
    """
    Adds first band with colorinterp=ALPHA as a mask band; if there is none - returns empty string
    """
    info = gdal.Info(str(src), stats=False, format='json')
    bandstats = [BandStats(**band) for band in info['bands']]
    for band in bandstats:
        if band.colorinterp.lower() == "alpha":
            maskopt = f" -mask {band.band}"
            return maskopt
    return ""
