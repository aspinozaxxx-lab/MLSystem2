import os
import shutil
from typing import Optional, Union
from osgeo import gdal
from pathlib import Path
from loguru import logger
import sys

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
        metadata = metadata['']
        self.band = int(band)
        self.type = type
        if min is not None:
            self.min = float(min)
        elif "STATISTICS_MINIMUM" in metadata:
            self.min = float(metadata["STATISTICS_MINIMUM"])
        else:
            raise ValueError("Minimum is not in image statistics!")

        if max is not None:
            self.max = float(max)
        elif "STATISTICS_MAXIMUM" in metadata:
            self.max = float(metadata["STATISTICS_MAXIMUM"])
        else:
            raise ValueError("Maximum is not in image statistics!")

        if mean is not None:
            self.mean = float(mean)
        elif "STATISTICS_MEAN" in metadata:
            self.mean = float(metadata["STATISTICS_MEAN"])
        else:
            raise ValueError("Mean is not in image statistics!")

        if stdDev is not None:
            self.stdDev = float(stdDev)
        elif "STATISTICS_STDDEV" in metadata:
            self.stdDev = float(metadata["STATISTICS_STDDEV"])
        else:
            raise ValueError("Stddev is not in image statistics!")

        if colorInterpretation is not None:
            self.colorinterp = colorInterpretation
        else:
            self.colorinterp = metadata.get('colorinterp')

    @property
    def is_uint8(self):
        return self.type.lower() == 'byte'

    def range_meanstd(self, std_width):
        """
        Returns range - tuple (min, max) for values scaling
        """
        return max(self.min, (self.mean - self.stdDev * std_width)), \
               min(self.max, (self.mean + self.stdDev * std_width))


def get_channel_opts(band, std_width: float = 2):
    m, M = band.range_meanstd(std_width=std_width)
    if m < M:
        options = f"-b 1 -scale_1 {m} {M}"
    else:
        # stddev==0 or std_width<=0
        options = "-b 1 -scale_1"
    return options


def linear_brightness_scale(src: Union[Path, str],
                            dst: Union[Path, str],
                            std_width: float = 2):
    """
    Single-channel file only
    """
    if sys.platform.startswith('win'):  # Windows workaround TODO: make platform-independent (use rasterio?)
        shutil.copy(src, dst)
        return

    gdal.UseExceptions()
    try:
        info = gdal.Info(str(src), stats=True, format='json')
    except RuntimeError as e:
        if "Failed to compute statistics, no valid pixels found in sampling" in str(e):
            # Known problem is whole image is NODATA
            logger.debug("Skipping statistics calculation, there are no valid pixels in image.")
            gdal.Translate(str(dst), str(src), options=f"-ot Byte -co BIGTIFF=IF_SAFER")
            return
        else:
            raise e
    bandstats = BandStats(**info['bands'][0])
    if bandstats.is_uint8:
        # 8bit files do not require brightness adjustment, but we need new files to continue processing
        # todo: add option "preprocess_8bit" to turn it on/off
        logger.debug(f"Skipping {src} conversion as it is already 8bit. Creating {dst} as symlink")
        os.symlink(src,dst)
        return

    channel_opts = get_channel_opts(bandstats, std_width=std_width)
    translate_opts = f"-ot Byte -co BIGTIFF=IF_SAFER {channel_opts}"
    logger.info(f"GDAL transform {src} to {dst} with options { translate_opts }")
    gdal.Translate(str(dst), str(src), options=translate_opts)
