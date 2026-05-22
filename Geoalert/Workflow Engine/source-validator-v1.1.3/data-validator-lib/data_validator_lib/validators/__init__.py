from .local import LocalValidator as local, LocalValidator as tiff, LocalValidator as tif
# several imports are needed so that local, tiff and tif source type are synonyms in API
from .sentinel_l2a import SentinelL2AValidator as sentinel_l2a
from .quadkey import QuadkeyValidator as quadkey
from .tms import TMSValidator as tms
from .wms import WMSValidator as wms
from .xyz import XYZValidator as xyz
from .basemap import BasemapValidator as basemap
from .head_imagery import HeadImageryValidator as head_imagery

# Developers must ensure that this list correspons to source_type param in dataloader,
# that is for every source type there is a key
# if a new validator is added, we must list it here to import and call it by name
__all__ = [local, tif, tiff, sentinel_l2a, quadkey, tms, wms, xyz, basemap]