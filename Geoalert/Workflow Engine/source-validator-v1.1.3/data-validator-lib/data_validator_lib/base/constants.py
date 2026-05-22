import re
# for basemap validator

URL_KEY = 'url'
ZOOM_KEY = 'zoom'
LOWEST_ZOOM = 0
HIGHEST_ZOOM = 23

# for local validator

METADATA_KEY = 'profile'
S3_LINK_KEY = 'url'
REQUIRED_METADATA_KEYS = {'crs', 'transform', 'dtype', 'count'}
RES_TOLERANCE = 0.1

# sentinel validator
INPUT_STRING_KEY = 'url'
AOI_KEY = 'aoi'

BASEMAPS_BLACKLIST_PATTERNS = [re.compile(pattern) for pattern in [
    r'tile\.opentopomap\.org',
    r'\.opentopomap\.ru',
    r'tiles\.topomap\.org',
    r'\.wikimapia\.org',
    r'maps\.2gis\.com',
    # common osm links
    r'tile\.openstreetmap\.',
    r'tile\.osm\.'
    r'tiles\.wmflabs\.org',
    r'tile\.thunderforest\.com',
    r'google\.com/vt/lyrs=[^s]',  # satellite google link contains "lyrs=s"
    # non-satellite yandex maps
    r'core-renderer-tiles\.maps\.yandex\.net',
    # maptiler layers: not all available, but obvious global
    r'api\.maptiler\.com/maps/(street|v3-4326|v3|v3-openmaptiles|terrain-rgb-v2|hillshades|cadastre)/',
    # bing maps: aerial/satellite contains &it=A, and the map identificator starts with &it=G
    r'tiles\.virtualearth\.net.*&it=G']
                               ]
