from ..base import Brick
from ..functional.postprocessing.shapely_ext.bbox import BBox
from ..functional import io
from ..functional.osm import load_osm_buildings, load_osm_roads, load_osm_landuse

# TODO: inherit OSM and ZKH from a single patent class
class _LoadOSMLike(Brick):
    input: str
    output: str

    def __call__(self, path):
        fc = io.read_fc(path, self.input)

        bbox = BBox(*fc.total_bounds).swne()

        fc_osm = self.load(bbox)
        io.save_fc(fc_osm, path, self.output)

    def load(self, bbox):
        raise NotImplementedError


class LoadOSMBuildingsLike(_LoadOSMLike):
    """Load Buildings from OSM with respect to bounds of given input GeoJSON collection

    Args:
        input: filename of GeoJSON collection to define bounds
        output: filename of GeoJSON collection to put buildings features in
    """

    def load(self, bbox):
        return load_osm_buildings(bbox)


class LoadOSMRoadsLike(_LoadOSMLike):
    """Load Roads (Highway) from OSM with respect to bounds of given input GeoJSON collection

    Args:
        input: filename of GeoJSON collection to define bounds
        output: filename of GeoJSON collection to put buildings features in
    """

    def load(self, bbox):
        return load_osm_roads(bbox)


class LoadOSMLanduseLike(_LoadOSMLike):
    """Load Landuse (landuse) from OSM with respect to bounds of given input GeoJSON collection

    Args:
        input: filename of GeoJSON collection to define bounds
        output: filename of GeoJSON collection to put buildings features in
    """
    def load(self, bbox):
        return load_osm_landuse(bbox)
