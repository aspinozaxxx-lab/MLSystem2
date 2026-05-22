import shapely.geometry as sg
from .. import shapely_ext as se
from typing import Union

PolygonType = Union[sg.Polygon, sg.MultiPolygon]


class SimplifiedGeometry:

    def __init__(
            self,
            origin_geometry: PolygonType,
            simple_geometry: PolygonType,
            simple_geometry_type: str,
            iou: float = None,
            hausdorff: float = None,
    ):
        self.origin_geometry = origin_geometry
        self.simple_geometry = simple_geometry
        self.simple_geometry_type = simple_geometry_type
        self.iou = iou if iou is not None else self._iou()
        self.hausdorff = hausdorff if hausdorff is not None else self._hausdorff()

    def _iou(self):
        return se.intersection_over_union(self.origin_geometry, self.simple_geometry, ignore_errors=True)

    def _hausdorff(self):
        return self.origin_geometry.hausdorff_distance(self.simple_geometry)
