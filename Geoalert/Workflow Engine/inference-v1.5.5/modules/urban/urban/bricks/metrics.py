import os
import shapely
from ..base import Brick
from ..functional import io, metrics
from gpdadapter import FeatureCollection
from typing import Optional, Sequence
from pydantic import Field


class VectorMetrics(Brick):  # TODO: deprecated
    pd_vector: str
    gt_vector: str
    output_vector: str
    metrics_list: Optional[Sequence] = Field(None)
    aoi_vector: Optional[str] = Field(None)
    aoi_band: Optional[str] = Field(None)
    crs: str = Field('EPSG:3857')
    verbose: bool = Field(False)

    def __call__(self, path):
        # Load pd and gt
        pd_fc = io.read_fc(path, self.pd_vector)
        pd_fc = pd_fc.to_crs(self.crs)

        gt_fc = io.read_fc(path, self.gt_vector)
        gt_fc = gt_fc.to_crs(self.crs)

        # Calculate aoi bound
        aoi_feature = None
        # If vector AOI provided
        if self.aoi_vector is not None and os.path.exists(f'{path}/{self.aoi_vector}.geojson'):
            aoi_fc = io.read_fc(path, self.aoi_vector)
            aoi_fc = aoi_fc.to_crs(self.crs)

            if len(aoi_fc) != 1:
                raise Exception('Only one AOI is acceptable')

            aoi_feature = aoi_fc[0]
            # if not isinstance(aoi_feature.shape, shapely.geometry.polygon.Polygon):
            #     raise Exception('AOI only could be Polygon')
        # If raster AOI provided
        elif self.aoi_band is not None and os.path.exists(f'{path}/{self.aoi_band}.tif'):
            aoi_bc = io.read_bc(path, [self.aoi_band, ])
            x_min, y_min, x_max,  y_max = aoi_bc.bounds
            aoi_geom = shapely.geometry.Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
            #aoi_feature = Feature(aoi_geom, crs=self.crs, properties={})
            #aoi_feature.reproject(self.crs)
        # If AOI not provided, and it should be calculated from gt-pd extents and bbox of both of them
        else:
            # GT limits
            x_min, y_min, x_max,  y_max = gt_fc.index.bounds
            # PD limits
            f_x_min, f_y_min, f_x_max, f_y_max = pd_fc.index.bounds
            # Total limits
            if f_x_min < x_min:
                x_min = f_x_min
            if f_y_min < y_min:
                y_min = f_y_min
            if f_x_max > x_max:
                x_max = f_x_max
            if f_y_max > y_max:
                y_max = f_y_max
            # Generate AOI
            aoi_geom = shapely.geometry.Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
            #aoi_feature = Feature(aoi_geom, crs=self.crs, properties={})

        # Clip gt-vec to aoi. Two stage clipping to make it fast
        new_gt_list, new_pd_list = [], []
        for f in gt_fc.intersection(aoi_feature):
            intersection = f.shape.intersection(aoi_feature.shape)
            #new_gt_list.append(Feature(intersection, crs=self.crs, properties=f.properties))
        gt_fc = FeatureCollection(new_gt_list, self.crs)

        for f in pd_fc.intersection(aoi_feature):
            intersection = f.shape.intersection(aoi_feature.shape)
            #new_pd_list.append(Feature(intersection, crs=self.crs, properties=f.properties))
        pd_fc = FeatureCollection(new_pd_list, self.crs)

        metrics_calculator = metrics.VectorMetricsCalculator(pd_fc=pd_fc, gt_fc=gt_fc)
        if self.metrics_list is None:
            metrics_list = metrics_calculator.available_metrics()
        else:
            metrics_list = self.metrics_list

        property_dict = {}
        for metric_name in metrics_list:
            metric_value = metrics_calculator.by_name(metric_name)
            property_dict[metric_name] = metric_value

        aoi_feature.properties = property_dict
        fc = FeatureCollection([aoi_feature, ], self.crs)
        io.save_fc(fc, path, self.output_vector)
