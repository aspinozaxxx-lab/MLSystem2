import os.path
import numpy as np
from geopandas import GeoSeries
from gpdadapter import FeatureCollection, query_to_dict
from ..base.brick import Brick
import rasterio
import rasterio.features
from ..functional import split_semantic_by_lines, io
from pydantic import Field
from typing import Literal, Any, Optional


class SplitByLines(Brick):
    """ Splits semantic mask by 'lines' mask (e.g. split hipped roofs into segments by ridge lines)
    Args:
        semantic_band: band with semantic mask
        lines_band: band with lines
    Returns:
        saves FeatureCollection with split polygons
    """
    semantic_band: str
    lines_band: str
    output_fc: str

    def __call__(self, path: str):
        with rasterio.open(os.path.join(path, self.semantic_band)+'.tif') as d:
            semantic_raster = d.read(1)
            transform = d.transform
            crs = d.crs
            f = rasterio.features.dataset_features(d, bidx=1, geographic=False)
            semantic = FeatureCollection.from_features(f, crs).simplify(1)

        with rasterio.open(os.path.join(path, self.lines_band)+'.tif') as d:
            pred = d.read(1)
            if not transform == d.transform and crs == d.crs:
                raise ValueError('Bands with different extents')

        pred_lines = GeoSeries(split_semantic_by_lines.skeletonize_pred(
            pred, semantic_raster)).affine_transform(
            [transform.a, transform.b, transform.d,
             transform.e, transform.xoff, transform.yoff]).to_list()

        pred_lines = FeatureCollection(pred_lines, crs=crs)  # split lines

        pred_lines.append(FeatureCollection(semantic.map(lambda x: x.exterior, column='geometry').to_list(),
                                            crs=crs))

        polygons = split_semantic_by_lines.polygonize_split(pred_lines)
        polygons = split_semantic_by_lines.dissolve_small_polygons(polygons)
        io.save_fc(polygons, path, self.output_fc)


class SplitByLinesWatershed(Brick):
    """ Splits semantic mask by 'lines' mask (e.g. split hipped roofs into segments by ridge lines)
    Args:
        semantic_band: band with semantic mask
        lines_band: band with lines
    Returns:
        saves band with split polygons
    """
    # TODO: windowed processing via Predictor
    semantic_band: str
    lines_band: str
    output_band: str
    threshold: int = Field(0)
    semantic_erosion: int = Field(0)
    lines_dilation: int = Field(0)
    min_size: int = Field(32)
    mode: Literal['semantic_and_lines', 'markers_and_lines'] = Field('semantic_and_lines')

    def __call__(self, path: str):
        with rasterio.open(os.path.join(path, self.semantic_band)+'.tif') as d:
            semantic= d.read(1)
            transform = d.transform
            crs = d.crs
            profile = d.profile

        with rasterio.open(os.path.join(path, self.lines_band)+'.tif') as d:
            lines = d.read(1)
            if not transform == d.transform and crs == d.crs:
                raise ValueError('Bands with different extents')

        if self.mode == 'markers_and_lines':
            semantic = np.logical_or(semantic, lines).astype(np.uint8)

        result_mask = split_semantic_by_lines.watershed_split(
            semantic, lines, self.threshold,
            self.semantic_erosion, self.lines_dilation, self.min_size)

        with rasterio.open(os.path.join(path, self.output_band)+'.tif', 'w', **profile) as d:
            d.write(result_mask, 1)


class ClassifyBySegments(Brick):
    """ Classify roof type (flat vs gable) by the number of segments
    Args:
        semantic_fc: semantic roofs
        segments_fc: roof segments
        tag: class tag
        value_if_flat:
        value_if_gable:
    Returns:
        saves FeatureCollection same as semantic_fc with assigned class
    """
    # TODO: windowed processing via Predictor
    semantic_fc: str
    segments_fc: str
    output_fc: Optional[str] = None
    tag: str = Field('gable_prob')
    value_if_flat: Any = Field(0)
    value_if_gable: Any = Field(1)

    def __call__(self, path: str):
        semantic_fc = io.read_fc(path, self.semantic_fc)
        segments_fc = io.read_fc(path, self.segments_fc, crs=semantic_fc.crs)
        indexes = query_to_dict(semantic_fc.query(segments_fc))
        semantic_fc[:, self.tag] = self.value_if_flat
        for idx in range(len(semantic_fc)):
            if len(indexes.pop(idx, [])) > 1:
                semantic_fc[idx, self.tag] = self.value_if_gable
        io.save_fc(semantic_fc, path, self.output_fc or self.semantic_fc)
