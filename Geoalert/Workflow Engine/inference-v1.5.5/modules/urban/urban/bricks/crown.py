from typing import Tuple, Literal
import numpy as np
from ..base import Brick
from ..functional import io
from ..functional.raster_ops.res_in_meters import get_resolution_in_meters
from ..functional.utils.to_fc_processor import ToFeatureCollectionProcessor
from ..functional.crown_delineation import get_crown_delineation_fn
from ..functional.regression import crop_by_mask
from pydantic import Field


class CrownDelineation(Brick):
    """
    Performs crown delineation using the `dalponteCIRC` or `watershed` algorithm

    Parameters
    ----------
    input_raster : str
        Input raster of height values (e.g., CHM) (in meters)
    output_vector : str
        Output vector with crown features
    sample_size : Tuple[int, int]
        Sample size
    bounds : int
        Bounds
    src_nodata : int
        Source nodata
    nodata_mask_mode : bool
        If true, mask put nodata values
    simplify : float
        Simplification parameter
    area_threshold : float
        Features with area less than threshold will be deleted
    score_threshold : float
        Features with confidence score less than threshold will be deleted
    verbose : bool
        If true, shows progress bar
    algorithm : str
        Algorithm to use - 'dalponteCIRC' or 'watershed'. Default is 'dalponteCIRC' which gives circular crowns
    ws_smooth : int
        Window size for smoothing input_raster (in pixels)
    ws_local_maxima : int
        Window size to detect the local maxima (in meters)
    min_height_local_maxima : int
        Threshold below which a pixel or a point cannot be a local maxima (in meters)
    min_height_for_tree : int
        Threshold below which a pixel or a point cannot be a part of a crown (in meters)
    th_seed : float
        A pixel is added to a region if its height is greater than the tree height multiplied by this value.
         It should be between 0 and 1
    th_crown : float
        A pixel is added to a region if its height is greater than the current mean height of the region
        multiplied by this value. It should be between 0 and 1
    max_crown : int
        Maximum radius of tree crown (in meters)
    min_pixel_number : int
        Minimum window size or diameter of tree crown (in pixels)
    max_pixel_number : int
        Maximum window size or diameter of tree crown (in pixels)
    """
    input_raster: str
    output_vector: str
    sample_size: Tuple[int, int] = Field((512, 512))
    bounds: int = Field(0)
    src_nodata: int = Field(0)
    nodata_mask_mode: bool = Field(False)
    simplify: float = Field(0)
    area_threshold: float = Field(0)
    score_threshold: float = Field(0)
    verbose: bool = Field(False)
    algorithm: Literal['dalponteCIRC', 'watershed'] = Field('dalponteCIRC')
    ws_smooth: int = Field(5)  # in pixels
    ws_local_maxima: int = Field(5)  # in meters
    min_height_local_maxima: int = Field(10)  # in meters
    min_height_for_tree: int = Field(2)  # in meters
    th_seed: float = Field(0.7)
    th_crown: float = Field(0.55)
    max_crown: int = Field(5)  # in meters
    min_pixel_number: int = Field(5)
    max_pixel_number: int = Field(100)
    _predictor: ToFeatureCollectionProcessor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._predictor = ToFeatureCollectionProcessor(input_channels=[self.input_raster],
                                                       processing_fn=lambda x: (np.zeros(0), np.zeros(0)),
                                                       sample_size=self.sample_size,
                                                       bound=self.bounds,
                                                       src_nodata=self.src_nodata,
                                                       nodata_mask_mode=self.nodata_mask_mode,
                                                       simplify=self.simplify,
                                                       area_threshold=self.area_threshold,
                                                       score_threshold=self.score_threshold,
                                                       properties_type='dict',
                                                       verbose=self.verbose)

    def __call__(self, path):
        bc = io.read_bc(path, [self.input_raster])
        resolution = get_resolution_in_meters(bc.crs, bc.res[0])

        self.max_pixel_number = min(self.max_pixel_number, bc.width, bc.height)
        if bc.width < self.min_pixel_number or bc.height < self.min_pixel_number:
            raise ValueError(f'Input raster has at least one dimension < {self.min_pixel_number}: width = {bc.width}, height = {bc.height}')

        self._predictor.processing_fn = get_crown_delineation_fn(algorithm=self.algorithm,
                                                                resolution=resolution,
                                                                ws_smooth=self.ws_smooth,
                                                                ws_local_maxima=self.ws_local_maxima,
                                                                min_height_local_maxima=self.min_height_local_maxima,
                                                                min_height_for_tree=self.min_height_for_tree,
                                                                th_seed=self.th_seed,
                                                                th_crown=self.th_crown,
                                                                max_crown=self.max_crown,
                                                                min_pixel_number=self.min_pixel_number,
                                                                max_pixel_number=self.max_pixel_number)
        fc = self._predictor.process(bc)
        io.save_fc(fc, path=path, name=self.output_vector)


class CrownMaxHeight(Brick):
    """ Extract the maximum height of a tree crown
    Args:
        input_raster: str
            Input raster of height values (e.g., CHM) (in meters)
        input_vector: str
            Input vector of tree crowns
        output_vector: str
            Output vector of tree crowns with `max_height`
        sample_size:
            Sample size (in pixels)
        min_height_for_tree: float
            Minimum height of tree (in meters)
    """
    input_raster: str
    input_vector: str
    output_vector: str = Field(default=None)
    sample_size: Tuple[int, int] = Field((200, 200))
    min_height_for_tree: float = 4

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output_vector = self.output_vector or self.input_vector

    def __call__(self, path):
        bc = io.read_bc(path, [self.input_raster])
        fc = io.read_fc(path, self.input_vector)
        if fc.crs != bc.crs:
            fc = fc.to_crs(bc.crs)
        to_drop = list()
        for f_idx in range(len(fc)):
            mask = fc[f_idx, 'geometry']
            sample = crop_by_mask(bc, mask, sample_size=self.sample_size, padding_value=0)
            max_height = int(round(sample.max().item()))
            if max_height < self.min_height_for_tree:
                to_drop.append(f_idx)
            else:
                fc[f_idx, 'max_height'] = max_height

        fc.drop(to_drop, inplace=True)
        io.save_fc(fc, path, self.output_vector)
