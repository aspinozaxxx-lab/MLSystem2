import os
import numpy as np
from typing import Tuple, List, Optional, Union, Sequence, Literal
from loguru import logger
import rasterio
import rasterio.features
from aeronet_raster.collectionprocessor import CollectionProcessor as Predictor
from aeronet_raster.utils.utils import parse_directory
import shapely
from ..base import Brick
from gpdadapter import FeatureCollection
from ..functional import io
from ..functional.raster_ops.split import split
from ..functional.raster_ops.merge import merge
from ..functional.raster_ops.brightness_adjustment import linear_brightness_scale
from ..functional.raster_ops import get_multi_threshold_fn, get_morphology_fn, get_binary_fn, zonal_stats
from ..functional.polygonize import polygonize
from ..functional.utils.mathutils import normalize
from functools import partial
from pydantic import Field, ConfigDict

# TODO: inherit all bricks with Predictor from a single parent

# ------------------------------------------------------------------------------------------------
# Raster operations (imagery)
# ------------------------------------------------------------------------------------------------

class SplitRaster(Brick):
    """Split single raster file to multiple files with one only band each

    Args:
        input: name of rater file (without extension)
        output: names for output bands (without extension)
        input_ext: input file extension (e.g. 'tif', 'tiff', 'TIFF', etc.)
        allow_singleband: allow to copy singleband image to produce multiband output
    """
    input: str
    output: Union[Tuple[str], List[str]]
    input_ext: str = Field('tif')
    allow_singleband: bool = Field(True)
    window_size: int = Field(10000)

    def __call__(self, path):
        src_path = os.path.join(path, '{name}.{ext}'.format(name=self.input, ext=self.input_ext))
        dst_path = path
        split(src_path, dst_path, self.output, allow_singleband=self.allow_singleband, window_size=self.window_size)


class MergeRaster(Brick):
    input: str
    output: Union[Tuple[str], List[str]]
    input_ext: str = Field('tif')
    window_size: int = Field(10000)

    def __call__(self, path):
        src_paths = [os.path.join(path, f'{name}.{self.input_ext}') for name in self.input]
        dst_path = os.path.join(path, f'{self.output}.{self.input_ext}')
        merge(src_paths, dst_path, self.input, window_size=self.window_size)


class BrightnessNormalization(Brick):
    """
    The most basic brick for scaling the data range. Treats every channel separately,
    uses single method with only std_width param

    Normalizes the brightness of input channels with linear scaling:
    out_value = (input_value - input_min)*(out_max - out_min)/(input_max - input_min) + out_min
    where input_max and input_min values are calculated as `mean +- std_width*stdDev`

    As a result, the image is transformed to 8 bit, enhancing contrast and brightness
    8bit images are skipped

    Args:
        input: label for the input channels which are normalized
        output: corresponding output labels, equal number to the input
        std_width: parameter for meanstd method
        verbose: if `True` tqdm progress bar is used
    """
    input: Sequence[str]
    output: Sequence[str]
    input_ext: str = Field('tif')
    std_width: int = Field(3)
    verbose: bool = Field(False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.input) != len(self.output):
            raise ValueError(f'Number of out channels {len(self.output)} must be equal to '
                             f'the number of input channels {len(self.input)}')
        if set(self.input).intersection(set(self.output)):
            raise ValueError(f"Input and output must not have the same names, "
                             f"got repeating {set(self.input).intersection(set(self.output))}")

    def __call__(self, path):
        input_files = parse_directory(path, self.input)
        output_files = [os.path.join(path, f'{name}.tif') for name in self.output]
        for src, dst in zip(input_files, output_files):
            linear_brightness_scale(src, dst, self.std_width)


class RoundRaster(Brick):
    """Rounds the raster values to the nearest integer"""
    input_masks: List[str]
    out_masks: List[str]
    dtype: str = Field('uint8')
    sample_size: Tuple[int, int] = Field((10000, 10000))
    bound: int = Field(0)
    verbose: bool = Field(False)
    _predictor: Predictor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.input_masks) != len(self.out_masks):
            raise ValueError('Number of out masks must be equal to the number of input masks')

        self._predictor = Predictor(
            input_channels=self.input_masks,
            output_labels=self.out_masks,
            processing_fn=lambda x: x.round().clip(0, np.iinfo(self.dtype).max).astype(self.dtype),
            # clip - to prevent 256->0
            sample_size=self.sample_size,
            bound=self.bound,
            verbose=self.verbose,
            dtype=self.dtype)

    def __call__(self, path):
        bc = io.read_bc(path, self.input_masks)
        out_bc = self._predictor.process(bc, path)


class AddConstant(Brick):
    """add constant to the raster values"""
    input_rasters: List[str]
    output_rasters: List[str]
    constant: float
    sample_size: Tuple[int, int] = Field((10000, 10000))
    verbose: bool = Field(False)
    _predictor: Predictor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.input_rasters) != len(self.output_rasters):
            raise ValueError('Number of out masks must be equal to the number of input masks')

        self._predictor = Predictor(
            input_channels=self.input_rasters,
            output_labels=self.output_rasters,
            processing_fn=lambda x: x + self.constant,
            sample_size=self.sample_size,
            bound=0,
            verbose=self.verbose
        )


    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        dtype = bc._bands[0].dtype
        self._predictor.dst_dtype = dtype
        if np.issubdtype(dtype, np.integer):
            min_value = np.iinfo(dtype).min
            max_value = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            min_value = np.finfo(dtype).min
            max_value = np.finfo(dtype).max
        else:
            raise ValueError(f'Dtype {dtype} is not supported, only integer and float dtypes are supported')

        self._predictor.processing_fn = lambda x: np.clip(x + self.constant, min_value, max_value).astype(dtype)
        out_bc = self._predictor.process(bc, path)


class ReplacePixelValue(Brick):
    """replace pixel value by another value"""
    input_rasters: List[str]
    output_rasters: List[str]
    input_pixels: List[float]
    output_pixels: List[float]
    sample_size: Tuple[int, int] = Field((10000, 10000))
    verbose: bool = Field(False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.input_rasters) != len(self.output_rasters):
            raise ValueError('Number of out masks must be equal to the number of input masks')

        if len(self.input_pixels) != len(self.output_pixels):
            raise ValueError('Number of input and output pixel values must be equal')

        if len(self.input_pixels) != len(self.input_rasters):
            raise ValueError('Number of input pixel values must be equal to the number of input rasters')

    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        dtype = bc._bands[0].dtype
        if np.issubdtype(dtype, np.integer):
            min_value = np.iinfo(dtype).min
            max_value = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            min_value = np.finfo(dtype).min
            max_value = np.finfo(dtype).max
        else:
            raise ValueError(f'Dtype {dtype} is not supported, only integer and float dtypes are supported')
        for output_pixel in self.output_pixels:
            if output_pixel < min_value or output_pixel > max_value:
                raise ValueError(f'Output pixel value `{output_pixel}` in `{self.output_pixels}` '
                                 f'is out of range `[{min_value}, {max_value}]`')

        def replace_pixel(x):
            for i in range(x.shape[0]):
                x[i] = np.where(x[i] == self.input_pixels[i], self.output_pixels[i], x[i]).astype(dtype)
            return x

        predictor = Predictor(
            input_channels=self.input_rasters,
            output_labels=self.output_rasters,
            processing_fn=replace_pixel,
            sample_size=self.sample_size,
            bound=0,
            verbose=self.verbose,
            dst_dtype=dtype
        )

        out_bc = predictor.process(bc, path)


# ------------------------------------------------------------------------------------------------
# Raster operations (masks)
# ------------------------------------------------------------------------------------------------

class ApplyMask(Brick):
    """Applies parent mask to all the child masks - and/or.
    
    Args:
        parent_mask: label for the parent mask which will be applied to the child masks
        child_masks: label for the input masks which are modified
        out_masks: corresponding output labels, equal number to the child_masks
        mask_operation: a symbolic name for the operation. 'and', 'or', 'mask' are currently allowed
            and: return min(child, parent)
            or: return max(child, parent)
            mask: return child where parent is not None, 0 otherwise
        n_workers: number of treads for data processing
        sample_size: size of pieces for processing
        verbose: if `True` tqdm progress bar is used
    """
    parent_mask: str
    child_masks: List[str]
    out_masks: List[str]
    mask_operation: Literal["and", "or", "mask"] = Field('and')
    reverse_parent: bool = Field(False)
    n_workers: int = Field(1)  # TODO: deprecate
    sample_size: Tuple[int, int] = Field((10000, 10000))
    verbose: bool = Field(False)
    _predictor: Predictor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # define mask operation as a predictor
        self._predictor = Predictor(
            input_channels=self.child_masks + [self.parent_mask],
            output_labels=self.out_masks,
            processing_fn=get_binary_fn(self.mask_operation, self.reverse_parent),
            sample_size=self.sample_size,
            bound=0,
            verbose=self.verbose,
            dtype='uint8'
        )


    def __call__(self, path):
        child_bc = io.read_bc(path, self.child_masks)
        dtype = child_bc.bands[0].dtype
        self._predictor.dst_dtype = dtype
        # TODO: reprojection to a different brick
        # The bands can be from different models and hence have different resolution
        # We reproject the parent mask to the child masks to make it faster (one reprojection instead of multiple)
        # and also because the child masks logically are the target masks and the parent one is a modifier,
        # so it would be better to not change the target ones
        parent = io.read_bc(path, [self.parent_mask])[0]
        if not parent.same(child_bc):
            parent = parent.reproject_to(child_bc, interpolation='nearest')
        child_bc.append(parent)
        out_bc = self._predictor.process(child_bc, path)


class MultiThresholding(Brick):
    """Makes a set of masks from a 2-dimensional image based on the given thresholds,
    including image < min and image > max, so that the output contains number of thresholds masks + 1

    Args:
        input_raster: a label for the input raster channel
        thresholds: a list of the thresholds
        out_masks: a list of the output mask labels, must be longer than `thresholds` by one
        strict_more: if True, the conditions for the mask layer looks like T_i < input_raster <= T_(i+1),
            where T_i, T_(i+1) are two thresholds. Otherwise, the same condition is T_i <= input_raster < T_(i+1),
        sample_size: size of pieces for processing
        verbose: if `True` tqdm progress bar is used
    """
    input_raster: str
    thresholds: List[float]
    out_masks: Optional[List[str]] = Field(None)
    strict_more: bool = Field(False)
    sample_size: Tuple[int, int] = Field((1024, 1024))
    verbose: bool = Field(False)
    _predictor: Predictor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.thresholds) + 1 != len(self.out_masks):
            raise ValueError('Number of out masks must be equal to the number of thresholds plus 1')

        # define mask operation as a predictor
        self._predictor = Predictor(
            input_channels=[self.input_raster],
            output_labels=self.out_masks,
            processing_fn=get_multi_threshold_fn(self.thresholds, self.strict_more),
            sample_size=self.sample_size,
            bound=0,
            verbose=self.verbose,
            dtype='uint8',
        )

    def __call__(self, path):
        input_bc = io.read_bc(path, [self.input_raster])
        out_bc = self._predictor.process(input_bc, path)


class MaskMorphology(Brick):
    """Makes a morphology operation on the binary mask, possible operations are any unary 
    operations from skimage.morphology. The binary mask is treated as bool (necessary for 
    some operations to work correctly) and output is cast to uint8 for saving
    See https://scikit-image.org/docs/dev/api/skimage.morphology

    Args:
        input_masks: label for the input masks which are modified
        out_masks: corresponding output labels, equal number to the child_masks
        mask_operation: a name of a unary operation from skimage.morphology. Possible variants:
            - binary_opening()
            - binary_closing()
            - binary_erosion()
            - binary_dilation()
            - remove_small_objects(min_size: int=64, connectivity: int=1)
            - remove_small_holes(area_threshold: int=64, connectivity: int=1))
            - skeletonize()
            - convex_hull_object(connectivity: int=2)
            - etc...
        selem: symbolic name of a structuring element for morphology operation, from skimage.morphology.selem.
                allowed options are 'disk', 'square', 'diamond', 'star'
        selem_size: size parameter of selem, means size for square, and radius for every other type.
        sample_size:  size of pieces for processing
        bound: size of tile intersection.
        verbose: if `True` tqdm progress bar is used
        **kwargs: keyword arguments for mask_operation.
            keyword arguments and their default values are described in mask_operation section.
    """

    input_masks: List[str]
    out_masks: List[str]
    mask_operation: Literal[
        'binary_opening',
        'binary_closing',
        'binary_erosion',
        'binary_dilation',
        'remove_small_objects',
        'remove_small_holes',
        'skeletonize',
        'convex_hull_object'] = Field('binary_erosion')
    selem: Optional[str] = Field(None)
    selem_size: int = Field(1)
    sample_size: Tuple[int, int] = Field((1024, 1024))
    bound: int = Field(1)
    verbose: bool = Field(False)
    _predictor = None
    _morphology_kwargs: dict = None

    model_config = ConfigDict(extra = "allow")

    def model_post_init(self, __context):
        if 'n_workers' in self.model_extra:
            logger.warning('n_workers is deprecated!')
            self.model_extra.pop('n_workers')
        self._morphology_kwargs = self.model_extra if self.model_extra else dict()
        if len(self.input_masks) != len(self.out_masks):
            raise ValueError('Number of out masks must be equal to the number of input masks')

        # define mask operation as a predictor
        self._predictor = Predictor(
            input_channels=self.input_masks,
            output_labels=self.out_masks,
            processing_fn=get_morphology_fn(self.mask_operation, self.selem, self.selem_size, **self._morphology_kwargs),
            sample_size=self.sample_size,
            bound=self.bound,
            verbose=self.verbose,
            dtype='uint8'
        )

    def __call__(self, path):
        bc = io.read_bc(path, self.input_masks)
        # The bands can be from different models and hence have different resolution
        # We reproject the parent mask to the child masks to make it faster (one reprojection instead of multiple)
        # and also because the child masks logically are the target masks and the parent one is a modifier,
        # so it would be better to not change the target ones
        out_bc = self._predictor.process(bc, path)

    def get_config(self):
        config = super().get_config()
        config.update(self._morphology_kwargs)
        return config

class VectorizeMasks(Brick):
    """Vectorize the binary raster masks (convert binary mask to polygons)
    
    Args:
        input_rasters: names of the raster masks to be vectorized
        output_fcs: names of output feature collections. If None same names as input rasters used
        value_property_name: property name to write threshold group value into
        normalize_value_property_to: normalize values written into threshold_property from (0, 255) to this range
                                     (e.g. if you want it to look like confidence from 0 to 100)
    """

    input_rasters: Sequence[str]
    output_fcs: Optional[Sequence[str]] = Field(None)
    value_property_name: Optional[str] = Field(None)
    normalize_value_property_to: Optional[Tuple[float, float]] = Field(None)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output_fcs = self.output_fcs or self.input_rasters

    def __call__(self, path):
        if self.value_property_name:
            for raster, vector in zip(self.input_rasters, self.output_fcs):
                with rasterio.open(os.path.join(path, raster)+'.tif') as d:
                    fc = FeatureCollection.from_features(
                        rasterio.features.dataset_features(d, bidx=1, geographic=False), d.crs)
                    if not fc.empty:
                        fc[:, self.value_property_name] = fc[:, 'val']
                        fc.drop('filename', axis=1, inplace=True)
                        fc.drop('val', axis=1, inplace=True)
                        fc[:, '_x_res'] = abs(d.res[0])
                        fc[:, '_y_res'] = abs(d.res[1])
                        fc[:, '_crs'] = str(d.crs)
                        fc._data.reset_index(drop=True, inplace=True)
                        if self.normalize_value_property_to:
                            fc.map(partial(normalize, from_range=(0, 255), to_range=self.normalize_value_property_to),
                                   self.value_property_name, inplace=True)
                    io.save_fc(fc, path, vector, make_valid=True,
                               drop_empty=True, dropna=True, explode=True,
                               remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)
        else:
            # read bands
            bc = io.read_bc(path, self.input_rasters)
            for band, name in zip(bc, self.output_fcs):
                fc: FeatureCollection = polygonize(band)
                fc[:, '_x_res'] = abs(bc.transform[0])
                fc[:, '_y_res'] = abs(bc.transform[4])
                fc[:, '_crs'] = str(bc.crs)
                io.save_fc(fc, path, name, make_valid=True, drop_empty=True, dropna=True, explode=True,
                           remove_repeated_points=True, keep_only_geometry_types=shapely.Polygon)  # save


class ZonalStats(Brick):
    """
    Calculate zonal stats for polygons
    Args:
        input_raster: name of the raster to be processed
        input_vector: name of the vector with polygonal zones
        statistics: statistics to calculate
        nodata_value: nodata value
        field_names: field names to set calculated zonal values
        output_vector: name of output vector file
        reproj_vector_name: name of temporary reprojected vector
        input_rastr_ext: extension of input raster
        input_vector_ext: extension of input vector
    """
    input_raster: str
    input_vector: str
    statistics: List[str]
    nodata_value: float = Field(0)
    field_names: Optional[List[str]] = Field(None)
    output_vector: Optional[str] = Field(None)
    reproj_vector_name: str = Field('_reproj')
    input_rastr_ext: str = Field('.tif')
    input_vector_ext: str = Field('.geojson')

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if len(self.statistics) == 0 or not set(self.statistics).issubset(['min', 'max', 'mean', 'median', 'std', 'sum', 'count']):
            raise ValueError('statistics must be a list of one or more of "min", "max", "mean", "median", "std", "sum", "count"')

        self.field_names = self.field_names or self.statistics
        if len(self.field_names) != len(self.statistics):
            raise ValueError('`field_names` must be a list of the same length as `statistics`')

        self.output_vector = self.output_vector or self.input_vector

    def __call__(self, path):
        work_path = path
        try:
            with rasterio.open(os.path.join(work_path, self.input_raster + self.input_rastr_ext)) as dataset:
                epsg_code = "EPSG:{}".format(dataset.crs.to_epsg())
        except Exception as e:
            logger.exception('Raster projection reading failed!', str(e))
            raise

        try:
            fc = io.read_fc(work_path, self.input_vector)
            fc.to_crs(epsg_code, inplace=True)
            fc.to_file(os.path.join(work_path, self.reproj_vector_name + self.input_vector_ext), hold_crs=True)
        except Exception as e:
            logger.exception('Data reprojection to rastr crs failed!', str(e))
            raise

        try:
            z_stats = zonal_stats.zonal_stats(os.path.join(work_path, self.input_raster + self.input_rastr_ext),
                                              os.path.join(work_path, self.reproj_vector_name + self.input_vector_ext),
                                              nodata_value=self.nodata_value)
        except Exception as e:
            logger.exception('Zonal stats calculation failed!', str(e))
            raise

        try:
            for idx in range(len(fc)):
                for field_name in self.field_names:
                    fc[idx, field_name] = self.nodata_value

            for idx in range(len(z_stats)):
                f_idx = z_stats[idx][zonal_stats.FID_KEY]
                z_values = z_stats[idx]

                for field_name, statistic in zip(self.field_names, self.statistics):
                    fc[f_idx, field_name] = z_values[statistic]

        except Exception as e:
            logger.exception('Merging results failed!', str(e))
            raise
        try:
            fc.to_file(os.path.join(work_path, self.output_vector + self.input_vector_ext))
        except Exception as e:
            logger.exception('Result saving failed!', str(e))
            raise


class ZonalMedian(ZonalStats):
    field_name: str
    statistics: List[str] = ['median']

    def model_post_init(self, __context):
        self.field_names = [self.field_name]
        super().model_post_init(__context)