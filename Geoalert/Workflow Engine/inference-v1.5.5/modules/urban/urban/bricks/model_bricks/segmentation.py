from aeronet_raster.collectionprocessor import CollectionProcessor as Predictor
from aeronet_raster.utils.coords import get_utm_zone
from rasterio.warp import calculate_default_transform
from ...functional import io
from typing import Tuple, List, Optional, Union, Literal, Sequence, Callable
from .modelbrick import ModelBrick
from loguru import logger
from ...base import defaults
from .postprocess import Postprocessor
from pydantic import Field
from ...functional.utils.to_fc_processor import ToFeatureCollectionProcessor


ProcessorType = Union[Predictor, ToFeatureCollectionProcessor, Callable]

class Segmentation(ModelBrick):
    """
    Segmentation model generates output according to model.output_labels

    Args:
        adapter: urban.ModelAdapter or dict config for constructing urban.ModelAdapter
        input_rasters (List[str]): filenames of input bands
        output_labels (List[str]): filenames of output bands (or vector files)
                                   if model returns more channels, than needs to be saved,
                                   pass empty string on corresponding index
        res (Tuple[float, float]): input resolution in ``crs`` units. Defaults to None.
        crs (str): input data projections. Defaults to None.
        nodata (Optional[Union[int, float]], optional): nodata value. Used to determine, if sample should
                                                        be skipped without processing. Defaults to None.
        sample_size: tuple of int, size of pieces for prediction
        bounds: int, size of bounds for each piece that will be cut off after processing
        res_tolerance: relative difference between the model-defined image spatial resolution
             and the actual image resolution, at which the image is not resampled.
             This allows to skip resampling if 1 - res_tolerance < img_res/model_res < 1 + res_tolerance
             which means that the images are at almost the same scale, and it will have little effect
             The resolution at each axis is checked separately
        max_upsampling: upper boundary for upsampling during the segmentation stage. I.e. if model requires 1m
             and data res is 3m, it is OK for default value 4, but if the data res is 5 m, it will fail.
        interpolation: str name of interpolation method to use
        verbose: if `True` tqdm progress bar is used
        bound_mode (str): 'drop' or 'weight', default 'drop', how to handle boundaries:
                          'drop' - drop boundaries, 'weight' - weight boundaries
        padding (str): 'none' or 'mirror', default 'none':
                       'none' - no padding, 'mirror' - mirror padding of nodata areas
        nodata_mask_mode (bool): whether to fill by dst_nodata where nodata mask is True
        processing_fn_use_block: bool, whether to pass 'block' argument to processing_fn
        write_output_to_dst: bool, whether to write output of processing_fn to dst
        postprocessors (list[Postprocessor]): list of Postprocessor objects to process every sample model returns
    """
    input_rasters: Sequence[str]
    output_labels: Sequence[str]
    res: Optional[Tuple[float, float]] = Field(default=None)
    min_res: Optional[float] = Field(default=None)
    max_res: Optional[float] = Field(default=None)
    crs: Optional[str] = Field(default=None)
    nodata: Optional[Union[int, float]] = Field(default=0)
    sample_size: Tuple[int, int] = Field(default=defaults.DEFAULT_SAMPLE_SIZE)
    bounds: int = Field(default=defaults.DEFAULT_BOUNDS)
    res_tolerance: float =  Field(default=0.0)
    max_upsampling: float =  Field(default=6)
    interpolation: str =  Field(default=defaults.DEFAULT_RESIZE_INTERPOLATION)
    bound_mode: Literal['drop', 'weight'] = Field(default='drop')
    padding: Literal['none', 'mirror'] = Field(default='none')
    nodata_mask_mode: bool = Field(default=False)
    processing_fn_use_block: bool = Field(default=False)
    write_output_to_dst: bool = Field(default=True)
    _predictor: ProcessorType

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.min_res is not None and self.max_res is not None and self.res is not None:
            if not (self.min_res <= self.res[0] <= self.max_res and self.min_res <= self.res[1] <= self.max_res):
                raise ValueError(f"Model resolution `res` = {self.res} is not in range [`min_res`, `max_res`] = [{self.min_res}, {self.max_res}]")
        self._predictor = Predictor(input_channels=self.input_rasters,
                                    output_labels=[fname for fname in self.output_labels if fname],
                                    processing_fn=self.processing_fn,
                                    n_workers=1,
                                    sample_size=self.sample_size,
                                    bound=self.bounds,
                                    verbose=self.verbose,
                                    dst_dtype=self.adapter.output_dtype or 'uint8',
                                    src_nodata=self.nodata,
                                    dst_nodata=self.nodata,
                                    bound_mode=self.bound_mode,
                                    padding=self.padding,
                                    nodata_mask_mode=self.nodata_mask_mode,
                                    processing_fn_use_block=self.processing_fn_use_block,
                                    write_output_to_dst=self.write_output_to_dst)

    def processing_fn(self, x):
        x = self.adapter(x)
        for postprocessor in self.postprocessors:
            logger.trace(f'Applying {postprocessor.__class__.__name__}')
            x = postprocessor(x)
            logger.trace(f'shape = {x.shape}, dtype = {x.dtype}')
        if '' in self.output_labels:
            x = x[[i for i, n in enumerate(self.output_labels) if n]]
            logger.trace(f'Dropped channels {[i for i, n in enumerate(self.output_labels) if n]}, shape = {x.shape}')
        return x

    def preprocess(self, bc):
        # get model CRS
        if self.crs == 'utm':
            model_crs = get_utm_zone(bc.crs, bc.transform, bc.shape)
        else:
            model_crs = self.crs
        # calculate default transform
        # abs(transform[0]): x_resolution
        # abs(transform[4]): y_resolution
        try:
            transform, width, height = calculate_default_transform(bc.crs, model_crs, bc.width, bc.height, *bc.bounds)
        except Exception:
            raise ValueError(f'The original data cannot be transformed to Web Mercator to calculate resolution'
                             f'There is most probably error in original image georeference:'
                             f'crs = {bc.crs}, '
                             f'transform = {bc.transform}, '
                             f'image size = {bc.height}, {bc.width}')

        # If input resolution is too low and the resulting image size will increase too much,
        # which most probably will produce BAD results and consume like infinite disk space, we should fail now
        if (self.res[0] * self.max_upsampling < abs(transform[0])
                or self.res[1] * self.max_upsampling < abs(transform[4])):
            raise ValueError(f'Model is designed for resolution {self.res}, and the segmentation allows '
                             f'it to be {self.max_upsampling} times worse. '
                             f'Got Resolution of input data ({abs(transform[0])}, {abs(transform[4])}).')

        # preprocess raster bands
        # The scales can slightly differ between the axes, so we cannot require them to be the same.
        # We just check each axis separately.
        # This can potentially give warped images if the x-scale is 0.9 and y-scale is 1.1, for example.
        #
        # Also, the problem is asymmetry: if tolerance = 0.5, the resolution can be 2 times higher (1 - 0.5 = 1/2)
        # or only 1.5 times lower (1 + 0.5 = 3/2).
        # But this is the same strategy that is applied in Albumentations,
        # and when tolerance is low, it will not be a problem

        # checking on res_tolerance (if not met, reproject regardless of crs)
        if self.min_res is not None and self.max_res is not None:
            min_res = (self.min_res, self.min_res)
            max_res = (self.max_res, self.max_res)
        else:
            min_res = ((1 - self.res_tolerance) * self.res[0], (1 - self.res_tolerance) * self.res[1])
            max_res = ((1 + self.res_tolerance) * self.res[0], (1 + self.res_tolerance) * self.res[1])

        if not (min_res[0] <= abs(transform[0]) <= max_res[0] and min_res[1] <= abs(transform[4]) <= max_res[1]):
            # Resample and reproject
            # reproject(dst_crs, dst_res=None, fp=None, interpolation='nearest'):
            bc = bc.reproject(model_crs, self.res, interpolation=self.interpolation)
        elif bc.crs != model_crs:
            # if the resolution is inside tolerance range, reproject only
            bc = bc.reproject(model_crs, interpolation=self.interpolation)
        return bc

    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        if self.crs is not None and self.res is not None:
            bc = self.preprocess(bc)
        labels_bc = self._predictor.process(bc, path)
