# TODO: deprecated

import cv2
import numpy as np
from typing import Tuple, Optional
from aeronet_raster.collectionprocessor import CollectionProcessor as Predictor
from ..base import Brick
from ..functional import io
from ..functional.raster_ops import separate_semantically_segmented_fields
from gpdadapter import FeatureCollection
from shapely.geometry import box
from shapely.ops import polygonize, unary_union
from pydantic import Field


class SeparateSemanticSegmentedFields(Brick):
    """
    This function applies all postprocessing steps at once to inference of semantic segmentation
    model.
    Those postprocessing steps are:
        * cleaning boundaries and fields mask
        * Skeletonization of boundaries
        * Probability subtraction and binarization
        * Watershed
        * Deleting too small fields

    Args:
        boundaries_mask_input: str, label for boundaries mask
        fields_mask_input: str, label for fields mask
        output: str, Single output semantic segmentation mask.
        area_filter: Float, value for thresholding minimum area of fields instances.
        boundaries_proba_thr:  float, threshold for boundaries applied before subtraction of probability
        opening_kernel_small_iterations: int, Number of opening iterations for cleaning fields mask and for dilation
        opening_kernel_big_iterations: int, Number of opening iterations applied to fields mask
        dilate_iterations_sure_bg: int, Number of dilation iterations applied to opening mask to get sure background
        dilate_iterations_sure_fg_proba: int, number of dilation iterations applied to probability mask 
                                                (after subtraction of probabilities)
        enhance_instances_delineation: bool, if True, then result segmentation mask will be eroded with smallest
                                        3x3 kernel, to avoid diagonal pixels. Need for vectorization.
        
        std_width: parameter for meanstd method
        out_dtype: data type for output image
        n_workers: number of treads for data processing
        sample_size: size of pieces for processing
        verbose: if `True` tqdm progress bar is used
        bound: int, size of overlapping boundaries for Predictor
    
    """
    boundaries_mask_input: str
    fields_mask_input: str
    output: str
    area_filter: float = Field(83)
    boundaries_proba_thr: float = Field(0.5)
    opening_kernel_small_iterations: int = Field(3)
    opening_kernel_big_iterations: int = Field(1)
    dilate_iterations_sure_bg: int = Field(5)
    dilate_iterations_sure_fg_proba: int = Field(1)
    enhance_instances_delineation: bool = Field(True)
    std_width: int = Field(3)
    out_dtype: str = Field('uint8')
    n_workers: int = Field(1)
    sample_size: Tuple[int, int] = Field((2048, 2048))
    bound: int = Field(256)
    verbose: bool = Field(False)
    small_kernel_size: int = Field(9)
    big_kernel_size: int = Field(17)

    def __call__(self, path):
        bc = io.read_bc(path, [self.boundaries_mask_input, self.fields_mask_input])
        
        # We construct the predictor here and not in __init__ because it may need the data from the image for setup
        predictor = Predictor(
            input_channels=[self.boundaries_mask_input, 
                            self.fields_mask_input],
            output_labels=[self.output, ],
            processing_fn=self.get_processing_fn(),
            n_workers=self.n_workers,
            sample_size=self.sample_size,
            bound=self.bound,
            verbose=self.verbose,
            dtype=self.out_dtype
        )
        predictor.process(bc, path)

    def calculate_input_range(self, bc):
        """
        We should be able to calculate the statistics of the image to get the input range for normalization
        However, it is not necessary right now, so we leave this without implementation
        and require the values to be passed as input
        Returns:
        """
        raise NotImplementedError

    def get_processing_fn(self,):
        kernel_small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (self.small_kernel_size, self.small_kernel_size))
        kernel_big_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                       (self.big_kernel_size, self.big_kernel_size))

        def fn(sample):
            result = separate_semantically_segmented_fields(
                raw_predictions=sample.astype(np.float32)/255.0,
                kernel_small_ellipse=kernel_small_ellipse,
                kernel_big_ellipse=kernel_big_ellipse,
                area_filter=self.area_filter,
                boundaries_proba_thr=self.boundaries_proba_thr,
                opening_kernel_small_iterations=self.opening_kernel_small_iterations,
                opening_kernel_big_iterations=self.opening_kernel_big_iterations,
                dilate_iterations_sure_bg=self.dilate_iterations_sure_bg,
                dilate_iterations_sure_fg_proba=self.dilate_iterations_sure_fg_proba,
                enhance_instances_deliniation=self.enhance_instances_delineation,
            )
            return result[0]
        return fn


class Boundaries2Polygons(Brick):
    """
    The field boundaries mask can be vectorized as linestrings (using roads postprocessing pipeline),
    and we can represent the fields as polygons formed by the boundaries.
    Args:
        linestrings_input: str, label for boundaries mask
        output: str, Single output semantic segmentation mask.
        aoi_image_input: str, if not None, the boundary of the image will be added to the linestrings set to make the
                              areas which contact with the boundary polygons, not empty spaces
        buffer: float, defines how much do we cut off the input image to form the aoi.
                       We can make AOI smaller to ensure that the lines will reach the boundary.
                       You should use negative value if you want to make aoi smaller!

    """
    linestrings_input: str
    output: str
    aoi_image_input: Optional[str] = Field(None)
    buffer: float = Field(0.0)

    def __call__(self, path):
        # if we use image extent and buffer, we need crs with known linear units
        # else, we will not spend time on reprojection
        if self.aoi_image_input:
            bc = io.read_bc(path, [self.aoi_image_input])
            crs = bc[0].crs
            aoi = box(*bc[0].bounds).buffer(self.buffer).boundary
            linestrings = io.read_fc(path, self.linestrings_input, crs=crs)
            linestrings.append({'geometry': aoi})
            polygons = polygonize(unary_union(linestrings.geometry))
            fc = FeatureCollection(polygons, crs=crs)
        else:
            linestrings = io.read_fc(path, self.linestrings_input)
            polygons = polygonize(unary_union(linestrings))
            fc = FeatureCollection(polygons)
        io.save_fc(fc=fc, path=path, name=self.output)
