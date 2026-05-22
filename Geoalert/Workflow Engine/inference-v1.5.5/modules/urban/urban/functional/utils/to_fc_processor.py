from typing import List, Tuple, Callable
from aeronet_raster import BandCollection, SequentialSampler
import numpy as np
from tqdm.auto import tqdm
import cv2
import shapely
from shapely.geometry import Polygon
from loguru import logger

from gpdadapter import FeatureCollection


class ToFeatureCollectionProcessor:
    """
    Similar to aeronet_raster.CollectionProcessor, but instead of writing into BandCollection,
    returns FeatureCollection with vectorized Features
    Args:
        input_channels: List[str] - names of input bands
        processing_fn: Callable - processing function, receives sample as numpy array, returns tuple (masks, scores)
        sample_size: Tuple[int] - sample size, excluding bounds
        bound: int - margin value for samples
        src_nodata: int - nodata value for source bands
        nodata_mask_mode: bool - if true, mask put nodata values
        padding: str - 'none' - no padding, 'mirror' - mirror padding of nodata areas
        simplify: float - simplification param for detected features in pixels
        area_threshold: float - features with area less than threshold will be deleted
        score_threshold: float - features with confidence score less than threshold will be deleted
        verbose: bool - if true, shows progress bar
    """

    def __init__(self,
                 input_channels: List[str],
                 processing_fn: Callable,
                 sample_size: Tuple[int, int] = (1024, 1024),
                 bound: int = 256,
                 src_nodata: int = 0,
                 nodata_mask_mode: bool = False,
                 padding: str = 'none',
                 simplify: float = 0,
                 area_threshold: float = 10,
                 score_threshold: float = 0,
                 properties_type: str = 'score',
                 verbose: bool = True):

        self.input_channels = input_channels
        self.processing_fn = processing_fn
        self.sample_size = sample_size
        self.bound = bound
        self.src_nodata = src_nodata
        self.nodata_mask_mode = nodata_mask_mode
        self.padding = padding
        self.simplify = simplify
        self.area_threshold = area_threshold
        self.score_threshold = score_threshold
        self.properties_type = properties_type
        self.verbose = verbose

    def _processing(self, sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # sample.shape=(C, H, W)
        if not np.all(sample == self.src_nodata):
            return self.processing_fn(sample)
        return np.zeros(0), np.zeros(0)

    def process(self, bc: BandCollection) -> FeatureCollection:
        self.src_nodata = self.src_nodata if bc.nodata is None else bc.nodata
        src = SequentialSampler(bc, self.input_channels, self.sample_size, self.bound, self.padding, self.src_nodata,
                                self.nodata_mask_mode)

        blocks_num = (bc.shape[1] // self.sample_size[0] + int((bc.shape[1] % self.sample_size[0]) != 0)) * \
                     (bc.shape[2] // self.sample_size[1] + int((bc.shape[2] % self.sample_size[1]) != 0))
        features = {'geometry': list()}
        with tqdm(src, total=blocks_num, disable=(not self.verbose)) as data:
            for sample, block in data:
                masks, properties = self._processing(sample)
                logger.trace(f" x: {block['x']} - {block['x'] + sample.shape[2]},"
                             f" y = {block['y']} - {block['y'] + sample.shape[1]},"
                             f" masks={len(masks)}")
                for mask, property_ in zip(masks, properties):
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:  # no contours detected for current instance
                        logger.warning(f'No contours detected for instance {mask.shape}')
                        continue
                    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    if len(cnt) < 3:  # cnt is invalid
                        logger.warning(f'Invalid contour {cnt}!!!')
                        continue
                    cnt[:, 0] += block['x']
                    cnt[:, 1] += block['y']
                    cnt = Polygon(cnt).simplify(self.simplify)
                    if cnt.area > self.area_threshold:
                        if self.properties_type == 'score':
                            score = property_
                            if score > self.score_threshold:
                                features['geometry'].append(shapely.affinity.affine_transform(cnt, bc.transform.to_shapely()))
                                if 'score' not in features:
                                    features['score'] = list()
                                features['score'].append(float(score))
                        elif self.properties_type == 'dict':
                            features['geometry'].append(
                                shapely.affinity.affine_transform(cnt, bc.transform.to_shapely()))
                            for k, v in property_.items():
                                if k not in features:
                                    features[k] = list()
                                features[k].append(v)
                        else:
                            raise ValueError(f'Unknown properties type: {self.properties_type}')
        fc = FeatureCollection(features, crs=bc.crs)
        return fc
