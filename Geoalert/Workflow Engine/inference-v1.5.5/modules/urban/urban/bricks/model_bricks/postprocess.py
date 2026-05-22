import numpy as np
from ...functional.remove_diagonal_connectivity import remove_diagonal_connectivity
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from ...base.registry_object import RegistryObject, CLASS_REGISTRY
from typing import Optional, Union, Literal
from pydantic import Field


class Postprocessor(RegistryObject):
    """Abstract class for ModelBrick sample postprocessing"""
    brick_class: str

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LabelsToOnehot(Postprocessor):
    """
    Converts label-encoded sample (H, W) or (1, H, W) to onehot-encoded (C, H, W)
    Args:
        n_classes (int): total number of classes (channels). Either n_classes or class_map must be specified
        class_map (dict): classes indexes to model outputs mapping. Either n_classes or class_map must be specified
        dtype (str): dtype to cast the result to
    """
    brick_class: Literal['LabelsToOnehot']
    n_classes: Optional[int] = Field(None)
    class_map: Union[list, tuple, str, None] = Field(None)
    dtype: str = Field('uint8')
    
    def model_post_init(self, __context):
        super().model_post_init(__context)
        if not self.class_map and not self.n_classes:
            raise ValueError('Either n_classes or class_map must be specified')
        if self.class_map:
            if isinstance(self.class_map, str):
                self.class_map = [int(i.strip()) for i in self.class_map.split(',')]
            self.n_classes = len(self.class_map)
        else:
            self.class_map = list(range(0, self.n_classes))

    def __call__(self, sample):
        class_map = np.array(self.class_map)
        if sample.ndim == 3:
            sample = sample[0]
        return (class_map == sample[..., None]).astype(self.dtype).transpose(2, 0, 1)


class SeparateInstances(Postprocessor):
    """
    Separates instances in mask with watershed algorythm based on aux_channel
    Args:
        channel (int): index of the channel in the mask with either semantic mask or instances 'insides'
                       result will be written in this channel
        aux_channel (int): index of the channel in the mask with either instances 'insides' or borders
        mode: one of 'semantic_and_markers' or 'markers_and_borders'
        min_marker (int): min size of marker
        drop_aux (bool): if True, drops aux_channel from result
    """
    brick_class: Literal['SeparateInstances']
    channel: int =  Field(0)
    aux_channel: int =  Field(1)
    mode: Literal['semantic_and_markers', 'markers_and_borders'] = Field('semantic_and_markers')
    min_marker: int =  Field(64)
    inflate_cnt: int =  Field(0)
    drop_aux: bool =  Field(True)

    def __call__(self, sample: np.ndarray):
        if sample.ndim != 3:
            raise ValueError(f'Expecting sample to have (C, H, W) shape, got {sample.shape}')
        if self.mode == 'semantic_and_markers':
            mask = sample[self.channel]
            markers = sample[self.aux_channel]
        else:
            mask = np.logical_or(sample[self.channel], sample[self.aux_channel]).astype(np.uint8)
            cnts = sample[self.aux_channel]
            if self.inflate_cnt:
                cnts = ndi.binary_dilation(cnts, iterations=self.inflate_cnt)
            markers = mask*(1-cnts)

        markers = ndi.label(markers)[0]
        markers = remove_small_objects(markers, min_size=self.min_marker)
        labels = watershed(- mask, markers=markers, mask=mask, watershed_line=True)
        result_mask = remove_diagonal_connectivity((labels > 0).astype('uint8'))
        sample[self.channel] = result_mask
        if self.drop_aux:
            sample = np.delete(sample, self.aux_channel, axis=0)
        return sample


ALL_POSTPROCESSOR_TYPES = Union[LabelsToOnehot, SeparateInstances]