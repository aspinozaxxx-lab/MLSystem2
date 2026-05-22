from loguru import logger
import numpy as np
import skimage.morphology as skm
from ..remove_diagonal_connectivity import remove_diagonal_connectivity


def reverse_mask(sample):
    return (sample == 0).astype(np.uint8)


def apply_mask(child, parent):
    return np.where(parent != 0, child, 0).astype(child.dtype)


unary_operations = {'open': skm.binary_opening,
                    'close': skm.binary_closing,
                    'erode': skm.binary_erosion,
                    'dilate': skm.binary_dilation,
                    'remove_diag_connect': remove_diagonal_connectivity
                    }

binary_operations = {'and': np.minimum,
                     'or': np.maximum,
                     'mask': apply_mask}


def get_binary_fn(operation_name, reverse_parent=False):
    """
    The following functions assume that the input BandSampleCollection has first BandSample as Parent
    which is to be applied to all the other BandSamples

    Intended for use in raster_ops bricks

    Args:
        operation_name:
        reverse_parent:

    Returns:

    """
    operation = binary_operations[operation_name]

    def binary_fn(sample):
        parent = sample[-1]
        if reverse_parent:
            parent = reverse_mask(parent)
        children = [layer for layer in sample[:-1]]
        return np.stack([operation(child, parent) for child in children], axis=0).astype(children[0].dtype)

    return binary_fn


def get_morphology_fn(operation_name, selem, selem_size, **kwargs):
    """
    returns a callable, unary mask operations, interface to one of the skimage functions to work
    with Predictor, intended for use with Morphology brick

    Args:
        operation_name: names of the unary operation defined in skimage.morphology module, like `binary_opening'
        selem: structuring element (name) from skimage.morphology.selem (disk, diamond, square, star)
        selem_size: size parameter for selem, a single int
        kwargs: args for the operation

    Returns:

    """
    if selem is not None:
        selem = getattr(skm, selem)(selem_size)
        kwargs['footprint'] = selem

    # backward compatibility - allowing to use a single operation
    if operation_name in unary_operations.keys():
        logger.warning('Deprecation warning! A single operation name selected from '
                       '`open` `close` `erode` and `dilate` will not be supported'
                       'Use a name of skimage.morphology operation instead')
        operation = unary_operations[operation_name]
    else:
        operation = (getattr(skm, operation_name))

    def fn(sample):
        result = [operation(layer.astype(bool), **kwargs) for layer in sample]
        return np.stack(result).astype('uint8')

    return fn
