import numpy as np


def get_threshold_fn(threshold, strict_more=False):

    def threshold_fn(sample):
        if strict_more:
            return (sample > threshold).astype('uint8')
        else:
            return (sample >= threshold).astype('uint8')

    return threshold_fn


def get_multi_threshold_fn(thresholds, strict_more=False):
    """
    Takes a raster image (1-band) and makes a set of binary masks.
    The thresholds are so that the outer values are included as the masks also.
    For example, if the thresholds are 100 and 200, the output is 3 masks:
    sample < 100, 100 <= sample < 200, 200 <= sample
    Args:
        thresholds: a list of the threshold values for the image to be
        strict_more: if True, the thresholds are used as `image > threshold`, else `image >= threshold`
    Returns:
        a set of images
    """
    thresholds = sorted(thresholds)

    def multi_threshold(sample):
        masks = []
        image = sample
        if strict_more:
            masks.append(image <= thresholds[0])
            for lower, upper in zip(thresholds[:-1], thresholds[1:]):
                masks.append((image > lower) * (image <= upper))
            masks.append(image > thresholds[-1])
        else:
            masks.append(image < thresholds[0])
            for lower, upper in zip(thresholds[:-1], thresholds[1:]):
                masks.append((image >= lower) * (image < upper))
            masks.append(image >= thresholds[-1])
        return np.concatenate(masks, axis=0).astype('uint8')
    return multi_threshold
