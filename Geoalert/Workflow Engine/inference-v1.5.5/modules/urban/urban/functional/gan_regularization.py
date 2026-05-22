import numpy as np
import cv2
from skimage import measure
from skimage.transform import rescale


def fix_limits(i_min, i_max, j_min, j_max, min_image_size=256):

    def closest_divisible_size(size, factor=4):
        mod = size % factor
        return size + factor - mod if mod else size

    height = i_max - i_min
    width = j_max - j_min

    # pad the rows

    # 16 is a magic number from original code
    # https://github.com/zorzi-s/projectRegularization/blob/f5d168a7e02e890c04f0c6003057ff4fa372b519/regularize.py
    if height < min_image_size:
        diff = min_image_size - height
    else:
        diff = closest_divisible_size(height) - height + 16

    i_min -= (diff // 2)
    i_max += (diff // 2 + diff % 2)

    # pad the columns
    if width < min_image_size:
        diff = min_image_size - width
    else:
        diff = closest_divisible_size(width) - width + 16

    j_min -= (diff // 2)
    j_max += (diff // 2 + diff % 2)

    return i_min, i_max, j_min, j_max


def predict_building(rgb, mask, model, sample_size):
    y = np.array(mask, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    n = y.shape[0]
    categorical = np.zeros((n, 2), dtype='float32')
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (2,)
    mask = np.reshape(categorical, output_shape)

    rgb = rgb[np.newaxis, :, :, :]
    mask = mask[np.newaxis, :, :, :]
    rgb = rgb / 255.0
    merged_input = np.concatenate([rgb, mask], axis=-1)

    model_h, model_w = sample_size

    # TODO: figure out what is happening here and refactor
    _, src_h, src_w, _ = merged_input.shape
    input_with_meta = np.zeros((1, model_h, model_w, 6))
    input_with_meta[0, 1, 1, 5] = src_h
    input_with_meta[0, 1, 2, 5] = src_w
    input_with_meta[:, :src_h, :src_w, :5] = merged_input

    pred = model(input_with_meta[0].transpose(2, 0, 1)).transpose(1, 2, 0)

    pred = pred[:src_h, :src_w, :]
    pred = np.argmax(pred[:, :, :], axis=-1)
    return pred


def regularization(sample, model, min_size=16,  border=256):
    sample_numpy = sample.transpose(1, 2, 0)
    sample_h, sample_w, _ = sample_numpy.shape
    rgb, ins_segmentation = sample_numpy[..., :3], sample_numpy[..., 3]
    
    ins_segmentation = np.squeeze(np.uint16(ins_segmentation))
    rgb = np.uint8(rgb)

    ins_segmentation = np.uint16(measure.label(ins_segmentation, background=0))

    max_instance = np.amax(ins_segmentation)

    ins_segmentation = np.uint16(cv2.copyMakeBorder(ins_segmentation, border, border, border, border,
                                                    cv2.BORDER_CONSTANT, value=0))
    rgb = np.uint8(cv2.copyMakeBorder(rgb, border, border, border, border,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0)))

    regularization = np.zeros(ins_segmentation.shape, dtype=np.uint16)

    for ins in range(1, max_instance+1):
        indices = np.argwhere(ins_segmentation == ins)
        building_size = indices.shape[0]
        if building_size > min_size:
            i_min = np.amin(indices[:, 0])
            i_max = np.amax(indices[:, 0])
            j_min = np.amin(indices[:, 1])
            j_max = np.amax(indices[:, 1])

            i_min, i_max, j_min, j_max = fix_limits(i_min, i_max, j_min, j_max)

            mask = np.copy(ins_segmentation[i_min:i_max, j_min:j_max] == ins)
            rgb_mask = np.copy(rgb[i_min:i_max, j_min:j_max, :])

            max_building_size = min(sample_h, sample_w)
            rescaled = False
            if mask.shape[0] > max_building_size and mask.shape[0] >= mask.shape[1]:
                f = max_building_size / mask.shape[0]
                mask = rescale(mask, f, anti_aliasing=False, preserve_range=True)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False)
                rescaled = True
            elif mask.shape[1] > max_building_size and mask.shape[1] >= mask.shape[0]:
                f = max_building_size / mask.shape[1]
                mask = rescale(mask, f, anti_aliasing=False)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, preserve_range=True)
                rescaled = True

            pred = predict_building(rgb_mask, mask, model, (sample_h, sample_w))

            if rescaled:
                pred = rescale(pred, 1/f, anti_aliasing=False, preserve_range=True)

            pred_indices = np.argwhere(pred != 0)

            if pred_indices.shape[0] > 0:
                pred_indices[:, 0] = pred_indices[:, 0] + i_min
                pred_indices[:, 1] = pred_indices[:, 1] + j_min
                x, y = zip(*pred_indices)
                regularization[x, y] = 1

            # Make border
            kernel = np.ones((3, 3), 'uint8')
            pred_b0 = (pred != 0).astype(np.uint8)
            pred_b1 = cv2.dilate(pred_b0, kernel, iterations=1)
            pred = pred_b1 - pred_b0
            pred_indices = np.argwhere(pred != 0)
            if pred_indices.shape[0] > 0:
                pred_indices[:, 0] = pred_indices[:, 0] + i_min
                pred_indices[:, 1] = pred_indices[:, 1] + j_min
                x, y = zip(*pred_indices)
                regularization[x, y] = 0

    return np.expand_dims(regularization[border:-border, border:-border], axis=0)
