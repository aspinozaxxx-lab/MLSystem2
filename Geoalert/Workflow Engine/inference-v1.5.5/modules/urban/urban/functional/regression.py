import rasterio.mask
import numpy as np
import shapely
import shapely.affinity
import rasterio.features
from gpdadapter import FeatureCollection
from aeronet_raster import BandCollection
from loguru import logger
from ..base import defaults
from typing import Final, Tuple, Callable, List, Iterable, Optional
from .utils.angleutils import height_from_vector, azimuth_from_vector, elevation_from_vector, vector_from_height
from .utils.buildingsutils import extrude


DEFAULT_CROP_SIZE: Final[tuple] = (512, 512)
PADDING_VALUE: Final[int] = 127
DEFAULT_CROP_BUFFER: Final[int] = 15  # in crs units, which mean to be epsg3857
DEFAULT_NEGATIVE_BUFFER: Final[int] = 5  # buffer to cut out neighbour buildings
DEFAULT_N_NEIGHBOURS: Final[int] = 10
DEFAULT_HEIGHT_MULTIPLIER: Final[float] = 1.3
MIN_PROPORTION_OF_REFERENCES: Final[float] = 0.3
DEFAULT_PROPOSAL_HEIGHT: Final[float] = 20


def get_split_rule(prop_name: str, thr: float) -> Callable[[FeatureCollection], bool]:
    return lambda x: x.properties.get(prop_name) is not None and x.properties[prop_name] > thr


def pad_or_crop_if_needed(image: np.ndarray,
                          target_size: Tuple[int, int],
                          padding_value: int = 127) -> np.array:
    """Fits an image represented as (C, H, W) or (H, W) to a given size (H_target, W_target)
    by padding or cropping if needed
    Args:
        image: np.array, (C, H, W) or (H, W)
        target_size: (H_target, W_target)
        padding_value: value to fill the padded area
    Returns:
        padded or cropped image
    """
    if image.ndim not in (2, 3):
        raise ValueError(f'Only (C, H, W) or (H, W) images are allowed, got {image.shape}')

    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    pad_or_crop = np.array(target_size) - np.array(image.shape[1:])  # negative values - crop, positive - pad
    pad = np.clip(pad_or_crop, 0,None)
    crop = np.clip(-pad_or_crop, 0, None)

    image = np.pad(image, ((0, 0),
                           (pad[0] // 2, pad[0] // 2 + pad[0] % 2),
                           (pad[1] // 2, pad[1] // 2 + pad[1] % 2)),
                   mode='constant',
                   constant_values=padding_value)
    return image[:,
                 crop[0] // 2:image.shape[1] - crop[0] // 2 - crop[0] % 2,
                 crop[1] // 2:image.shape[2] - crop[1] // 2 - crop[1] % 2]


def crop_by_mask_old(bc: BandCollection,
                     geometry: shapely.geometry.Polygon,
                     sample_size: tuple = DEFAULT_CROP_SIZE,
                     padding_value: int = PADDING_VALUE,
                     **kwargs) -> np.ndarray:
    """
    Crops image from rasterio dataset by a mask (slower version since rasterio.mask reads the whole image)
    Args:
        bc: BandCollection
        geometry: shapely geometry (e.g. Polygon or Multipolygon)
        sample_size: target size of the image (H, W)
        padding_value: value to fill pixels outside geometry
    Returns:
         image as np.array of shape (3, H, W)
    """
    image = np.stack([rasterio.mask.mask(band._band, [geometry], crop=True)[0][0] for band in bc], axis=0)
    image[:, np.sum(image, axis=0) == 0] = padding_value
    pad_or_crop = np.array(sample_size) - np.array(image.shape[1:])
    pad = np.maximum(pad_or_crop, np.zeros(2, int))
    crop = np.negative(np.minimum(pad_or_crop, np.zeros(2, int)))
    image = np.pad(image, ((0, 0),
                           (pad[0] // 2, pad[0] // 2 + pad[0] % 2),
                           (pad[1] // 2, pad[1] // 2 + pad[1] % 2)),
                   mode='constant',
                   constant_values=padding_value)
    return image[:,
                 crop[0] // 2:image.shape[1] - crop[0] // 2 - crop[0] % 2,
                 crop[1] // 2:image.shape[2] - crop[1] // 2 - crop[1] % 2]


def crop_by_mask(bc: BandCollection,
                 geometry: shapely.geometry.Polygon,
                 sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                 padding_value: int = PADDING_VALUE,
                 **kwargs):
    """
    Crops image from rasterio dataset by a mask
    Args:
        bc: BandCollection
        geometry: shapely geometry (e.g. Polygon or Multipolygon)
        sample_size: target size of the image (H, W)
        padding_value: value to fill pixels outside geometry
    Returns:
         image as np.array of shape (3, H, W)
    """
    geometry = shapely.affinity.affine_transform(geometry, (~bc.transform).to_shapely())
    x, y = (int(i[0]) for i in geometry.centroid.coords.xy)

    mask = np.clip(rasterio.features.rasterize([shapely.affinity.translate(geometry,
                                                                   -x + sample_size[1] // 2,
                                                                   -y + sample_size[0] // 2)],
                                                out_shape=sample_size), 0, 1)

    image = bc.sample(y - sample_size[0] // 2, x - sample_size[1] // 2, *sample_size).numpy()
    image[:, mask == 0] = padding_value
    return image


def crop_by_window_with_mask(bc: BandCollection,
                             geometry: shapely.geometry.Polygon,
                             sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                             **kwargs):
    """
    Crops image from rasterio dataset by a rectangular window centered at the mask geometrical center
    and adds the mask as 4-th channel
    Args:
        bc: BandCollection
        geometry: shapely geometry (e.g. Polygon or Multipolygon)
        sample_size: target shape of image (H, W)
    Returns:
         image as np.array of shape (4, H, W) with rasterized geometry in 4-th channel
    """
    geometry = shapely.affinity.affine_transform(geometry, (~bc.transform).to_shapely())
    x, y = (int(i[0]) for i in geometry.centroid.coords.xy)

    mask = np.clip(rasterio.features.rasterize([shapely.affinity.translate(geometry,
                                                                   -x + sample_size[1] // 2,
                                                                   -y + sample_size[0] // 2)],
                                                out_shape=sample_size), 0, 1)

    image = bc.sample(y - sample_size[0] // 2, x - sample_size[1] // 2, *sample_size).numpy()

    return np.concatenate((image, np.expand_dims(mask, 0)), axis=0)


def get_feature_mask(fc: FeatureCollection,
                     idx: int,
                     angles: Optional[dict] = None,
                     buffer: float = DEFAULT_CROP_BUFFER,
                     proposal_height_tag:  Optional[str] = None,
                     default_proposal_height: float = DEFAULT_PROPOSAL_HEIGHT,
                     subtract_neighbours=False,
                     negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
                     height_mul: float = DEFAULT_HEIGHT_MULTIPLIER) -> shapely.geometry.Polygon:
    """Returns mask including pseudo-wall, generated from height and excluding neighbour rooftops"""
    pos_mask = fc[idx, 'geometry']
    if angles:
        if proposal_height_tag and pos_mask.properties.get(proposal_height_tag):
            height = fc[idx, proposal_height_tag] * height_mul
        else:
            height = default_proposal_height*height_mul
        vec = vector_from_height(angles[defaults.SAT_AZIMUTH_TAG], angles[defaults.SAT_ELEVATION_TAG], height)
        pos_mask = extrude(fc[idx, 'geometry'], vec).buffer(buffer)
    else:
        pos_mask = fc[idx, 'geometry'].buffer(buffer)
    if subtract_neighbours:
        neg_masks_indexes = fc.query(pos_mask)
        for neg_idx in neg_masks_indexes:
            if neg_idx != idx:
                pos_mask = pos_mask.difference(fc[neg_idx, 'geometry'].buffer(negative_buffer))
    return pos_mask


def get_embeddings(data: Iterable,
                   model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Produces an array of embeddings from list of images
    Args:
        data: iterable, returning images in applicable form for the model
        model: Callable receives image as (C, W, H) np.ndarray, returns np.array of size (emb_len, )
    Returns:
         np.array of embeddings, shape=(n_samples, emb_dim)
    """
    embs = list()
    for sample in data:
        embs.append(model(sample))
    return np.stack(embs)


def get_closest(x: np.array,
                embs: np.array,
                n_neighbours: int = DEFAULT_N_NEIGHBOURS,
                **kwargs) -> Tuple[List[int], List[float]]:
    """
    Finds embeddings, closest to the input in terms of l2 distance
    Args:
        x: target embedding, 1d array, shape=(emb_dim,)
        embs: np.array of embeddings, shape=(n_samples, emb_dim)
        n_neighbours: number of neighbours to return
    Returns:
         tuple[indexes, scores] = indexes of the closest embeddings ordered by distance and corresponding distances
    """
    d = np.linalg.norm(np.repeat(np.expand_dims(x, 0), len(embs), axis=0) - embs, axis=1)
    indexes = list()
    scores = list()
    if n_neighbours > len(embs):
        n_neighbours = len(embs)
    for _ in range(n_neighbours):
        indexes.append(np.argmin(d))
        scores.append(d[indexes[-1]])
        d[indexes[-1]] = np.max(d) + 1
    return indexes, scores


def get_height_from_embeddings(x: np.array,
                               embs: np.array,
                               heights: np.array,
                               n_neighbours: int = DEFAULT_N_NEIGHBOURS) -> Tuple[float, float]:
    """
    Calculates height of a building based on the known heights of neighbours in embedding space
    Args:
        x: target embedding, 1d array, shape=(emb_dim,)
        embs: np.array of embeddings, shape=(n_samples, emb_dim)
        heights: np.array of heights, shape=(n_samples, )
        n_neighbours: number of neighbours to pass in get_closest()
    Returns:
         tuple: float, float - estimated height and average weights (for confidence score)
    """
    if heights.shape[0] != embs.shape[0]:
        raise ValueError(f'embeddings len must match heights len, got {len(embs)}, {len(heights)}')
    pred_idxs, weights = get_closest(x, embs, n_neighbours)
    return np.average([heights[p_i] for p_i in pred_idxs], weights=weights), np.mean(weights)


def simple_regression(bc: BandCollection,
                      fc: FeatureCollection,
                      model: Callable,
                      sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                      tag: str = defaults.REGR_HEIGHT_TAG,
                      undefined_result: float = defaults.UNDEFINED_HEIGHT,
                      failsafe: bool = True,
                      buffer: float = DEFAULT_CROP_BUFFER,
                      subtract_neighbours: bool = False,
                      negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
                      padding_value: int = PADDING_VALUE,
                      method: str = 'crop_by_mask') -> FeatureCollection:
    """
    Predicts building heights for every building (Feature) in fc using regression model and writes
    result into corresponding feature property
    Args:
        bc: input BandCollection
        fc: input FeatureCollection, must have same crs, will be modified in-place
        model: Callable receives image as (C, W, H) np.ndarray, returns np.array of size (1, )
        sample_size: image size appropriate for the model, (H, W)
        tag: name of the property to write the result to
        undefined_result: value to write into target_property in case of exception
        failsafe: if an instance fails, continues with default height
        buffer: float value to inflate the mask
        subtract_neighbours: bool =False,
        negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
        padding_value: int = PADDING_VALUE,
        method: 'crop_by_mask' or 'add_mask'
    Returns:
        Modified FeatureCollection
    """
    for f_idx in range(len(fc)):
        try:
            mask = get_feature_mask(fc, f_idx, buffer=buffer,
                                    subtract_neighbours=subtract_neighbours,
                                    negative_buffer=negative_buffer)
            if method == 'crop_by_mask':
                sample = crop_by_mask(bc, mask, sample_size=sample_size, padding_value=padding_value)
            elif method == 'add_mask':
                sample = crop_by_window_with_mask(bc, mask, sample_size=sample_size)
            else:
                raise ValueError(f'Unknown method {method}, must be "crop_by_mask" or "add_mask"')
            res = model(sample).item()
            fc[f_idx, tag] = res
        except Exception as e:
            if failsafe:
                logger.opt(exception=True).warning(e)
                fc[f_idx, tag] = undefined_result
                continue
            else:
                raise e
    return fc


def predict_heights_with_regression(bc: BandCollection,
                                    fc: FeatureCollection,
                                    model: Callable,
                                    sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                                    regr_height_tag: str = defaults.REGR_HEIGHT_TAG,
                                    undefined_result: float = defaults.UNDEFINED_HEIGHT,
                                    failsafe: bool = True,
                                    angles: Optional[dict] = None,
                                    buffer: float = DEFAULT_CROP_BUFFER,
                                    proposal_height_tag: Optional[str] = None,
                                    default_proposal_height: float = DEFAULT_PROPOSAL_HEIGHT,
                                    subtract_neighbours: bool = False,
                                    negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
                                    height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
                                    padding_value: int = PADDING_VALUE,
                                    method: str = 'crop_by_mask') -> FeatureCollection:
    """
    Predicts building heights for every building (Feature) in fc using regression model and writes
    result into corresponding feature property
    Args:
        bc: input BandCollection
        fc: input FeatureCollection, must have same crs, will be modified in-place
        model: Callable receives image as (C, W, H) np.ndarray, returns np.array of size (1, )
        sample_size: image size appropriate for the model, (H, W)
        regr_height_tag: name of the property to write the result to
        undefined_result: value to write into target_property in case of exception
        failsafe: if an instance fails, continues with default height
        angles: dict meta angles as dict
        buffer: float value to inflate the mask
        proposal_height_tag: float tag in feature properties with proposal height
        default_proposal_height: float
        subtract_neighbours: bool =False,
        negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
        height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
        padding_value: int = PADDING_VALUE,
        method: 'crop_by_mask' or 'add_mask'
    Returns:
        Modified FeatureCollection
    """
    for f_idx in range(len(fc)):
        try:
            mask = get_feature_mask(fc, f_idx, angles=angles, buffer=buffer, proposal_height_tag=proposal_height_tag,
                                    default_proposal_height=default_proposal_height,
                                    subtract_neighbours=subtract_neighbours,
                                    negative_buffer=negative_buffer, height_mul=height_mul)
            if method == 'crop_by_mask':
                sample = crop_by_mask(bc, mask, sample_size=sample_size, padding_value=padding_value)
            elif method == 'add_mask':
                sample = crop_by_window_with_mask(bc, mask, sample_size=sample_size)
            else:
                raise ValueError(f'Unknown method {method}, must be "crop_by_mask" or "add_mask"')
            res = model(sample).item()
            if res <= 0:
                logger.warning(f'Got regression value {res} <= 0')
                raise ValueError
            fc[f_idx, regr_height_tag] = int(np.round(res, defaults.HEIGHT_DECIMALS))
        except Exception as e:
            if e in model.exceptions:
                raise e
            if failsafe:
                logger.opt(exception=True).warning(e)
                fc[f_idx, regr_height_tag] = undefined_result
                continue
            else:
                raise e
    return fc


def predict_fp_shift_with_regression(bc: BandCollection,
                                     fc: FeatureCollection,
                                     model: Callable,
                                     sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                                     failsafe: bool = True,
                                     angles: Optional[dict] = None,
                                     buffer: float = DEFAULT_CROP_BUFFER,
                                     proposal_height_tag: Optional[str] = None,
                                     default_proposal_height: float = DEFAULT_PROPOSAL_HEIGHT,
                                     subtract_neighbours: bool = False,
                                     negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
                                     height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
                                     padding_value: int = PADDING_VALUE,
                                     method: str = 'crop_by_mask') -> FeatureCollection:
    """
    Predicts building heights for every building (Feature) in fc using regression model and writes
    result into corresponding feature property
    Args:
        bc: input BandCollection
        fc: input FeatureCollection, must have same crs, will be modified in-place
        model: Callable receives image as (C, W, H) np.ndarray, returns np.array of size (1, )
        sample_size: image size appropriate for the model, (H, W)
        failsafe: if an instance fails, continues with default height
        angles: dict meta angles as dict
        buffer: float value to inflate the mask
        proposal_height_tag: float tag in feature properties with proposal height
        default_proposal_height: float
        subtract_neighbours: bool =False,
        negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
        height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
        padding_value: int = PADDING_VALUE,
        method: 'crop_by_mask' or 'add_mask'
    Returns:
        Modified FeatureCollection
    """
    for f_idx in range(len(fc)):
        try:
            mask = get_feature_mask(fc, f_idx, angles=angles, buffer=buffer, proposal_height_tag=proposal_height_tag,
                                    default_proposal_height=default_proposal_height,
                                    subtract_neighbours=subtract_neighbours,
                                    negative_buffer=negative_buffer, height_mul=height_mul)
            if method == 'crop_by_mask':
                sample = crop_by_mask(bc, mask, sample_size=sample_size, padding_value=padding_value)
            elif method == 'add_mask':
                sample = crop_by_window_with_mask(bc, mask, sample_size=sample_size)
            else:
                raise ValueError(f'Unknown method {method}, must be "crop_by_mask" or "add_mask"')
            res = model(sample)
            assert len(res) == 2
            fc[f_idx, 'x_shift'] = float(res[0])
            fc[f_idx, 'y_shift'] = float(res[1])
            fc[f_idx, 'sat_azimuth_from_fp_shift'] = float(azimuth_from_vector(res))
            if proposal_height_tag and proposal_height_tag in fc.columns:
                fc[f_idx, 'sat_elevation_from_fp_shift'] = float(elevation_from_vector(res,
                                                                                       fc[f_idx, proposal_height_tag]))
            if angles:
                fc[f_idx, 'fp_shift_height'] = float(height_from_vector(res, angles['sat_elevation']))

        except Exception as e:
            if failsafe:
                fc[f_idx, 'x_shift'] = 0
                fc[f_idx, 'y_shift'] = 0
                fc[f_idx, 'sat_azimuth_from_fp_shift'] = 0
                fc[f_idx, 'fp_shift_height'] = 0
                logger.opt(exception=True).warning(e)
                continue
            else:
                raise e
    return fc


def predict_heights_with_embeddings(bc: BandCollection,
                                    fc: FeatureCollection,
                                    model: Callable[[np.ndarray], np.ndarray],
                                    sample_size: Tuple[int, int] = DEFAULT_CROP_SIZE,
                                    emb_height_tag: str = defaults.EMB_HEIGHT_TAG,
                                    emb_confidence_tag: str = defaults.EMB_CONFIDENCE_TAG,
                                    reference_height_tag: str = defaults.SW_HEIGHT_TAG,
                                    split_rule: Callable = get_split_rule(defaults.SW_CONFIDENCE_TAG,
                                                                          defaults.SW_CONFIDENCE_THRESHOLD),
                                    min_proportion_of_references: float = MIN_PROPORTION_OF_REFERENCES,
                                    undefined_result: float = defaults.UNDEFINED_HEIGHT,
                                    failsafe: bool = True,
                                    n_neighbours: int = DEFAULT_N_NEIGHBOURS,
                                    angles: Optional[dict] = None,
                                    buffer: float = DEFAULT_CROP_BUFFER,
                                    proposal_height_tag: Optional[str] = None,
                                    default_proposal_height: float = DEFAULT_PROPOSAL_HEIGHT,
                                    subtract_neighbours: bool = False,
                                    negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
                                    height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
                                    padding_value: int = PADDING_VALUE,
                                    method: str = 'crop_by_mask') -> FeatureCollection:
    """
    Predicts building heights for every building (Feature) in fc by proximity of embeddings and writes
    result into corresponding feature property. Input fc must have some reference Features with known heights
    Args:
        bc: input BandCollection
        fc: input FeatureCollection, must have same crs, will be modified in-place
        model: Callable receives image as (C, W, H) np.ndarray, returns np.array of size (emb_len, )
        sample_size: image size appropriate for the model, (H, W)
        emb_height_tag: name of the property to write the result to
        emb_confidence_tag: name of the property to write confidence score to
        reference_height_tag: name of the property, containing know heights for some buildings
        split_rule: Callable to separate reference and target features, receives Feature, returns True if
                    this Feature belongs to reference, e.g. lambda x: x.properties['_height_confidence'] > 0.2
        min_proportion_of_references: minimum proportion of reference features in input, if actual proportion
                                      is smaller, exception is raised
        undefined_result: value to write into target_property in case of exception
        failsafe: if an instance fails, continues with default height
        n_neighbours: number of neighbours to pass to get_height()
        angles: dict meta angles as dict
        buffer: float value to inflate the mask
        proposal_height_tag: float tag in feature properties with proposal height
        default_proposal_height: float
        subtract_neighbours: bool =False,
        negative_buffer: float = DEFAULT_NEGATIVE_BUFFER,
        height_mul: float = DEFAULT_HEIGHT_MULTIPLIER,
        padding_value: int = PADDING_VALUE,
        method: 'crop_by_mask' or 'add_mask'
    Returns:
        Modified FeatureCollection
    """
    # We will have problem with empty fc later, so return it immediately in this case
    if len(fc) == 0:
        return fc
    reference_embs = list()
    heights = list()
    target_embs = dict()

    for f_idx in range(len(fc)):
        try:
            mask = get_feature_mask(fc, f_idx, angles=angles, buffer=buffer, proposal_height_tag=proposal_height_tag,
                                    default_proposal_height=default_proposal_height,
                                    subtract_neighbours=subtract_neighbours,
                                    negative_buffer=negative_buffer, height_mul=height_mul)
            if method == 'crop_by_mask':
                sample = crop_by_mask(bc, mask, sample_size=sample_size, padding_value=padding_value)
            elif method == 'add_mask':
                sample = crop_by_window_with_mask(bc, mask, sample_size=sample_size)
            else:
                raise ValueError(f'Unknown method {method}, must be "crop_by_mask" or "add_mask"')
            emb = model(sample)
            while emb.ndim > 1:
                emb = emb[0]
            if emb.ndim < 1:
                emb = np.array((emb,))
            if split_rule(fc[f_idx]):
                reference_embs.append(emb)
                heights.append(float(fc[f_idx, reference_height_tag]))
                fc[f_idx, emb_height_tag] = float(fc[f_idx, reference_height_tag])
                fc[f_idx, emb_confidence_tag] = 1.
            else:
                target_embs[f_idx] = emb

        except Exception as e:
            if failsafe:
                logger.opt(exception=True).warning(e)
                fc[f_idx, emb_height_tag] = undefined_result
                fc[f_idx, emb_confidence_tag] = 0.
                continue
            else:
                raise e

    if len(reference_embs) < min_proportion_of_references * len(fc):
        logger.warning(f'Not enough reference buildings, {len(reference_embs)} from {len(fc)}')
        for i in target_embs.keys():
            fc[i, emb_height_tag] = undefined_result
            fc[i, emb_confidence_tag] = 0.
        return fc

    reference_embs = np.stack(reference_embs)
    heights = np.array(heights)
    for i, emb in target_embs.items():
        try:
            pred, score = get_height_from_embeddings(emb, reference_embs, heights, n_neighbours=n_neighbours)
            if pred <= 0:
                logger.warning(f'Got embedding estimation value {pred} <= 0')
                raise ValueError
            fc[i, emb_height_tag] = np.round(pred, defaults.HEIGHT_DECIMALS)
            fc[i, emb_confidence_tag] = np.round(1/score, defaults.CONFIDENCE_DECIMALS)
        except Exception as e:
            if failsafe:
                logger.opt(exception=True).warning(e)
                fc[i, emb_height_tag] = undefined_result
                fc[i, emb_confidence_tag] = 0.
                continue
            else:
                raise e

    return fc
