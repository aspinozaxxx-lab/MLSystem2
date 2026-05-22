import cv2
import numpy as np
from skimage.morphology import skeletonize


def separate_semantically_segmented_fields(raw_predictions, 
                                           kernel_small_ellipse: np.ndarray, 
                                           kernel_big_ellipse: np.ndarray, 
                                           area_filter: float, 
                                           boundaries_proba_thr: float = 0.5,
                                           opening_kernel_small_iterations: int = 3,
                                           opening_kernel_big_iterations: int = 1,
                                           dilate_iterations_sure_bg: int = 5,
                                           dilate_iterations_sure_fg_proba: int = 1,
                                           enhance_instances_deliniation: bool = True,
                                           ):
    """
    Takes a raster mask (2-band) from semantic segmentation model [boundaries_mask, fields_mask]
    and makes a binary masks with separated instances.
    
    This function applies all postprocessing steps at one to inference of semantic segmentation
    model.
    Those postprocessing steps are:
        * cleaning boundaries and fields mask
        * Skeletonization of boundaries
        * Probability subtraction and binarization
        * Watershed
        * Deleting too small fields
    
    Args:
        raw_predictions: numpy array, containing a 2 2-dimensional image
        kernel_small_ellipse: numpy array, Empirically selected kernel for opening boundaries and fields during
                                cleaning and dilation for sure_bg and skeletonized masks
        kernel_big_ellipse: numpy array, Empirically selected kernel for opening  fields during cleaning
        area_filter: Float, value for thresholding minimum area of fields instances.
        boundaries_proba_thr:  float, threshold for boundaries applied before subtraction of probability
        opening_kernel_small_iterations: int, Number of opening iterations for cleaning fields mask and for dilation
        opening_kernel_big_iterations: int, Number of opening iterations applied to fields mask
        dilate_iterations_sure_bg: int, Number of dilation iterations applied to opening mask to get sure background
        dilate_iterations_sure_fg_proba: int, number of dilation iterations applied to probability mask 
                                                (after subtraction of probabilities)
        enhance_instances_deliniation: bool, if True, then result segmentation mask will be eroded with smallest
                                        3x3 kernel, to avoid diagonal pixels. Need for vectorization.
    Returns:
        segm_res: List[np.ndarray(1, X, Y)], semantic segmentation mask.
    """
    # raw_predictions = reshape_as_image(np.squeeze(raw_model_res))
    # Enhance boundaries in straight areas of fields boundaries
    raw_predictions = raw_predictions
    raw_predictions = np.ma.transpose(raw_predictions, [1, 2, 0])
    enhanced_boundaries = enhance_linear_boundaries(raw_predictions[:, :, 0])
    # Get thresholds for boundaries
    thresh = ((raw_predictions[:, :, 1] - enhanced_boundaries > 0.5)*255).astype(np.uint8)

    # Cleaning from small "dots" and "splashes of fields"
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small_ellipse,
                               iterations=opening_kernel_small_iterations)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_big_ellipse, iterations=opening_kernel_big_iterations)
    
    # dilation of fields
    # 0 is background.
    sure_bg = cv2.dilate(opening, kernel_small_ellipse, iterations=dilate_iterations_sure_bg)
    
    # Boundaries where opening is zero and where proba of mask > boundaries_proba_thr
    # To filter boundary "noise"
    proba_boundaries_mask = (opening == 0)*(raw_predictions[:, :, 0] > boundaries_proba_thr)

    # Because model is trained to predict thick masks, we need to skeletonize boundaries mask to get
    # more fine boundaries. Boundaries mask is auxiliary mask, to better delineate fields
    # to get more consistent deliniation we additionally dilate skeletonized boundaries mask
    proba_boundaries_mask = cv2.dilate(skeletonize(proba_boundaries_mask > 0).astype(np.uint8)*255,
                                       kernel_small_ellipse,
                                       iterations=opening_kernel_small_iterations) / 255 > 0.5
    # Get fine probability in generated boundaries mask
    proba_boundaries = raw_predictions[:, :, 0] * proba_boundaries_mask
    
    # Probability of fields. To filter out noise in fields prediction we cut off it by using 
    # sure_background mask. This approach allows for more
    proba_fields = raw_predictions[:, :, 1] * (raw_predictions[:, :, 1] > 0.5) * (sure_bg > 0)
    # Get probability of fields minus boundaries probability
    proba = (proba_fields - proba_boundaries)
    # proba < 0 means that confidence in field is lower than confidence in boundaries
    proba[proba < 0] = 0
    # proba > 0 means confidence in field is more than confidence in boundaries
    proba[proba > 0] = 1
    # Prepare proba mask for watershed algorithm as sure_fg mask
    proba = (proba*255).astype(np.uint8)
    proba = cv2.morphologyEx(proba, cv2.MORPH_OPEN, kernel_small_ellipse,
                             iterations=dilate_iterations_sure_fg_proba)
    proba[proba > 0] = 255
    proba = np.repeat(np.expand_dims(proba, 2), 3, 2) 
    
    # SURE_FG_PROBA is mask for markers. For masks we need better deliniation
    sure_fg_proba = ((proba_fields - proba_boundaries > 0.5)*255).astype(np.uint8)
    # Clean from remaining small noise to avoid small holes in result.
    sure_fg_proba = cv2.morphologyEx(sure_fg_proba, cv2.MORPH_OPEN, kernel_small_ellipse, 
                                     iterations=dilate_iterations_sure_fg_proba)
    sure_fg_proba[sure_fg_proba>0] = 255
    # Get unknown regions for fill by watershed algorithm
    unknown = cv2.subtract(sure_bg, sure_fg_proba)

    ret, markers = cv2.connectedComponents(sure_fg_proba)
    markers = markers+1
    markers[unknown == 255] = 0
    # Remove tiny fields that are certainly noise
    for segm_index in np.unique(markers):
        if (markers == segm_index).sum() < area_filter:
            markers[markers == segm_index] = 1

    watershed_res = cv2.watershed(proba, markers)
    # tiny_kernel is needed to get tiny boundaries masks from instances
    # for better visual deliniation 
    tiny_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    segm_res = watershed_res.copy().astype(np.int64)
    # Set background and unknown to zero
    segm_res[segm_res <= 1] = 0
    boundaries_mask = np.zeros_like(watershed_res, dtype=np.int64)
    # For each instance create binary fine boundary mask 1 pixel wide. 
    for segm_index in np.sort(np.unique(segm_res))[1:]:
        instance_mask = (segm_res == segm_index).astype(np.uint8)*255
        boundaries_mask += ((cv2.dilate(instance_mask, tiny_kernel, iterations=1) > 0).astype(int) 
                            - (instance_mask > 0).astype(int))
        boundaries_mask[boundaries_mask > 0] = 255
        boundaries_mask[boundaries_mask < 0] = 0
    boundaries_mask[boundaries_mask > 0] = 255
    boundaries_mask = boundaries_mask
    segm_res = segm_res.astype(np.int64)
    segm_res[segm_res > 0] = 255
    # Remove boundaries pixels to avoid "shortwire" between instances
    if enhance_instances_deliniation:
        segm_res -= boundaries_mask
    segm_res[segm_res > 0] = 255
    segm_res[segm_res < 0] = 0
    segm_res = segm_res.astype(np.uint8)
    return [segm_res[np.newaxis, :, :], ]


def enhance_linear_boundaries(raw_boundaries, threshold=200, minLineLength=50, maxLineGap=5):
    """Enhances fields boundaries where boundaries forms straight dense line:
    Uses Hough transform and draw lines to enhance strong linear boundaries.

    Args:
        raw_boundaries: numpy array, containing a 2-dimensional image
        threshold: threshold for cv2.HoughLinesP
        minLineLength: minLineLength for cv2.HoughLinesP
        maxLineGap: maxLineGap for cv2.HoughLinesP
    
    Returns:
        enhanced_boundaries: numpy array, containing a 2-dimensional image
    """
    lines = cv2.HoughLinesP((raw_boundaries > 0.5).astype(np.uint8).copy()*255,
                            1, np.pi/180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    output = (raw_boundaries.copy()*255).astype(np.uint8)
    if not isinstance(lines, type(None)):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), [255, 255, 255], 2)
    return output.astype(np.float32)/255
