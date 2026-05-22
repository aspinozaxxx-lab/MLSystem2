# based on https://github.com/manaakiwhenua/pycrown/tree/3d8db4e03d82d5846ff68fc5fa9c18803d823464/pycrown
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import median_filter, maximum_filter, center_of_mass
from skimage.segmentation import watershed
from numba import jit, float32, int32, float_


@jit(nopython=True, nogil=True, parallel=False)
def get_neighbourhood(radius):
    """ creates list of row and column coordinates for circular indexing around
    a central pixel and for different distances from the centre

    Parameters
    ----------
    radius :    int
                radius of circular kernel

    Returns
    -------
    ndarray
        array of column coordinates _relative_ to the central pixel
    ndarray
        array of row coordinates _relative_ to the central pixel
    ndarray
        indices for splitting the array of row/column coordinates into
        different distances from the centre
    """
    # build a circular kernel
    xy = np.arange(-radius, radius + 1).reshape(radius * 2 + 1, 1)
    kernel = xy ** 2 + xy.reshape(1, radius * 2 + 1) ** 2

    # numba v0.39 doesn't support np.unique, so use a workaround
    sfkernel = np.sort(kernel.flatten())
    unique = list(sfkernel[:1])
    for x in sfkernel:
        if x != unique[-1]:
            unique.append(x)

    nums = unique[1:]
    start = 1
    for num in range(len(nums)):
        if nums[num] >= radius ** 2:
            continue
        n1, n0 = np.where(kernel == nums[num])
        if start:
            neighbours_x = list(n1.astype(np.int32))
            neighbours_y = list(n0.astype(np.int32))
            breaks = [len(n0)]
            start = 0
        else:
            neighbours_x += list(n1.astype(np.int32))
            neighbours_y += list(n0.astype(np.int32))
            breaks.append(len(n0))
    breaks = np.array(breaks, dtype=np.int32)
    neighbours_x = np.array(neighbours_x, dtype=np.int32) - radius
    neighbours_y = np.array(neighbours_y, dtype=np.int32) - radius
    return neighbours_x, neighbours_y, breaks


@jit(int32[:, :](float32[:, :], int32[:, :], float_, float_, float_, float_),
     nopython=True, nogil=True, parallel=False)
def _crown_dalponteCIRC(Chm, Trees, th_seed, th_crown, th_tree, max_crown):
    """
    Crown delineation based on Dalponte and Coomes (2016) and
    lidR R-package (https://github.com/Jean-Romain/lidR/)
    In contrast to the moving window growing scheme from the original algorithm
    this code implements a circular region growing around the tree top which
    leads to smoother crown patterns and speeds up the calculation by one order
    of magnitude

    Parameters
    ----------
    Chm :       ndarray
                Canopy height model as n x m raster
    Trees :     ndarray
                Tree top pixel coordinates as 2xn ndarray
    th_tree :   float
                Threshold below which a pixel cannot be a tree. Default 2
    th_seed :   float
                Growing threshold 1. A pixel is added to a region if its height
                is greater than the tree height multiplied by this value. It
                should be between 0 and 1. Default 0.45
    th_crown :  float
                Growing threshold 2. A pixel is added to a region if its height
                is greater than the current mean height of the region
                multiplied by this value. It should be between 0 and 1.
                Default 0.55.
    max_crown : float
                Maximum value of the crown diameter of a detected tree (in
                pixels). Default 10

    Returns
    -------
    ndarray
        Raster of tree crowns
    """

    ntops = Trees.shape[1]
    npixel = np.ones(ntops)
    tidx_x, tidx_y = Trees[0], Trees[1]
    Crowns = np.zeros_like(Chm, dtype=np.int32)
    nrows = Chm.shape[0]
    ncols = Chm.shape[1]
    sum_height = np.zeros(ntops)
    for i in range(ntops):
        Crowns[tidx_y[i], tidx_x[i]] = i + 1
        sum_height[i] = Chm[tidx_y[i], tidx_x[i]]

    tree_idx = np.arange(ntops)
    neighbours = np.zeros((4, 2)).astype(np.int32)

    # Create the circular look-up indices
    neighbours_x, neighbours_y, breaks = get_neighbourhood(int(max_crown))

    step = 0
    for n_neighbours in breaks:
        grown = False
        for tidx in tree_idx:

            # Pixel coordinates of current seed
            seed_y = tidx_y[tidx]
            seed_x = tidx_x[tidx]
            # Seed height
            seed_h = Chm[seed_y, seed_x]
            # Mean height of the crown
            mh_crown = sum_height[tidx] / npixel[tidx]

            # Go through neighbourhood
            for n in range(n_neighbours):
                # Pixel coordinates of current neighbour
                nb_x = seed_x + neighbours_x[step + n]
                nb_y = seed_y + neighbours_y[step + n]

                # avoid out-of-bounds exceptions
                if nb_x < 1 or nb_x > ncols - 2 or nb_y < 1 or nb_y > nrows - 2:
                    continue

                # Neighbour height
                nb_h = Chm[nb_y, nb_x]

                # Perform different checks:
                # 1. Neighbour height is above minimum threshold
                # 2. Neighbour does not belong to other crown
                # 3. Neighbour height is above threshold 1
                # 4. Neighbour height is above threshold 2
                # 5. Neighbour height below treetop+5%
                # 7. Neighbour is not too far from the tree top (x-dir)
                # 8. Neighbour is not too far from the tree top (y-dir)
                if nb_h > th_tree and \
                        not Crowns[nb_y, nb_x] and \
                        nb_h > (seed_h * th_seed) and \
                        nb_h > (mh_crown * th_crown) and \
                        nb_h <= (seed_h * 1.05) and \
                        abs(seed_x - nb_x) < max_crown and \
                        abs(seed_y - nb_y) < max_crown:

                    # Positions of the 4 neighbours
                    neighbours[0, 0] = nb_y - 1
                    neighbours[0, 1] = nb_x
                    neighbours[1, 0] = nb_y
                    neighbours[1, 1] = nb_x - 1
                    neighbours[2, 0] = nb_y
                    neighbours[2, 1] = nb_x + 1
                    neighbours[3, 0] = nb_y + 1
                    neighbours[3, 1] = nb_x

                    for j in range(4):

                        # if neighbours[j, 0] <= 0 or \
                        #     neighbours[j, 0] >= nrows or \
                        #     neighbours[j, 1] <= 0 or \
                        #     neighbours[j, 1] >= ncols:
                        #     continue

                        # Check that pixel is connected to current crown
                        if Crowns[neighbours[j, 0], neighbours[j, 1]] == tidx + 1:
                            # If all conditions are met, add neighbour to crown
                            Crowns[nb_y, nb_x] = tidx + 1
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
                            break

        step += n_neighbours
        if not grown:
            break

    return Crowns


def get_kernel(radius=5, circular=False):
    """
    Returns a block or disc-shaped filter kernel with given radius

    Parameters
    ----------
    radius :    int, optional
                radius of the filter kernel
    circular :  bool, optional
                set to True for disc-shaped filter kernel, block otherwise

    Returns
    -------
    ndarray
        filter kernel
    """

    if circular:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        return x ** 2 + y ** 2 <= radius ** 2
    else:
        return np.ones((int(radius), int(radius)))


def smooth_raster(raster, ws=5, circular=False):
    """
    Smooth a raster with a median filter

    Parameters
    ----------
    raster :      ndarray
                  raster to be smoothed
    ws :          int
                  window size of smoothing filter
    circular :    bool, optional
                  set to True for disc-shaped filter kernel, block otherwise

    Returns
    -------
    ndarray
        smoothed raster
    """

    return median_filter(raster, footprint=get_kernel(ws, circular=circular))


def tree_detection(raster, ws=5, hmin=20):
    """
    Detect individual trees from CHM raster based on a maximum filter.

    Parameters
    ----------
    raster :        ndarray
                    raster of height values (e.g., CHM)
    ws :            float
                    moving window size (in pixels) to detect the local maxima
    hmin :          float
                    Minimum height (in meters) of a tree. Threshold below which a pixel
                    or a point cannot be a local maxima
    Returns
    -------
    tuple (ndarray, int, ndarray)
        ndarray of tree top pixel coordinates
        number of detected trees
        ndarray of tree top coordinates
    """

    # Maximum filter to find local peaks
    raster_maximum = maximum_filter(raster, footprint=get_kernel(ws, circular=True))
    tree_maxima = raster == raster_maximum

    # remove tree tops lower than minimum height
    tree_maxima[raster < hmin] = 0

    # label each tree
    tree_markers, num_objects = ndimage.label(tree_maxima)

    if num_objects == 0:
        return tree_markers, num_objects, None

    # if canopy height is the same for multiple pixels, place the tree top in the center of mass of the pixel bounds
    yx = np.array(center_of_mass(raster, tree_markers, range(1, num_objects + 1)), dtype=np.float32) + 0.5
    xy = np.array((yx[:, 1], yx[:, 0]))
    return tree_markers, num_objects, xy


def crown_delineation_dalponte(chm, tree_coords, th_seed=0.45, th_crown=0.55, th_tree=2, max_crown=10):
    """
    Function calling external crown delineation algorithm based on the Dalponte algorithm

    Parameters
    ----------
    chm :   ndarray
                 raster of height values (e.g., CHM)
    tree_coords :      ndarray
                 2xn array of tree top pixel coordinates (starting points for crown delineation)
    th_seed :    float
                 factor 1 for minimum height of tree crown
    th_crown :   float
                 factor 2 for minimum height of tree crown
    th_tree :    float
                 minimum height of tree seed (in meters)
    max_crown :  float
                 maximum radius of tree crown (in pixels)

    Returns
    -------
    ndarray
        raster of individual tree crowns
    """
    crowns = _crown_dalponteCIRC(chm.astype(np.float32),
                                 tree_coords.astype(np.int32),
                                 float(th_seed),
                                 float(th_crown),
                                 float(th_tree),
                                 float(max_crown))

    return crowns


def crown_delineation_watershed(chm, tree_markers, th_tree=2):
    """
    Function calling external crown delineation algorithm based on the watershed algorithm

    Parameters
    ----------
    chm :   ndarray
                 raster of height values (e.g., CHM)

    tree_markers :     ndarray
                2d array of tree top pixel coordinates
    th_tree :    float
                 minimum height of tree seed (in meters)

    Returns
    -------
    ndarray
        raster of individual tree crowns
    """
    # TODO: add max_crown parameter
    crowns = watershed(-chm, tree_markers, mask=chm >= th_tree)

    return crowns


def get_crown_delineation_fn(algorithm, resolution, ws_smooth, ws_local_maxima, min_height_local_maxima,
                             min_height_for_tree, th_seed, th_crown, max_crown, min_pixel_number, max_pixel_number):
    # min_pixel_number < ws_local_maxima_in_pixels < max_pixel_number
    ws_local_maxima_in_pixels = int(min(max(ws_local_maxima / resolution, min_pixel_number), max_pixel_number))
    # `max_crown` is radius in pixels so `min_pixel_number` and `max_pixel_number` are divided by 2
    # min_pixel_number/2 < max_crown_in_pixels < max_pixel_number/2
    max_crown_in_pixels = int(min(max(max_crown / resolution, min_pixel_number / 2), max_pixel_number / 2))

    def crown_delineation_fn(sample):
        chm = sample[0]
        if ws_smooth > 0:
            chm = smooth_raster(chm, ws=ws_smooth, circular=False)

        tree_markers, num_objects, tree_coords = tree_detection(chm,
                                                                ws=ws_local_maxima_in_pixels,
                                                                hmin=min_height_local_maxima)
        if num_objects == 0:
            return np.zeros(0), np.zeros(0)

        crowns = None
        if algorithm == 'dalponteCIRC':
            crowns = crown_delineation_dalponte(chm,
                                                tree_coords=tree_coords,
                                                th_seed=th_seed,
                                                th_crown=th_crown,
                                                th_tree=min_height_for_tree,
                                                max_crown=max_crown_in_pixels)
        elif algorithm == 'watershed':
            crowns = crown_delineation_watershed(chm, tree_markers, th_tree=min_height_for_tree)

        masks = []
        properties = []
        for i in range(1, crowns.max() + 1):
            mask = (crowns == i).astype(np.uint8)
            max_height = int(round(sample[0][mask.astype(bool)].max().item()))
            if max_height >= min_height_for_tree:
                masks.append(mask)
                properties.append({'max_height': max_height})

        return masks, properties

    return crown_delineation_fn
