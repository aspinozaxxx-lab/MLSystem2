class Tag:
    # ----------------------------------------------------------------
    # Building postprocessing tags
    # ----------------------------------------------------------------

    BLD_MAIN_ANGLE = '_main_angle'

    # aligning tags
    BLD_ROAD_ID = '_road_id'
    BLD_ROAD_ANGLE = '_road_angle'
    BLD_CLUSTER_ID = '_cluster_id'
    BLD_ROTATION_ANGLE = '_rotation_angle'
    BLD_IS_ROTATED = '_is_rotated'

    # simplification tags
    BLD_SHAPE_TYPE = 'shape_type'
    BLD_IS_SIMPLIFIED = '_is_simplified'
    BLD_SIMPL_IOU = '_simplification_iou'
    BLD_SIMPL_HDF = '_simplification_hausdorff'


class Shape:
    UNKNOWN = 'UNK'
    RECTANGLE = 'RECTANGLE'
    CIRCLE = 'CIRCLE'
    LSHAPE = 'L-SHAPE'
    OSM = 'OSM'
    GRID_SNAP = 'GRID_SNAP'
    DYN_GRID = 'DYNAMIC_GRID'
