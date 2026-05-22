import yaml
from urban import Brick

config_text = """
brick_class: UnifiedVectorProcessing
input: '100'
output: '200'
bricks:
    -  brick_class: SimplifyAsShape
       shape_type: RECTANGLE
       min_area: 100
       min_iou: 0.7
       max_hausdorff: 5.
       rect_neg_buffer: 0.1
       iou_confidence_tag: iou_confidence
    -  brick_class: SimplifyAsShape
       shape_type: CIRCLE
    -  brick_class: Simplify
       input: input
       output: output
       crs: 'utm'
       rate: 2.0     
"""

config_output_text = """
brick_class: UnifiedVectorProcessing
input: '100'
output: '200'
crs: 'utm'
bricks:
    -  brick_class: SimplifyAsShape
       input: '100'
       output: '200'
       shape_type: RECTANGLE
       min_area: 100
       min_iou: 0.7
       max_hausdorff: 5.0
       rect_neg_buffer: 0.1
       verbose: False
       iou_confidence_tag: iou_confidence
    -  brick_class: SimplifyAsShape
       input: '100'
       output: '200'
       shape_type: CIRCLE
       min_area: 0.0
       min_iou: 0.8
       max_hausdorff: 10.0
       verbose: False
       iou_confidence_tag: Null
    -  brick_class: Simplify
       input: '100'
       output: '200'
       crs: 'utm'
       rate: 2.0     
"""


def test_parse_simplification_config():
    config_dict = yaml.safe_load(config_text)
    brick = Brick.from_config(config_dict)
    assert len(brick.bricks) == 3
    assert brick._bricks[0].min_area == 100
    assert brick._bricks[0].min_iou == 0.7
    assert brick._bricks[0].max_hausdorff == 5.
    assert brick._bricks[0].rect_neg_buffer == 0.1
    assert brick._bricks[0].shape_type == "RECTANGLE"
    assert brick._bricks[1].shape_type == "CIRCLE"
    assert brick._bricks[2].rate == 2.0
    assert brick._bricks[2].crs == "utm"
    # override input/output
    assert brick._bricks[2].input == '100'
    assert brick._bricks[2].output == '200'
