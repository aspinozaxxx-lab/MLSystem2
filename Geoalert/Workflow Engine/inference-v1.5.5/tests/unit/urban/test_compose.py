import pytest
from urban import Compose
from urban.base.compose import Block, parse_config, ConfigValidationError
from pydantic import ValidationError


CONFIGS_PATH = 'tests/test_pipelines/compose_tests/'


def test_brick_pipeline():
    path = CONFIGS_PATH + 'brick_pipeline.yml'
    comp = Compose.load(path)

    assert comp.bricks[0].name == 'SplitRaster'
    assert comp.bricks[1].name == 'FilterSmallObjects'
    assert len(comp.bricks) == 2


def test_brick_pipeline_pass_params():
    path = CONFIGS_PATH + 'brick_pipeline_pass_params.yml'
    pipeline_params = {'input': 'rgb_new11',
                       'min_area': -55.0,
                       'text_threshold': -100,
                       'text_prompt2': 'text prompt new 21'}

    pipeline = parse_config(path, pipeline_params=pipeline_params)

    assert pipeline['config']['bricks'][0]['input'] == pipeline_params['input']
    assert pipeline['config']['bricks'][0]['min_area'] == pipeline_params['min_area']
    assert pipeline['config']['bricks'][0]['text_prompt2'] == pipeline_params['text_prompt2']
    assert pipeline['config']['bricks'][0]['input_ext'] == 'tif'
    assert pipeline['config']['bricks'][0]['box_threshold'] == -0.5
    assert pipeline['config']['bricks'][0]['text_threshold'] == 0.100
    assert pipeline['config']['bricks'][0]['model_name1'] == 'triton-text2box-model-02'
    assert pipeline['config']['bricks'][0]['model_name2'] == 'triton-building-model-03'
    assert pipeline['config']['bricks'][0]['text_prompt'] == 'red roof . car . 3 house'
    assert pipeline['config']['bricks'][0]['text_prompt14'] == 'building'


def test_block_pipeline_empty_enable_blocks():
    path = CONFIGS_PATH + 'block_pipeline.yml'
    with pytest.raises(ValueError, match='enable_blocks param is'):
        comp = Compose.load(path, enable_blocks={})


def test_block_pipeline_wo_enable_blocks():
    # Should raise exception
    path = CONFIGS_PATH + 'block_pipeline.yml'
    with pytest.raises(ValueError, match='enable_blocks param is None'):
        comp = Compose.load(path)


def test_block_pipeline_with_excessive_enabled_blocks():
    # enable value for required block is allowed
    path = CONFIGS_PATH + 'block_pipeline.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    comp = Compose.from_config(config, enable_blocks={'SimplifyAsShapeBlock': False, 'SplitRaster': True})
    assert comp.bricks[0].name == 'SplitRaster'
    assert comp.bricks[1].name == 'Segmentation'
    assert len(comp.bricks) == 2


def test_block_pipeline_should_not_include_disabled_block():
    path = CONFIGS_PATH + 'block_pipeline.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    comp = Compose.from_config(config, enable_blocks={'SimplifyAsShapeBlock': False})
    assert comp.bricks[0].name == 'SplitRaster'
    assert comp.bricks[1].name == 'Segmentation'
    assert len(comp.bricks) == 2


def test_block_pipeline_should_include_enabled_block():
    path = CONFIGS_PATH + 'block_pipeline.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    comp = Compose.from_config(config, enable_blocks={'SimplifyAsShapeBlock': True})
    assert comp.bricks[0].name == 'SplitRaster'
    assert comp.bricks[1].name == 'Segmentation'
    assert comp.bricks[2].name == 'SimplifyAsShape'
    assert len(comp.bricks) == 3


#  ==== block class init
def test_block_initialization_wrong_type():
    data = {'name': '1',
            'optional': True,
            'inputs': 'input.tif',
            'outputs': ['out.tif', 'out.geojson'],
            'bricks': []}
    with pytest.raises(ValidationError):
        block = Block(**data)


def test_block_correct_initialization():
    data = {'name': '1',
            'optional': True,
            'inputs': ['input.tif'],
            'outputs': ['out.tif', 'out.geojson'],
            'bricks': []}
    block = Block(**data)


#  ======


# ===== validation
def test_block_config_must_have_enough_inputs():
    path = CONFIGS_PATH + 'block_pipeline_bad_inputs.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    with pytest.raises(ConfigValidationError, match='inputs'):
        comp = Compose.from_config(config)


def test_block_config_must_have_enough_outputs():
    path = CONFIGS_PATH + 'block_pipeline_bad_outputs.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    with pytest.raises(ConfigValidationError, match='Outputs'):
        comp = Compose.from_config(config)


def test_block_config_must_have_required_bricks():
    path = CONFIGS_PATH + 'block_pipeline_only_optional.yml'
    pipeline = parse_config(path)
    config = pipeline['config']
    with pytest.raises(ConfigValidationError, match='Config must contain at least one required block'):
        comp = Compose.from_config(config, enable_blocks={'SampleBlock': True,
                                                          'SampleBlock2': False})


def test_block_config_optional_must_not_produce_new():
    path = CONFIGS_PATH + 'block_pipeline_optional_with_new_output.yaml'
    pipeline = parse_config(path)
    config = pipeline['config']
    with pytest.raises(ConfigValidationError, match='Optional block must not produce new outputs'):
        comp = Compose.from_config(config)


def test_block_config_blocks_must_have_different_names():
    path = CONFIGS_PATH + 'block_pipeline_same_names.yaml'
    pipeline = parse_config(path)
    config = pipeline['config']
    with pytest.raises(ConfigValidationError, match='All blocks in config must have unique names'):
        comp = Compose.from_config(config, enable_blocks={"SampleBlock": False})
