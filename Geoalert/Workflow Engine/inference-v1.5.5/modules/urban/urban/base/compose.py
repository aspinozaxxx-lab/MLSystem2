import os
import time
import yaml
import warnings
import datetime as dt
from loguru import logger
from typing import List, Optional, Dict, Mapping, Any, Union, Sequence
from pathlib import Path
from we_queue_client.utils import log_time
from .registry_object import RegistryObject
from .parser import parse_config
from .brick import Brick
from pydantic import BaseModel, Field


class ConfigValidationError(ValueError):
    pass


class Block(BaseModel):
    name: str
    optional: bool
    inputs: List[str]
    outputs: List[str]
    bricks: List[dict] = Field(default_factory=list)


class Compose(RegistryObject):
    bricks: Sequence[Brick] = Field(default_factory=list)  #TODO: optional?
    inputs: Sequence[str] = Field(default_factory=list)
    outputs: Sequence[str] = Field(default_factory=list)
    namespace: Optional[str] = None

    @log_time(level='INFO', log_args=False, log_kwargs=False)
    def __call__(self, path: Union[Path, str], resume: int = 0):
        if self.namespace is not None:
            path = os.path.join(path, self.namespace)

        if not os.path.exists(path) or not os.path.isdir(path):
            raise ValueError(f"Folder does not exist: {path}")
        if not all(inp in os.listdir(path) for inp in self.inputs):
            raise ValueError(f"Inputs are missing in the folder: expected {self.inputs}, got {os.listdir(path)}")

        for i, brick in enumerate(self.bricks):
            if i < resume:
                continue

            index = str(i).zfill(2)
            logger.info("Executing: [{}] {}".format(index, repr(brick)))
            start_time = time.time()
            brick(path)
            duration_min = int((time.time() - start_time) // 60)
            duration_sec = int((time.time() - start_time) % 60)
            logger.info("Completed: [{}] {} in {}:{}.".format(
                index,
                brick.name,
                str(duration_min).zfill(2),
                str(duration_sec).zfill(2),
            ))
        return self.outputs

    def get_config(self) -> dict:
        config = super().get_config()
        config['bricks'] = [b.get_config() for b in self.bricks]
        return config

    @staticmethod
    def validate_config(config: dict, enable_blocks: Optional[dict]) -> None:
        if 'bricks' in config.keys():
            if enable_blocks:
                raise ConfigValidationError(f"Config without blocks should not "
                                            f"be provided with enable_blocks param, got {enable_blocks}")
            if 'blocks' in config.keys():
                raise ConfigValidationError(f"Config should contain either list of Bricks, or list of Blocks, not both")

            Compose._validate_plain_config(config['bricks'])
        elif 'blocks' in config.keys():
            Compose._validate_blocked_config(config, enable_blocks)
        else:
            raise ConfigValidationError("Config should contain either list of Bricks, or list of Blocks")

    @staticmethod
    def _validate_blocked_config(config, enable_blocks):
        blocks = [Block(**block_config) for block_config in config['blocks']]

        existing_files = set(config['inputs'])
        outputs = set(config['outputs'])
        optional_blocks = set()
        required_blocks = set()
        for block in blocks:
            if block.name in optional_blocks.union(required_blocks):
                raise ConfigValidationError(f"All blocks in config must have unique names. "
                                            f"The name {block.name} repeats")

            if not set(block.inputs) <= existing_files:
                raise ConfigValidationError(f'Block {block.name} has inputs {set(block.inputs).difference(existing_files)} '
                                            f'which are not present at the start: {existing_files}')
            if block.optional:
                optional_blocks.add(block.name)
                if not set(block.outputs) <= existing_files:
                    raise ConfigValidationError(f'Optional block must not produce new outputs. Block {block.name} '
                                                f'has outputs {set(block.outputs).difference(existing_files)}'
                                                f'which are not present at the start of the block')
            else:
                required_blocks.add(block.name)
                existing_files = existing_files.union(set(block.outputs))
            Compose._validate_plain_config(block.bricks)

        if not outputs <= existing_files:
            raise ConfigValidationError(f"Outputs {outputs.difference(existing_files)} "
                                        f"are not produced by any of the blocks")

        # If optional blocks are not enabled/disabled explicitly, the processing fails,
        # as it is caller's responsibility to manage blocks
        if not isinstance(enable_blocks, Dict) \
           or not set(enable_blocks.keys()) >= optional_blocks \
           or not all(isinstance(val, bool) for val in enable_blocks.values()):
            raise ConfigValidationError(f"enable_blocks param is {enable_blocks}. "
                                        f"It must contain True or False "
                                        f"for every optional block: {optional_blocks}")
        if not required_blocks:
            raise ConfigValidationError("Config must contain at least one required block!")

    @staticmethod
    def _validate_plain_config(bricks):
        """ This code adds VectorizeMasks bricks to the pipeline if ModelBrick has 'vectorize' property
        i = 0
        while i < len(bricks):
            if bricks[i]['_class'] == 'Segmentation' and bricks[i].get('vectorize'):
                logger.warning('Vectorization inside Segmentation brick is deprecated, adding VectorizeMasks')
                bricks.insert(i+1, {'_class': 'VectorizeMasks',
                                    'input_rasters': [f for f in bricks[i]['output_labels'] if f]})
                bricks[i].pop('vectorize')
                i += 1
            i += 1"""
        pass

    @staticmethod
    def read_bricks(brick_configs: Sequence[dict]):
        return [Brick.from_config(brick_config) for brick_config in brick_configs]

    @staticmethod
    def add_block(block_config: Dict, enable_blocks: Mapping[str, bool] = None):
        block = Block(**block_config)
        is_disabled = block.optional and not enable_blocks[block.name]
        if not is_disabled:
            logger.info("Adding block: {}".format(block.name))
            return Compose.read_bricks(block.bricks)
        else:
            logger.info("Skipping block: {}".format(block.name))
            return []

    @staticmethod
    def from_config(config,
                    enable_blocks: Optional[Dict[str, bool]] = None):
        Compose.validate_config(config, enable_blocks)
        block_configs = config.get('blocks', None)

        if block_configs is None:
            bricks = Compose.read_bricks(config['bricks'])
        else:
            bricks = []
            for block_config in block_configs:
                bricks += Compose.add_block(block_config, enable_blocks)

        return Compose(bricks=bricks,
                       inputs=config.get('inputs', []),
                       outputs=config.get('outputs', []),
                       namespace=config.get('namespace', None))

    def save(self, path, overwrite=True):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)

        import urban

        pipeline = dict(
            config=self.get_config(),
            version=urban.__version__,
            date=str(dt.datetime.now().date())
        )

        with open(path, 'w') as f:
            yaml.safe_dump(pipeline, f, encoding='utf-8', indent=4)

    @staticmethod
    def load(path, enable_blocks: Optional[Dict[str, bool]] = None, pipeline_params: Optional[Dict[str, Any]] = None):
        pipeline = parse_config(path, pipeline_params=pipeline_params)

        import urban
        current_version = urban.__version__
        pipeline_version = pipeline["version"]

        if current_version != pipeline_version:
            warnings.warn(
                (
                    'Version mismatch. '
                    'Current version - {}; config version - {}.'
                    'This may cause unexpected results.'
                ).format(current_version, pipeline_version)
            )

        return Compose.from_config(pipeline["config"], enable_blocks=enable_blocks)
