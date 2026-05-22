from loguru import logger
from .registry import CLASS_REGISTRY
from .registry_object import RegistryObject
from ..functional import io
from gpdadapter import FeatureCollection
from typing import Union, Optional, Sequence, Any
from pydantic import Field
from pathlib import Path


class Brick(RegistryObject):
    """Brick should perform an atomic operation over a predefined set of input files and write a predefined set
     of output files without side effects"""
    # TODO: input and output filenames as properties
    def __call__(self, path: Union[str, Path]):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def from_config(config: dict):
        cls_name = config.pop('brick_class')
        logger.trace("Got name from config: {}".format(cls_name))
        try:
            cls = CLASS_REGISTRY[cls_name]
        except KeyError:
            raise ValueError(f"Brick name {cls_name} not found in registry")
        logger.trace("Got class from registry: {}".format(cls_name))
        brick = cls(**config)
        logger.trace("Initialized brick: {}".format(cls_name))
        return brick


class VectorProcessingBrick(Brick):
    """
    Base class for a brick that reads FeatureCollection, processes it and saves to a file.
    Manages read-write (in a specified CRS)
    The child class should implement the `process` method and have
    a `crs` attribute which means coordinate system to reproject FeatureCollection to
    `input` and `output` attributes
    """
    # TODO: input and output filenames as properties, additional inputs
    # read params
    input: str = Field(pattern=r'^[^\.\\\/]+$', description='Input file name, without extension')  # TODO: review pattern
    crs: Optional[str] = Field(default=None, description='CRS to reproject before processing')  # TODO: add CRS string validation
    make_valid_input: bool = Field(default=False, description="Apply 'make_valid()' before processing")
    dropna_input: bool = Field(default=False, description="Drop NaN or None values before processing")
    drop_empty_input: bool = Field(default=False, description="Drop empty geometries before processing")
    explode_input: bool = Field(default=False, description="Split collections into simple geometries before processing")
    remove_repeated_points_input: bool = Field(default=False, description="Remove repeated points before processing")
    input_geom_types: Union[str, Sequence[str], None] = Field(default=None,
                                                              description="Remove all other geometry types before processing")  # TODO: add geom type validation

    # write params
    output: Optional[str] = Field(None, pattern=r'^[^\.\\\/]+$',
                                  description='Output file name, without extension. If empty, same as input')
    hold_crs: bool = Field(default=False, description='Save output without reprojecting to 4326')
    make_valid_output: bool = Field(default=False, description="Apply 'make_valid()' after processing")
    dropna_output: bool = Field(default=False, description="Drop NaN or None values after processing")
    drop_empty_output: bool = Field(default=False, description="Drop empty geometries after processing")
    explode_output: bool = Field(default=False, description="Split collections into simple geometries after processing")
    remove_repeated_points_output: bool = Field(default=False, description="Remove repeated points after processing")
    output_geom_types: Union[str, Sequence[str], None] = Field(default=None,
                                                               description="Remove all other geometry types before processing")
    verbose: bool = Field(False)

    def model_post_init(self, __context: Any) -> None:
        self.output = self.output or self.input

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        raise NotImplementedError

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs=self.crs, make_valid=self.make_valid_input, dropna=self.dropna_input,
                        drop_empty=self.drop_empty_input, explode=self.explode_input,
                        remove_repeated_points=self.remove_repeated_points_input,
                        keep_only_geometry_types=self.input_geom_types)
        logger.debug(f"Processing {len(fc)} features")
        fc = self.process(fc)
        io.save_fc(fc, path, self.output, hold_crs=self.hold_crs, make_valid=self.make_valid_output,
                   dropna=self.dropna_output, drop_empty=self.drop_empty_output, explode=self.explode_output,
                   remove_repeated_points=self.remove_repeated_points_output,
                   keep_only_geometry_types=self.output_geom_types)


class PolygonProcessingBrick(VectorProcessingBrick):
    """Abstract superclass for Polygons-only fc processing with all the validation and UTM reprojecting
     enabled by default """
    crs: Optional[str] = Field(default='utm', description='CRS to reproject before processing')  # TODO: add CRS string validation
    make_valid_input: bool = Field(default=True, description="Apply 'make_valid()' before processing")
    dropna_input: bool = Field(default=True, description="Drop NaN or None values before processing")
    drop_empty_input: bool = Field(default=True, description="Drop empty geometries before processing")
    explode_input: bool = Field(default=True, description="Split collections into simple geometries before processing")
    remove_repeated_points_input: bool = Field(default=True, description="Remove repeated points before processing")
    input_geom_types: Union[str, Sequence[str], None] = Field(default='Polygon',
                                                              description="Remove all other geometry types before processing")  # TODO: add geom type validation

    # write params
    make_valid_output: bool = Field(default=True, description="Apply make_valid() after processing")
    dropna_output: bool = Field(default=True, description="Drop NaN or None values after processing")
    drop_empty_output: bool = Field(default=True, description="Drop empty geometries after processing")
    explode_output: bool = Field(default=True, description="Split collections into simple geometries after processing")
    remove_repeated_points_output: bool = Field(default=True, description="Remove repeated points after processing")
    output_geom_types: Union[str, Sequence[str], None] = Field(default='Polygon',
                                                               description="Remove all other geometry types before processing")