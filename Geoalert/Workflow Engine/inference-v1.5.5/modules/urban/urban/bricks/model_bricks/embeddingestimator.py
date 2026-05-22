from ...base import defaults
from ...functional import io
from ...functional import regression as R
from typing import List, Optional, Tuple
from .modelbrick import ModelBrick
from pydantic import Field


class EmbeddingEstimator(ModelBrick):
    """Estimates heights by proximity in embedding space using reference values

    Args:
        adapter: urban.ModelAdapter or dict config for constructing urban.ModelAdapter
        input_rasters (List[str]): filenames of input bands
        input_vector str: filename of input featurecollection containing masks
        output_vector str: filename of output featurecollection, if None, same as input_vector
        height_tag: name of Feature property to write the result to
        emb_confidence_tag: name of Feature property to write confidence to
        reference_height_tag: name of Feature property with reference height
        reference_confidence_tag: name of Feature property with reference_confidence
        split_threshold: threshold for reference_confidence_property to split buildings into references and targets
        min_proportion_of_references: minimum required proportion of reference buildings
        buffer: margin around geometry when it is cropped from raster
        undefined_result: value to return if height estimation fails
        n_neighbours: numer of neighbours in embedding space for height estimation
        failsafe: if an instance fails, continues with default height
    """
    input_rasters: List[str]
    input_vector: str
    output_vector: str = Field(default=None)
    sample_size: Tuple[int, int] = Field(default=defaults.DEFAULT_SAMPLE_SIZE)
    height_tag: str = Field(default=defaults.REGR_HEIGHT_TAG)
    undefined_result: float = Field(default=defaults.UNDEFINED_HEIGHT)
    failsafe: bool = Field(default=True)
    meta: Optional[str] = Field(default=None)
    buffer: float = Field(default=R.DEFAULT_CROP_BUFFER)
    proposal_height_tag: Optional[str] = Field(default=None)
    default_proposal_height: float = Field(default=20)
    subtract_neighbours: bool = Field(default=False)
    negative_buffer: float = Field(default=R.DEFAULT_NEGATIVE_BUFFER)
    height_mul: float = Field(default=R.DEFAULT_HEIGHT_MULTIPLIER)
    padding_value: int = Field(default=R.PADDING_VALUE)
    emb_confidence_tag: str = defaults.EMB_CONFIDENCE_TAG
    reference_height_tag: str = defaults.SW_HEIGHT_TAG
    reference_confidence_tag: str = defaults.SW_CONFIDENCE_TAG
    split_threshold: float = defaults.SW_CONFIDENCE_THRESHOLD
    min_proportion_of_references: float = R.MIN_PROPORTION_OF_REFERENCES
    n_neighbours: int = R.DEFAULT_N_NEIGHBOURS

    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)
        fc = io.read_fc(path, self.input_vector)
        if fc.crs != bc.crs:
            fc = fc.to_crs(bc.crs)
        split_rule = R.get_split_rule(self.reference_confidence_tag, self.split_threshold)
        fc = R.predict_heights_with_embeddings(bc=bc,
                                               fc=fc,
                                               model=self.adapter,
                                               sample_size=self.sample_size,
                                               emb_height_tag=self.height_tag,
                                               emb_confidence_tag=self.emb_confidence_tag,
                                               reference_height_tag=self.reference_height_tag,
                                               split_rule=split_rule,
                                               min_proportion_of_references=self.min_proportion_of_references,
                                               undefined_result=self.undefined_result,
                                               buffer=self.buffer,
                                               n_neighbours=self.n_neighbours,
                                               failsafe=self.failsafe)

        out_name = self.output_vector
        io.save_fc(fc, path, out_name)
