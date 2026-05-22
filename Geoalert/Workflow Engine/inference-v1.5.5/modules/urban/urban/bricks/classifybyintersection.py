from ..base import Brick
from typing import List, Optional
from ..functional import io
from ..functional.classification import classify
from ..base.defaults import BUILDING_CLASS_TAG
from pydantic import Field


class ClassifyByIntersection(Brick):
    """Given semantic segmentation mask (with all features belongs to a single class)
     and multiclass segmentation mask (with same features classified), assigns class label to
     the first mask based on IoU
    Args:
        semantic_mask (str): filename of semantic mask fc
        class_mask (List[str]): filenames of masks for each class
        tag (str): tag for classification result,
        output: (str): filename of output, if None, same as input
    """
    semantic_mask: str
    class_mask: List[str]
    tag: str = Field(BUILDING_CLASS_TAG)
    output: Optional[str] = Field(None)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.output = self.output or self.semantic_mask

    def __call__(self, path):
        fc = io.read_fc(path, self.semantic_mask)
        classes_fc = {cls: io.read_fc(path, cls) for cls in self.class_mask}
        fc = classify(fc, classes_fc, tag=self.tag)
        io.save_fc(fc, path, self.output)
