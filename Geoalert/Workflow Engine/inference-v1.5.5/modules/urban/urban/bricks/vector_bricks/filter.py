# TODO: inherit all Vector Filter bricks from this one

#from ..base.vector_processing_brick import VectorProcessingBrick, FeatureCollection
#from typing import Optional, Callable, Tuple


'''class Filter(VectorProcessingBrick):
    """Base class for FeatureCollection filter.
    Args:
        filter_fn: Function to apply to each row. Must accept pd.Series and return bool."""
    def __init__(self,
                 input: str,
                 filter_fn: Callable,
                 output: Optional[str] = None):
        super().__init__()
        self.input = input
        self.output = output or input
        self.filter_fn = filter_fn

    def process(self, fc: FeatureCollection) -> FeatureCollection:
        return fc.filter(self.filter_fn)'''
