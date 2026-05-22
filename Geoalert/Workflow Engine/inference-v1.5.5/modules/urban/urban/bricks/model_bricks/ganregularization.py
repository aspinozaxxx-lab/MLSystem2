from .segmentation import Segmentation
from ...functional.gan_regularization import regularization
from loguru import logger


class GANRegularization(Segmentation):
    """GAN regularization to improve segmentation masks"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.adapter.input_dtype != 'float32':
            logger.warning(f'Gan regularization input_dtype = {self.adapter.input_dtype}, '
                           f'but must be float32. Check your config')
            self.adapter.input_dtype = 'float32'
            self.adapter.set_preprocess_fn()
        if self.adapter.output_dtype != 'float32':
            logger.warning(f'Gan regularization output_dtype = {self.adapter.output_dtype}, '
                           f'but must be float32. Check your config')
            self.adapter.output_dtype = 'float32'
            self.adapter.set_postprocess_fn()

    def processing_fn(self, x):
        return regularization(x, self.adapter)
