import numpy as np
from we_queue_client.utils import log_time
from glob import glob
from typing import Optional, Union, List, Sequence
from .segmentation import Segmentation
from ...functional import io
from ..adapters import ModelAdapter
from .postprocess import SeparateInstances
from pydantic import Field

ENCODER_FIXED_SHAPE = (1024, 1024)


class SAMAutoMaskGenerator(Segmentation):
    """Using a SAM model, generates masks for the entire image. Has encoder and decoder adapters.
    Decoder returns mask and mask without borders of objects.
    Args:
        adapter (Union[ModelAdapter, SerializedModel]): urban.ModelAdapter or dict config for
            constructing urban.ModelAdapter. Encoder
        dec_adapter (Union[ModelAdapter, SerializedModel]): urban.ModelAdapter or dict config for
            constructing urban.ModelAdapter. Decoder"""

    dec_adapter: ModelAdapter
    min_marker: float = Field(64)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.dec_adapter.preprocess_fn = None
        self.dec_adapter.postprocess_fn = None
        self.adapter.preprocess_fn = None
        self.adapter.postprocess_fn = None

    def processing_fn(self, sample):
        if not isinstance(sample, np.ndarray):
            sample = sample.numpy()
        img_emb = np.expand_dims(self.encode(sample).transpose(1, 2, 0), 0)
        dec_input_data = self.prepare_dec_input_data(np.expand_dims(sample.transpose(1, 2, 0), 0), img_emb)
        out = self.decode(dec_input_data.transpose(2, 0, 1))
        mask = self.postprocessors[0](out)
        return mask.astype(self._predictor.dst_dtype)

    @log_time(level='DEBUG', log_args=False, log_kwargs=False)
    def encode(self, data: np.ndarray) -> np.ndarray:
        return self.adapter(data)

    @log_time(level='DEBUG', log_args=False, log_kwargs=False)
    def decode(self, data: np.ndarray) -> np.ndarray:
        return self.dec_adapter(data)

    def prepare_dec_input_data(self, image: np.ndarray, img_emb: np.ndarray) -> np.ndarray:
        """
        Args:
            image: np.ndarray 1xHxWxC
            img_emb: np.ndarray 1x64x64x256
        Returns:
            input_data: np.ndarray MAX_HxMAX_Wx2
        """
        # 1024x1024
        img_emb = img_emb.reshape(*ENCODER_FIXED_SHAPE)

        # Create a binary `non_black` mask by finding the pixel locations in the image where
        # the pixel values are not nodata (0 if nodata is None)

        nodata = self.nodata if self.nodata else 0
        # HxW
        non_black = np.logical_not((image[0] == nodata).all(2))
        non_black = non_black.astype(img_emb.dtype)

        max_h = max(non_black.shape[0], img_emb.shape[0])
        max_w = max(non_black.shape[1], img_emb.shape[1])

        # Pad the non_black mask (non_black_pad) and the reshaped image embedding (img_emb_pad)
        # to match the maximum height and width calculated earlier. This ensures that both
        # arrays have the same dimensions.
        # MAX_HxMAX_W
        non_black_pad = np.pad(non_black, ((0, max_h - non_black.shape[0]), (0, max_w - non_black.shape[1])),
                               constant_values=None)
        # MAX_HxMAX_W
        img_emb_pad = np.pad(img_emb, ((0, max_h - img_emb.shape[0]), (0, max_w - img_emb.shape[1])),
                             constant_values=None)

        # MAX_HxMAX_Wx2
        input_data = np.stack((non_black_pad, img_emb_pad), 2)

        return input_data


class SAMPromptMaskGenerator(SAMAutoMaskGenerator):
    """Using a SAM model, generates masks for the input prompts. Has encoder and decoder adapters.
    Decoder returns mask and mask without borders of objects.
    Args:
        adapter (Union[ModelAdapter, SerializedModel]): urban.ModelAdapter or dict config for
            constructing urban.ModelAdapter. Encoder
        dec_adapter (Union[ModelAdapter, SerializedModel]): urban.ModelAdapter or dict config for
            constructing urban.ModelAdapter. Decoder"""

    input_prompts: str
    _all_prompts: Sequence = None
    processing_fn_use_block: bool = Field(True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._all_prompts = list()
        self._predictor.processing_fn_use_block = self.processing_fn_use_block

    @staticmethod
    def preprocess_prompts(all_prompts, block):
        """
        Cut prompts to block coordinates.
        """
        block_x = block['x']
        block_y = block['y']
        block_width = block['width']
        block_height = block['height']

        block_left_x = block_x
        block_top_y = block_y
        block_right_x = block_x + block_width
        block_bottom_y = block_y + block_height

        prompt_boxes = []
        prompt_points = []
        for prompts in all_prompts:
            for prompt in prompts:
                # boxes: 4, format: XYXY
                if prompt.shape[0] == 4:
                    bbox_left_x, bbox_top_y, bbox_right_x, bbox_bottom_y = prompt

                    # check if bbox is inside block
                    if bbox_right_x < block_left_x or bbox_left_x > block_right_x or bbox_bottom_y < block_top_y or bbox_top_y > block_bottom_y:
                        continue

                    # cut bbox to block edges
                    if bbox_left_x < block_left_x:
                        bbox_left_x = block_left_x
                    if bbox_top_y < block_top_y:
                        bbox_top_y = block_top_y
                    if bbox_right_x > block_right_x:
                        bbox_right_x = block_right_x
                    if bbox_bottom_y > block_bottom_y:
                        bbox_bottom_y = block_bottom_y

                    # coordinates relative to current window
                    bbox_left_x -= block_x
                    bbox_top_y -= block_y
                    bbox_right_x -= block_x
                    bbox_bottom_y -= block_y

                    prompt_boxes.append([bbox_left_x, bbox_top_y, bbox_right_x, bbox_bottom_y])
                # point: 3, format: XYL, where L - label {0, 1} of point (belongs point to object or not)
                elif prompt.shape[0] == 3:
                    point_x, point_y, label = prompt

                    # check if point is inside block
                    if point_x < block_left_x or point_x > block_right_x or point_y < block_top_y or point_y > block_bottom_y:
                        continue

                    # coordinates relative to current window
                    point_x -= block_x
                    point_y -= block_y

                    prompt_points.append([point_x, point_y, label])

        return prompt_boxes, prompt_points

    def processing_fn(self, sample, block):
        if not isinstance(sample, np.ndarray):
            sample = sample.numpy()

        prompt_boxes, prompt_points = self.preprocess_prompts(self._all_prompts, block)

        if len(prompt_points) > 0:
            raise NotImplementedError('Point prompts are not implemented yet')

        # skip window without box prompts
        if len(prompt_boxes) == 0:
            empty_raster = np.full(shape=(len(self.output_labels), block['height'], block['width']),
                                   fill_value=self.nodata,
                                   dtype=self._predictor.dst_dtype)
            return empty_raster
        else:
            img_emb = np.expand_dims(self.encode(sample).transpose(1, 2, 0), 0)
            dec_input_data = self.prepare_dec_input_data(np.expand_dims(sample.transpose(1, 2, 0), 0), img_emb)
            dec_box_input_data = (dec_input_data.transpose(2, 0, 1), np.asarray(prompt_boxes).astype(np.int64))
            out = self.decode(dec_box_input_data)
            mask = self.postprocessors[0](out)

            return mask.astype(self._predictor.dst_dtype)

    def __call__(self, path):
        bc = io.read_bc(path, self.input_rasters)

        if len(self.input_prompts) > 1:
            raise NotImplementedError(f'Only 1 prompt is supported! Got len of prompts = {len(self.input_prompts)}')

        # find files by regex
        prompt_paths = sorted(glob(f'{path}/boxes*.npy'))
        # if len(self.input_prompts) == 1:
        #     # find files by regex
        #     if '*' in self.input_prompts[0]:
        #         prompt_paths = sorted(glob(f'{path}/{self.input_prompts[0]}'))
        #     else:
        #         prompt_paths = [f'{path}/{self.input_prompts[0]}']
        # else:
        #     prompt_paths = []
        #     for prompt_name in self.input_prompts:
        #         prompt_paths.append(f'{path}/{prompt_name}')

        for prompt_path in prompt_paths:
            prompt_data = np.load(prompt_path)
            # boxes: np.ndarray Nx4, N - number of boxes, format: XYXY
            # points: np.ndarray Mx3, M - number of points, format: XYL, where
            # L - label {0, 1} of point (belongs point to object or not)
            if prompt_data.shape[1] == 4 or prompt_data.shape[1] == 3:
                self._all_prompts.append(prompt_data)
            else:
                raise ValueError(f'Prompt with shape "{prompt_data.shape}" is not supported!')

        if self.crs is not None and self.res is not None:
            bc = self.preprocess(bc)

        labels_bc = self._predictor.process(bc, path)


class Text2Box(Segmentation):
    input_prompts: Optional[List[str]] = Field(None)
    box_threshold: float = Field(0.2)
    text_threshold: float = Field(0.2)
    max_size_threshold: float = Field(0.25)
    _all_prompts: Sequence[np.array]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.postprocessors = [SeparateInstances(min_marker=self.min_marker)]
        self.adapter.preprocess_fn = None
        self.adapter.postprocess_fn = None
        self._all_prompts = [np.array([input_prompt], dtype=np.object_) for input_prompt in self.input_prompts]

    def processing_fn(self, sample, block):
        if not isinstance(sample, np.ndarray):
            sample = sample.numpy()

        if len(self._all_prompts) > 1:
            raise NotImplementedError(f'Only 1 prompt is supported! Got len of prompts = {len(self._all_prompts)}')
        else:
            text_prompt = self._all_prompts[0]

        input_data = (sample,
                      text_prompt,
                      np.array([self.box_threshold], dtype=np.float32),
                      np.array([self.text_threshold], dtype=np.float32),
                      np.array([self.max_size_threshold], dtype=np.float32))

        out = self.adapter(input_data)

        # if there are boxes
        if out.sum() > 0:
            global_boxes = out + np.array([block['x'], block['y'], block['x'], block['y']])
            global_boxes = global_boxes.astype(np.int64)

            np.save(f"{self.path}/boxes_{block['x']}_{block['y']}.npy", global_boxes)

        out_raster = np.full(shape=(len(self.output_labels), block['height'], block['width']),
                             fill_value=self.nodata, dtype=self._predictor.dst_dtype)

        # if there are boxes
        if out.sum() > 0:
            for box_ in out:
                x1, y1, x2, y2 = box_.astype(np.int64)
                out_raster[:, y1:y2, x1:x2] = 1

        return out_raster

    def __call__(self, path):
        self.path = path
        bc = io.read_bc(path, self.input_rasters)

        if self.crs is not None and self.res is not None:
            bc = self.preprocess(bc)

        labels_bc = self._predictor.process(bc, path)
