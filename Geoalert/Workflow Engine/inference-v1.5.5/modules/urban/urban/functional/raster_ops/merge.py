import rasterio
from tqdm import tqdm
from .split import generate_windows



def merge(src_fps,
          dst_fp,
          src_channels,
          window_size=10000):

    with rasterio.open(src_fps[0]) as src:
        kwargs = src.meta.copy()
    kwargs.update({'count': 3})

    with rasterio.open(dst_fp, 'w', **kwargs) as dst:
        for i, (src_fp, src_ch) in tqdm(enumerate(zip(src_fps, src_channels))):
            with rasterio.open(src_fp) as src:
                if not window_size:
                    window_size = max(src.height, src.width)
                for window in generate_windows(dataset_height=src.height, dataset_width=src.width,
                                               window_height=window_size, window_width=window_size):

                    data = src.read(1, window=window)

                    dst.write(data, i+1, window=window)