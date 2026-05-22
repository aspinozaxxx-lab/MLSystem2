import random
import numpy as np
from loguru import logger
import asyncio
from aiohttp import ClientError, ClientResponseError
from typing import Mapping, Optional

from copy import copy
from PIL import Image
from io import BytesIO
from .agents import USER_AGENTS
from .utils import to_quad_key
from .errors import (WrongNdim,
                     WrongChannelsNum,
                     WrongTileSize,
                     WrongSourceType,
                     TileNotLoaded,
                     TileNotReadable)


def make_rgb(image):
    """
    Make RGB from RGBA or panchrom image.
    Args:
        image: input image of shape  (h,w), (h,w,1), (h,w,3) or (h,w,4) All other dimensions apart will throw an error
    Returns:
        transformed to 3-channels image. 4th channel is deleted, 1-channel is copied to all others
    """
    if image.ndim != 3:
        raise WrongNdim(real_ndim=image.shape)
    # 1-channel image
    if image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    elif image.shape[2] == 4:
        # transform image alpha channel to nodata mask (zeros), clipping real values to 1:255
        image = np.where(image[:, :, -1:] != 0, image[:, :, :3].clip(1, 255), np.zeros_like(image[:, :, :3]))
    # Other
    elif image.shape[2] != 3:
        raise WrongChannelsNum(expected_nchannels=[1, 3, 4], real_nchannels=image.shape[2])
    # In case of (h,w,3) image is returned as is
    return image


def extract_image_from_response(content,
                                tile_size,
                                nchannels=None):
    """Extract image from response, validate shape and handle errors

    Args:
        response: aiohttp response.read() result
        tile_size (int): size of loaded image (need in case of error)
        ignore_errors (bool): if `True` and error is raised return "black" image with shape (*tile_size, 3)

    Returns:
        image (np.array): loaded tile image

    Raises:
         ValueError: if shape of loaded image is not suit (*tile_size, 3)

    """
    image = Image.open(BytesIO(content))
    if image.mode not in ['RBG', 'RGBA']:
        # Convert image from paletted and grayscale
        # todo: refactor to remove make_rgb if it can be replaced by this convert
        image = image.convert('RGBA')
    image = np.asarray(image)
    # 1-channel images are read as 2-dim array, expand it for consistency
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    if tile_size != image.shape[0] or tile_size != image.shape[1]:
        raise WrongTileSize(expected_size=(tile_size, tile_size), real_size=image.shape[:2])

    if nchannels is None:
        image = make_rgb(image)
    elif image.shape[2] != nchannels:
        raise WrongChannelsNum(expected_nchannels=3, real_nchannels=image.shape[2])

    return image


class Loader:
    agents = USER_AGENTS

    default_headers = {
        'user_agent': 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) '
                      'AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1',
    }

    def __init__(
            self,
            url: str,
            session,
            source_type: str = 'xyz',
            header: Optional[Mapping] = None,
            credentials=None,
            retry_attempts=5,
            retry_delay=1,
            response_timeout=10,
            rotate_agents=False,
            tile_size=256,
            ignore_errors=False,
            proxy=None,
            connection_limit: int = 100
    ):

        if not source_type in {'xyz', 'tms', 'quadkey'}:
            raise WrongSourceType(source_type)

        self.url = url
        self.session = session
        self.source_type = source_type
        self.headers = header
        self.credentials = credentials
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.response_timeout = response_timeout
        self.rotate_agents = rotate_agents
        self.tile_size = tile_size
        self.ignore_errors = ignore_errors
        self.proxy = proxy
        self.semaphore = asyncio.Semaphore(connection_limit)

    def nodata_tile(self, tile):
        return np.zeros((self.tile_size, self.tile_size, tile.nchannels or 3), dtype="uint8")

    def get_headers(self):
        headers = copy(Loader.default_headers)
        if self.headers:
            headers.update(**self.headers)

        if self.rotate_agents:
            headers['user_agent'] = random.choice(Loader.agents)

        return headers

    async def retrying_request(self, tile_url):
        for attempt in range(self.retry_attempts + 1):
            try:
                async with self.session.get(tile_url,
                                            data=None,
                                            headers=self.get_headers(),
                                            auth=self.credentials,
                                            timeout=self.response_timeout,
                                            proxy=self.proxy,
                                            raise_for_status=True) as response:
                    status = response.status
                    content = await response.read()
            except (ClientError, asyncio.TimeoutError) as e:
                # we retry on timeouts or client error
                if attempt == self.retry_attempts or \
                        (isinstance(e, ClientResponseError) and e.status in (401, 403)):
                    # we do not retry rejected authodization because it is useless
                    # and can trigger server security
                    # actually, there may be even less errors to retry
                    raise e
                logger.info(f"Tile at {tile_url} not loaded due to {str(e)}. " +
                            f"Sleeping for {self.retry_delay*(2**attempt)} sec. Left {self.retry_attempts - attempt - 1} attempts")
                logger.opt(exception=True).trace("Error details: ")
                await asyncio.sleep(self.retry_delay*(2**attempt))
            else:
                logger.trace(f" Tile at {tile_url} loaded")
                return status, content

    async def load_tile(self, tile):
        z = tile.z
        x = tile.x
        y = tile.y

        if self.source_type == 'tms':
            y = 2 ** z - y - 1

        if self.source_type == 'quadkey':
            tile_url = self.url.format(q=to_quad_key(x, y, z))
        else:
            tile_url = self.url.format(z=z, x=x, y=y)

        try:
            async with self.semaphore:
                # do not allow more than allowed simultaneous requests
                status, content = await self.retrying_request(tile_url)
        except Exception as e:
            if self.ignore_errors:
                logger.info(f"Error {e} ignored while loading tile from basemap")
                logger.opt(exception=True).debug("Ignored error trace:")
                return self.nodata_tile(tile)
            else:
                # Maybe an ugly solution, but here we again need to form different user errors
                # depending on the exception classes. Known exceptions like timeoutError and clientResponseError
                # get their own messages, and all the others are returned as a generic tileNotLoaded
                try:
                    raise e
                except asyncio.TimeoutError as e:
                    raise TileNotLoaded(tile_location=tile_url,
                                        proxy=self.proxy,
                                        exception_message='Timeout on request to tile',
                                        http_status=None)
                except ClientResponseError as e:
                    logger.debug(f"ClientResponseError: request info = {e.request_info},"
                                 f"headers = {e.headers}")
                    if e.status == 401:
                        raise TileNotLoaded(tile_location=tile_url,
                                            proxy=self.proxy,
                                            exception_message='Server requires valid credentials!',
                                            http_status=e.status)
                    elif e.status == 403:
                        raise TileNotLoaded(tile_location=tile_url,
                                            proxy=self.proxy,
                                            exception_message='Tile download is forbidden!',
                                            http_status=e.status)
                    else:
                        raise TileNotLoaded(tile_location=tile_url,
                                            proxy=self.proxy,
                                            exception_message='Server returned error response',
                                            http_status=e.status)
                except Exception as e:
                    logger.exception('Unexpected error in tile loading')
                    raise TileNotLoaded(tile_location=tile_url,
                                        proxy=self.proxy,
                                        exception_message=str(e),
                                        http_status=None)
        try:
            image = extract_image_from_response(content,
                                                tile_size=self.tile_size,
                                                nchannels=tile.nchannels)
        except Exception as e:
            if self.ignore_errors or (204 == status):
                # status codes 204 - no content will be treated as intended absence of the data and not alert an error
                logger.info("Error ignored while reading tile from response")
                logger.opt(exception=True).debug("Exception info:")
                return self.nodata_tile(tile)
            else:
                logger.info("Error while reading tile from response")
                logger.opt(exception=True).debug("Exception info:")
                raise TileNotReadable(tile_location=tile_url,
                                      proxy=self.proxy,
                                      exception_message=str(e))

        return image
