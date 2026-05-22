import asyncio
import shutil
from pathlib import Path

import aiohttp
from loguru import logger


async def load_image(url, headers, timeout):
    async with aiohttp.ClientSession() as session:
        async with session.get(url,
                               data=None,
                               headers=headers,
                               timeout=timeout,
                               raise_for_status=True) as response:
            content = await response.read()
    return content

def download(url, server, output_fp, header, timeout=300, **kwargs):
    """
    Load tiles by link.
    Args:
        url: image_id in system
        server: URL with pattern "image_id" to be filled with "url"
        output_fp: where to save file
        header: how to sign the request
        timeout: how long do we wait for the server response
    """
    data = asyncio.run(
        load_image(url=server.format(image_id=url), headers=header, timeout=timeout)
    )
    output_dir = Path(output_fp).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_fp, 'wb') as dst:
        dst.write(data)
    logger.info(f"Loaded image {url} from {server}")
