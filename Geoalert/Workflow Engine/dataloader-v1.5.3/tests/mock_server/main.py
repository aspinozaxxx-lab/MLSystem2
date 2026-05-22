import os
import uvicorn
import numpy as np
from skimage.io import imsave, imread
from asyncio import sleep
from loguru import logger
from fastapi import FastAPI, Request, Response, APIRouter, status

from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI

app = FastAPI()

LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)ms [%(name)s] %(levelprefix)s %(message)s"

test_tile = './tile.png'
FILETYPES = {'png': [1, 3, 4], 'jpg': [1, 3]}
DTYPES = ['uint8']


def get_filename(filetype, dtype, count, folder='./'):
    if dtype not in DTYPES:
        raise ValueError('Unsupported dtype')
    if filetype not in FILETYPES:
        raise ValueError('Unsupported file type')
    count = int(count)
    if count > 5 or count < 1:
        raise ValueError('Unsupported channels num')
    return os.path.join(folder, f'{dtype}_{count}.{filetype}')

# todo: add other tile size
# todo: add different values to rasters to make sure that the data is merged in the right order


def generate_img(filetype, dtype, count, folder='.'):
    name = get_filename(filetype, dtype, count, folder)
    data = np.ones(shape=(256, 256, count), dtype=dtype)
    for i in range(count):
        # Make value equal to channel num
        data[:, :, i] = data[:, :, i]*((i+1)*10)
    maxval = 255 if dtype == 'uint8' else 65535

    # alpha channel - full opacity
    if count == 2 or count == 4:
        data[:, :, -1] = maxval

    imsave(name, data.astype(dtype), check_contrast=False)
    new = imread(name)
    logger.info(f"After reading {filetype}: MIN {new.min()}, MAX {new.max()}")


@app.on_event("startup")
def generate_imgs():
    """
    We generate all the imgs we want to return later
    """
    for dtype in DTYPES:
        for filetype, counts in FILETYPES.items():
            for count in counts:
                generate_img(filetype, dtype, count)


@app.on_event("shutdown")
def remove_imgs():
    for dtype in DTYPES:
        for filetype, counts in FILETYPES.items():
            for count in counts:
                try:
                    os.remove(get_filename(filetype, dtype, count))
                except:
                    pass


@app.get("/tile/{sleeptime}/{z}/{x}/{y}")
async def return_tile_sleep(request: Request, response: Response, sleeptime, z, x, y):
    """
    Sleep for z seconds then reply depending on X
    """
    try:
        x = int(x)
        sleeptime = float(sleeptime)
    except:
        return Response(status_code=500)
    await sleep(sleeptime)
    if x == 401:
        return Response(status_code=401, content="")
    elif x == 403:
        return Response(status_code=403, content="")
    elif x == 404:
        return Response(status_code=404, content="")
    elif x == 204:
        return Response(status_code=204, content="")
    # else
    img = open(get_filename('png', 'uint8', 3), 'rb').read()
    return Response(status_code=200, content=img)


@app.get("/tiles/{filetype}/{dtype}/{count}/{z}/{x}/{y}")
async def return_tile(request: Request,
                      response: Response,
                      filetype,
                      dtype,
                      count,
                      z, x, y):
    """
    Sleep for z seconds then reply depending on X
    """
    try:
        count = int(count)
        filename = get_filename(filetype, dtype, count)
    except Exception as e:
        logger.warning(str(e))
        return Response(status_code=500)
    img = open(filename, 'rb').read()
    return Response(status_code=200, content=img)


@app.get("/")
async def root(request: Request, response: Response):
    return {"user status": "authenticated"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
