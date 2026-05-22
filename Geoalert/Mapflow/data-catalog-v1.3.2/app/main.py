import sys
import uvicorn

from threading import Thread
from loguru import logger
from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from routers import mosaic, data, healthcheck
from service import workflow_def_check, start_polling_we
from schemas.http_credentials import HTTPCredentialsCustom
from dependencies.auth_dependencies import StandardHTTPSecurity

app = FastAPI(docs_url=Config.ROUTE_PREFIX + "/docs",
              openapi_url=Config.ROUTE_PREFIX + "/openapi.json",
              redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.openapi_version = "3.0.0"

security = StandardHTTPSecurity()

# mosaic management
app.include_router(mosaic.router)

# data management
app.include_router(data.router)

# healthcheck endpoints
app.include_router(healthcheck.router)


@app.get("/")
async def root(request: Request, response: Response, credentials: HTTPCredentialsCustom = Depends(security)):
    return {"user status": "authenticated"}


@app.on_event("startup")
def startup_event():

    logger.remove()
    logger.add(sink=sys.stdout, level=Config.LOG_LEVEL)
    config = Config()
    logger.info(f"Starting with config: {repr(config)}")
    # check if wd.yaml provided with app is different from which is stored in workflow_def table
    workflow_def_check()
    # start polling WE on ongoing processings
    p = Thread(target=start_polling_we)
    p.start()


if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
