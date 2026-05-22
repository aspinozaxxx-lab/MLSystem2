import json

import uvicorn
import base64
from loguru import logger
from fastapi import FastAPI, Request, Response, status
from sample_credentials import sample_base

app = FastAPI(dependencies=[])

workflow_id = 1000


@app.get("/")
async def root(request: Request):
    return {"message": "Mock server is up and working. Send authorization requests to /auth/"}


@app.get('/api/v0/test/')
async def test(request: Request):
    return "OK"


@app.get("/auth/", status_code=200)
async def authorize_user(request: Request, response: Response):
    if not request.headers.get("authorization"):
        response.status_code = status.HTTP_403_FORBIDDEN
        return {"code": "ACCESS_DENIED", "message": "Access denied."}
    # get raw credentials
    credentials = request.headers["authorization"].split()[1]

    # decode
    decoded = base64.b64decode(credentials).decode("ascii")

    # check if credentials are valid or not
    username = decoded.split(sep=":")[0]
    password = ]

    if username not in sample_base:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"code": "AUTHENTICATION_ERROR", "message": "Wrong login/password."}

    if username in sample_base and not password = sample_base[username].get("password"):
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"code": "AUTHENTICATION_ERROR", "message": "Wrong login/password."}

    if username in sample_base and password = sample_base[username].get("password"):
        logger.info(f'Sending result: {username},'
                    f' {sample_base[username].get("is_admin")},'
                    f' {sample_base[username].get("memoryLimit")}')
        return {
            "email": username,
            "isAdmin": sample_base[username].get("is_admin"),
            "memoryLimit": sample_base[username].get("memoryLimit"),
        }

    return {"message": "Something went wrong. Please contact administrator."}
    # decoded = base64.b64decode(credentials).decode("ascii")


@app.post("/api/v0/definitions", status_code=200)
async def post_wd(request: Request, response: Response):
    return {"id": 100,
             "name": "MyAwesomeDefinitionName",
             "latestVersion":
                 {"id": 101,
                  "workflowDefinitionId": 4416262,
                  "version": 0
                  },
             "versions": [
                 {"id": 101,
                  "workflowDefinitionId": 100,
                  "version": 0}]
             }


def aoi_id(wf_id):
    return wf_id + 10


@app.post("/api/v0/workflows/page", status_code=200)
async def page_workflows(request: Request, response: Response):
    requested_workflows = await request.body()
    workflow_ids = json.loads(requested_workflows)['filter']['workflowIds']
    ''' {'filters': {'workflowIds':[]}}'''
    res = [
        {"id": id_,
         "workflowDefinitionId": 100,
         "stages": [],
         "areasOfInterest": [{"id": aoi_id(id_),
                              "geometry": {}}],
         "status": "OK",
         } for id_ in workflow_ids
    ]
    return res


@app.post("/api/v0/workflows", status_code=200)
async def add_workflow(request: Request, response: Response):
    """
    Answeres with a partial WE response, starting with workflow_id = 1100 and incrementing by 100 each time
    """
    wd = await request.body()
    wd_id = json.loads(wd)['workflowDefinitionId']
    ''' {'filters': {'workflowIds':[]}}'''
    global workflow_id
    workflow_id += 100
    res = {
        "id": workflow_id,
        "workflowDefinitionId": wd_id,
        "stages": [],
        "areasOfInterest": [{"id": aoi_id(workflow_id),
                             "geometry": {}}],
        "status": "IN_PROGRESS",
        }

    return res

if __name__ == "__main__":
    logger.info("Launching!")
    uvicorn.run(app, host="127.0.0.1", port=8050)
