import requests
import time
import yaml
from loguru import logger
from uuid import uuid4
from config import Config
from crud.workflow import get_workflow_def, update_workflow_def


PATH_TO_WD_YML = Config.PATH_TO_WD_YML
WORKFLOW_LOGIN = Config.WORKFLOW_ENGINE_LOGIN
WORKFLOW_PASSWORD = 
WORKFLOW_DEF_POST_URL = Config.WORKFLOW_ENGINE_URL + '/api/v0/definitions'


def workflow_def_check() -> None:
    """
    Checks if /wd.yml is different from the one which is in workflow_def table.
    If different post new wd.yml file to https://workflow-{app_env}.mapflow.ai/api/v0/definitions and update ID in db.
    curl example:
        curl -X POST -F definition=@path_to_wd_yml -u login:password https://workflow-staging.mapflow.ai/api/v0/definitions .

    path_to_wd_yml, login and password, app_env are taken from env variables.
    :return: None
    """
    while True:
        try:
            wd_record = get_workflow_def()
            break
        except Exception as e:
            logger.warning(f"DB is not responding. Waiting for {Config.SLEEP} before reconnecting")
            time.sleep(Config.SLEEP)
    if wd_record is None:
        wd_from_db = None
    else:
        wd_from_db = get_workflow_def().yaml
    with open(PATH_TO_WD_YML) as file:
        local_wd_yaml = file.read()
    if wd_from_db == local_wd_yaml:
        # We already have the same WD in our DB and it is already posted to WE
        return

    wd_for_sending = PATH_TO_WD_YML + '_mod'
    change_name_of_wd(PATH_TO_WD_YML, wd_for_sending)
    while True:
        try:
            files = {'definition': open(wd_for_sending, 'rb')}
            response = requests.post(WORKFLOW_DEF_POST_URL,
                                     auth=(WORKFLOW_LOGIN, WORKFLOW_PASSWORD),
                                     files=files)
            response.raise_for_status()
            wd_id = response.json()["id"]
            break
        except Exception as e:
            logger.warning(f"Workflow server {WORKFLOW_DEF_POST_URL} is unavailable. Error {e}"
                             f"Waiting for {Config.SLEEP} before reconnecting")
            time.sleep(Config.SLEEP)
    update_workflow_def(id=wd_id,
                        yaml=local_wd_yaml)


def change_name_of_wd(file, out_file):
    """
    Replace WD name with random UUID to avoid versioning conflicts
    TODO: handle proper version history according to WE?
    """
    data = yaml.safe_load(open(file).read())
    data['name'] = str(uuid4())
    open(out_file, 'w').write(yaml.safe_dump(data))
