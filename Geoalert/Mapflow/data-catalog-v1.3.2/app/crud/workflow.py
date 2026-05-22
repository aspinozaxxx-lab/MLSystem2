from typing import List, Dict
from loguru import logger
from sqlalchemy.dialects.postgresql import UUID

from functional.database import SessionLocal
from model import Workflow, WorkflowStatus, WorkflowDef


# Workflow definitions
def get_workflow_def() -> WorkflowDef:
    """
    Returns first (which is the only) record in workflow_def table
    :return: WorkflowDef
    """
    with SessionLocal.begin() as session:
        return session.query(WorkflowDef).first()


def update_workflow_def(id: int, yaml: str) -> None:
    with SessionLocal.begin() as session:
        workflow_def = session.query(WorkflowDef).first()
        if workflow_def is None:
            workflow_def = WorkflowDef(id=id, yaml=yaml)
            session.add(workflow_def)
        else:
            workflow_def.id = id
            workflow_def.yaml = yaml
        session.commit()


# Workflows
def get_workflows_by_id(uids: List[UUID]) -> Workflow:
    with SessionLocal.begin() as session:
        workflows = session.query(Workflow).filter(Workflow.image_id.in_(uids)).all()
    return workflows


def get_workflows_by_status(statuses: List[str]) -> List[Workflow]:
    with SessionLocal.begin() as session:
        workflow = session.query(Workflow).filter(Workflow.status.in_(statuses)).all()
    return workflow


def add_workflow_with_image_id(image_id: UUID, status: str):
    # TODO: change status to WorkflowStatus after merging
    with SessionLocal.begin() as session:
        new_workflow = Workflow(image_id=image_id,
                                status=status, we_id=None)
        session.add(new_workflow)
        session.commit()


def update_statuses(workflows_to_update: Dict[int, WorkflowStatus]):
    with SessionLocal.begin() as session:
        workflows = session.query(Workflow).filter(Workflow.we_id.in_(workflows_to_update.keys())).all()
        for workflow in workflows:
            workflow.status = workflows_to_update[workflow.we_id]
        session.commit()


def update_workflow_by_image_id(image_id, workflow_id, status):
    with SessionLocal.begin() as session:
        workflow = session.query(Workflow).filter(Workflow.image_id == image_id).first()
        workflow.we_id = workflow_id
        workflow.status = status
        session.commit()


def delete_workflow_record_by_image_id(image_id):
    with SessionLocal.begin() as session:
        session.query(Workflow).filter(Workflow.image_id == image_id).delete()
        session.commit()
