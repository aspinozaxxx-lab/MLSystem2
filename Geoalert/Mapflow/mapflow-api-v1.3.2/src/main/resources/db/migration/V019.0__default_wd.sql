CREATE TABLE project_workflow_def(
    project_id UUID NOT NULL CONSTRAINT project_workflow_def_fk_project REFERENCES project ON DELETE CASCADE,
    workflow_def_id UUID NOT NULL CONSTRAINT project_workflow_def_fk_workflow_def REFERENCES workflow_def ON DELETE CASCADE,
    PRIMARY KEY(project_id, workflow_def_id)
                                 );

CREATE TABLE user_workflow_def(
    user_id UUID NOT NULL CONSTRAINT user_workflow_def_fk_project REFERENCES app_user ON DELETE CASCADE,
    workflow_def_id UUID NOT NULL CONSTRAINT user_workflow_def_fk_workflow_def REFERENCES workflow_def ON DELETE CASCADE,
PRIMARY KEY(user_id, workflow_def_id)
                                 );

INSERT INTO project_workflow_def (SELECT project_id, id FROM workflow_def);
INSERT INTO user_workflow_def (SELECT p.user_id, wd.id FROM workflow_def wd JOIN project p ON (p.id = wd.project_id));

ALTER TABLE workflow_def DROP COLUMN "project_id";
ALTER TABLE workflow_def ADD COLUMN IF NOT EXISTS "is_default" boolean NOT NULL default false;
