INSERT INTO user_workflow_def (user_id, workflow_def_id)
SELECT p.user_id, pwd.workflow_def_id FROM project_workflow_def pwd
    JOIN project p ON p.id = pwd.project_id
    LEFT JOIN user_workflow_def uwd ON p.user_id = uwd.user_id
WHERE uwd.user_id IS NULL
GROUP BY p.user_id, pwd.workflow_def_id;