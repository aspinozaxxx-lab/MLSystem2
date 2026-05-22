DELETE from user_projects where role = 'maintainer';

INSERT INTO user_projects (user_id, project_id, role)
SELECT app_user.id, project.id, 'maintainer'
FROM project, app_user 
WHERE app_user."role" = 0 on conflict do nothing;
