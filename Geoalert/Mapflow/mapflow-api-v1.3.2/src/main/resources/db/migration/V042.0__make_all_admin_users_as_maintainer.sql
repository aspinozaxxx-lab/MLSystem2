INSERT INTO user_projects (user_id, project_id, role)
SELECT app_user.id, project.id, 'maintainer'
FROM project, app_user 
WHERE app_user."role" = 1 on conflict do nothing;
