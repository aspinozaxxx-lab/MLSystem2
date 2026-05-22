insert into
  user_projects (user_id, project_id, role)
select
  DISTINCT ON (u.user_id, up.project_id) u.user_id as user_id,
  up.project_id as project_id,
  'maintainer' as role
from
  user_projects up
  inner join team_member tm on tm.user_id = up.user_id
  inner join (
    SELECT
      up.user_id as user_id,
      t.id as team_id
    FROM
      team_member tm
      JOIN user_projects up ON tm.user_id = up.user_id
      JOIN team t ON tm.team_id = t.id
    WHERE
      t.archived = FALSE
      AND tm.role = 'OWNER'
      AND tm.team_id = (
        SELECT
          team_id
        FROM
          team_member
        WHERE
          user_id = up.user_id
        LIMIT
          1
      )
  ) u on u.team_id = tm.team_id ON CONFLICT DO NOTHING;