DO
$$
    BEGIN
        IF NOT EXISTS (SELECT * FROM pg_type typ INNER JOIN pg_namespace nsp ON nsp.oid = typ.typnamespace
            WHERE nsp.nspname = current_schema() AND typ.typname = 'member_role')
            THEN
            CREATE TYPE MEMBER_ROLE AS ENUM ('owner', 'readonly', 'maintainer', 'contributor');
        END IF;
    END;
$$
LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS user_projects (
    user_id UUID NOT NULL CONSTRAINT fk_team_member REFERENCES app_user (id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL CONSTRAINT fk_team_project REFERENCES project (id) ON UPDATE CASCADE ON DELETE CASCADE,
    role MEMBER_ROLE NOT NULL,
    UNIQUE (user_id, project_id)
);

INSERT INTO user_projects (user_id, project_id, role) SELECT user_id, id, 'owner' FROM project;