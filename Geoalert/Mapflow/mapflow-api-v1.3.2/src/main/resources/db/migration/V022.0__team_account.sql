CREATE TABLE team
(
    id       UUID PRIMARY KEY NOT NULL,
    name     VARCHAR(256)     NOT NULL,
    created  TIMESTAMP,
    updated  TIMESTAMP,
    archived BOOLEAN DEFAULT false
);

CREATE UNIQUE INDEX team_name_index ON team (name);

CREATE TABLE team_member
(
    team_id          UUID NOT NULL CONSTRAINT team_member_fk_team REFERENCES team ON DELETE CASCADE,
    user_id          UUID NOT NULL CONSTRAINT team_member_fk_user REFERENCES app_user ON DELETE CASCADE,
    role             VARCHAR(64) DEFAULT 'MEMBER',
    active_until     TIMESTAMP DEFAULT NULL,
    PRIMARY KEY (user_id, team_id)
);

