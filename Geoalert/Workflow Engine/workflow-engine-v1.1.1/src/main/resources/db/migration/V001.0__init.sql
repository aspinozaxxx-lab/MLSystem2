CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SEQUENCE hibernate_sequence
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE TABLE area_of_interest
(
    id          bigint PRIMARY KEY NOT NULL,
    geometry    geometry,
    workflow_id bigint

);

CREATE INDEX area_of_interest__workflow_id_idx ON area_of_interest USING btree (workflow_id);

CREATE TABLE artifact
(
    id                  bigint PRIMARY KEY NOT NULL,
    artifact_type       character varying(255),
    cache_key           character varying(32),
    uri                 character varying(1024),
    area_of_interest_id bigint,
    workflow_id         bigint


);
CREATE INDEX artifact__area_of_interest_id_idx ON artifact USING btree (area_of_interest_id);
CREATE INDEX artifact__workflow_id_idx ON artifact USING btree (workflow_id);

CREATE TABLE raster_layer
(
    id         bigint PRIMARY KEY NOT NULL,
    layer_type character varying(255),
    uri        character varying(255)
);

CREATE TABLE raster_source
(
    id                 bigint PRIMARY KEY NOT NULL,
    confirmed          boolean            NOT NULL,
    params             hstore,
    raster_source_type character varying(255),
    workflow_id        bigint
);
CREATE INDEX raster_source__workflow_id_idx ON raster_source USING btree (workflow_id);

CREATE TABLE stage
(
    id                  bigint PRIMARY KEY NOT NULL,
    stage_definition_id bigint,
    workflow_id         bigint
);
CREATE INDEX stage__workflow_id_idx ON stage USING btree (workflow_id);

CREATE TABLE stage_definition
(
    id                         bigint PRIMARY KEY NOT NULL,
    action                     character varying(255),
    description                character varying(255),
    name                       character varying(255),
    params                     hstore,
    retries                    integer,
    retry_interval             integer,
    workflow_definition_ver_id bigint
);

CREATE TABLE stage_definition_previous_stages
(
    stage_definition_id bigint NOT NULL,
    previous_stages_id  bigint NOT NULL
);
ALTER TABLE stage_definition_previous_stages ADD PRIMARY KEY (stage_definition_id, previous_stages_id);

CREATE TABLE stage_previous_stages
(
    stage_id           bigint NOT NULL,
    previous_stages_id bigint NOT NULL
        );
ALTER TABLE stage_previous_stages
    ADD PRIMARY KEY (stage_id, previous_stages_id);

CREATE TABLE stage_status
(
    id            bigint PRIMARY KEY NOT NULL,
    error_message character varying(1024),
    start_date    timestamp without time zone,
    status        integer,
    update_date   timestamp without time zone,
    stage_id      bigint
);
CREATE INDEX stage_status__stage_id_idx ON stage_status USING btree (stage_id);

CREATE TABLE stage_status_messages
(
    stage_status_id bigint NOT NULL,
    code            character varying(128),
    message         character varying(1024),
    parameters      hstore
);

CREATE TABLE task
(
    id       bigint PRIMARY KEY NOT NULL,
    request  text,
    stage_id bigint
);
CREATE INDEX task__stage_id_idx ON task USING btree (stage_id);

CREATE TABLE task_status
(
    id          bigint PRIMARY KEY NOT NULL,
    attempts    integer,
    code        integer,
    message     text,
    start_date  timestamp without time zone,
    status      integer,
    update_date timestamp without time zone,
    task_id     bigint
);
CREATE INDEX task_status__task_id_idx ON task_status USING btree (task_id);

CREATE TABLE task_status_messages
(
    task_status_id bigint NOT NULL,
    code           character varying(128),
    message        character varying(1024),
    parameters     hstore
);

CREATE TABLE vector_layer
(
    id       bigint PRIMARY KEY NOT NULL,
    layer_id uuid
);


CREATE TABLE workflow
(
    id                         bigint PRIMARY KEY NOT NULL,
    create_date                timestamp without time zone,
    meta                       hstore,
    params                     hstore,
    processing_id              character varying(64),
    system                     character varying(64),
    raster_layer_id            bigint,
    vector_layer_id            bigint,
    workflow_definition_ver_id bigint
);

CREATE TABLE workflow_definition
(
    id   bigint PRIMARY KEY NOT NULL,
    name character varying(255)
);


CREATE TABLE workflow_definition_ver
(
    id                     bigint PRIMARY KEY NOT NULL,
    version                integer,
    workflow_definition_id bigint
);

CREATE TABLE workflow_status
(
    id          bigint PRIMARY KEY NOT NULL,
    status      integer,
    update_date timestamp without time zone,
    workflow_id bigint
);
CREATE INDEX workflow_status__workflow_id_idx ON workflow_status USING btree (workflow_id);

ALTER TABLE area_of_interest ADD CONSTRAINT area_of_interest_fk_workflow FOREIGN KEY (workflow_id) REFERENCES workflow ON DELETE CASCADE;
ALTER TABLE artifact ADD CONSTRAINT artifact_fk_area_of_interest FOREIGN KEY (area_of_interest_id) REFERENCES area_of_interest ON DELETE CASCADE;
ALTER TABLE artifact ADD CONSTRAINT artifact_fk_workflow FOREIGN KEY (workflow_id) REFERENCES workflow ON DELETE CASCADE;
ALTER TABLE raster_source ADD CONSTRAINT raster_source_fk_workflow FOREIGN KEY (workflow_id) REFERENCES workflow ON DELETE CASCADE;
ALTER TABLE stage ADD CONSTRAINT stage_fk_stage_definition FOREIGN KEY (stage_definition_id) REFERENCES stage_definition ON DELETE CASCADE;
ALTER TABLE stage ADD CONSTRAINT stage_fk_workflow FOREIGN KEY (workflow_id) REFERENCES workflow ON DELETE CASCADE;
ALTER TABLE stage_definition ADD CONSTRAINT stage_definition_fk_workflow_definition_ver FOREIGN KEY (workflow_definition_ver_id) REFERENCES workflow_definition_ver ON DELETE CASCADE;
ALTER TABLE stage_definition_previous_stages ADD CONSTRAINT stage_definition_previous_stages_fk_stage_definition FOREIGN KEY (stage_definition_id) REFERENCES stage_definition ON DELETE CASCADE;
ALTER TABLE stage_definition_previous_stages ADD CONSTRAINT stage_definition_previous_stages_fk_previous_stage_definition FOREIGN KEY (previous_stages_id) REFERENCES stage_definition ON DELETE CASCADE;
ALTER TABLE stage_previous_stages ADD CONSTRAINT stage_previous_stages_fk_stage FOREIGN KEY (stage_id) REFERENCES stage ON DELETE CASCADE;
ALTER TABLE stage_previous_stages ADD CONSTRAINT stage_previous_stages_fk_previous_stage FOREIGN KEY (previous_stages_id) REFERENCES stage ON DELETE CASCADE;
ALTER TABLE stage_status ADD CONSTRAINT stage_status_fk_stage FOREIGN KEY (stage_id) REFERENCES stage ON DELETE CASCADE;
ALTER TABLE stage_status_messages ADD CONSTRAINT stage_status_messages_fk_stage_status FOREIGN KEY (stage_status_id) REFERENCES stage_status ON DELETE CASCADE;
ALTER TABLE task ADD CONSTRAINT task_fk_stage FOREIGN KEY (stage_id) REFERENCES stage ON DELETE CASCADE;
ALTER TABLE task_status ADD CONSTRAINT task_status_fk_task FOREIGN KEY (task_id) REFERENCES task ON DELETE CASCADE;
ALTER TABLE task_status_messages ADD CONSTRAINT task_status_messages_fk_task_status FOREIGN KEY (task_status_id) REFERENCES task_status ON DELETE CASCADE;
ALTER TABLE workflow ADD CONSTRAINT workflow_fk_raster_layer FOREIGN KEY (raster_layer_id) REFERENCES raster_layer ON DELETE CASCADE;
ALTER TABLE workflow ADD CONSTRAINT workflow_fk_vector_layer FOREIGN KEY (vector_layer_id) REFERENCES vector_layer ON DELETE CASCADE;
ALTER TABLE workflow ADD CONSTRAINT workflow_fk_workflow_definition_ver FOREIGN KEY (workflow_definition_ver_id) REFERENCES workflow_definition_ver ON DELETE CASCADE;
ALTER TABLE workflow_definition_ver ADD CONSTRAINT workflow_definition_ver_fk_workflow_definition FOREIGN KEY (workflow_definition_id) REFERENCES workflow_definition ON DELETE CASCADE;
ALTER TABLE workflow_status ADD CONSTRAINT workflow_status_fk_task FOREIGN KEY (workflow_id) REFERENCES workflow ON DELETE CASCADE;
