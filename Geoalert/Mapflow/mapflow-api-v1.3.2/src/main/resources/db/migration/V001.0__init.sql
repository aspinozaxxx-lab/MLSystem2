CREATE TABLE app_user(
  id UUID PRIMARY KEY NOT NULL,
  login VARCHAR(64) NOT NULL,
  password VARCHAR(1024) NOT NULL,
  email VARCHAR(64) NOT NULL,
  role INT NOT NULL,
  created TIMESTAMP
);
CREATE UNIQUE INDEX app_user_login_index ON app_user (login);
INSERT INTO app_user (id, login, password, role, email, created) VALUES ('61cd6899-19e8-44a0-97db-b86f1a9b7af4', 'admin@admin.com', '5000:212a1530b2759046c383bc8d834409e8:9ab1fffad73e85b13f0c25335a3437651936d75875845cc50d96fd58fd19a97de9cf5335615ee7b2d379155bdc4ae0c2edce3e828d98966b1724ea6a34215b60', 0, 'admin@admin.com', '2019-12-16 19:10:29.492358');

CREATE TABLE project(
  id UUID PRIMARY KEY NOT NULL,
  name VARCHAR(128),
  description VARCHAR(1024),
  user_id UUID NOT NULL CONSTRAINT project_fk_app_user REFERENCES app_user,
  created TIMESTAMP,
  updated TIMESTAMP
);
CREATE INDEX project_user_id_index ON project (user_id);

CREATE TABLE workflow_def(
  id UUID PRIMARY KEY NOT NULL,
  project_id UUID NOT NULL CONSTRAINT workflow_def_fk_project REFERENCES project ON DELETE CASCADE,
  name VARCHAR(128),
  description VARCHAR(1024),
  we_id BIGINT,
  we_name VARCHAR(128),
  yml BYTEA,
  created TIMESTAMP,
  updated TIMESTAMP
);
CREATE INDEX workflow_def_project_id_index ON workflow_def (project_id);
CREATE UNIQUE INDEX workflow_def_we_name_uindex ON workflow_def (we_name);

CREATE TABLE vector_layer(
  id UUID PRIMARY KEY NOT NULL,
  external_id VARCHAR(64),
  name VARCHAR(128)
);

CREATE TABLE raster_layer(
  id UUID PRIMARY KEY NOT NULL,
  uri VARCHAR(256)
);

CREATE TABLE processing(
  id UUID PRIMARY KEY NOT NULL,
  project_id UUID NOT NULL CONSTRAINT processing_fk_project REFERENCES project ON DELETE CASCADE,
  vector_layer_id UUID NOT NULL CONSTRAINT processing_fk_vector_layer REFERENCES vector_layer,
  raster_layer_id UUID NOT NULL CONSTRAINT processing_fk_raster_layer REFERENCES raster_layer,
  workflow_def_id UUID CONSTRAINT processing_workflow_def_fk_workflow_def REFERENCES workflow_def ON DELETE CASCADE,
  name VARCHAR(128),
  description VARCHAR(1024),
  created TIMESTAMP,
  updated TIMESTAMP
);
CREATE INDEX processing_project_id_index ON processing (project_id);
CREATE INDEX processing_vector_layer_id_index ON processing (vector_layer_id);
CREATE INDEX processing_raster_layer_id_index ON processing (raster_layer_id);
CREATE INDEX processing_workflow_def_id_index ON processing (workflow_def_id);

CREATE TABLE aoi(
  id UUID PRIMARY KEY NOT NULL,
  processing_id UUID NOT NULL CONSTRAINT aoi_fk_processing REFERENCES processing ON DELETE CASCADE,
  geometry GEOMETRY NOT NULL,
  area BIGINT NOT NULL,
  status INT NOT NULL,
  percent_completed INT NOT NULL
);
CREATE INDEX aoi_processing_id_index ON aoi (processing_id);
CREATE INDEX aoi_geometry_index ON aoi USING gist (geometry);

CREATE TABLE workflow(
  id UUID PRIMARY KEY NOT NULL,
  aoi_id UUID NOT NULL CONSTRAINT workflow_fk_aoi REFERENCES aoi ON DELETE CASCADE,
  workflow_def_id UUID CONSTRAINT workflow_fk_workflow_def REFERENCES workflow_def,
  external_id VARCHAR(32),
  geometry GEOMETRY NOT NULL,
  area BIGINT NOT NULL,
  status INT NOT NULL
);
CREATE INDEX workflow_aoi_id_index ON workflow (aoi_id);
CREATE INDEX workflow_status_index ON workflow (status);
CREATE INDEX workflow_geometry_index ON workflow USING gist (geometry);