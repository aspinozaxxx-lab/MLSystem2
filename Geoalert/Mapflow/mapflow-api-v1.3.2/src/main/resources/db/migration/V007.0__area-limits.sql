CREATE TABLE processed_area(
  processing_id UUID PRIMARY KEY NOT NULL,
  user_id UUID NOT NULL CONSTRAINT processed_area_fk_app_user REFERENCES app_user,
  area BIGINT NOT NULL,
  created TIMESTAMP,
  updated TIMESTAMP
);
CREATE INDEX processed_area_user_id_index ON processed_area (user_id);

ALTER TABLE app_user ADD COLUMN area_limit BIGINT;
UPDATE app_user SET area_limit = 0;
ALTER TABLE app_user ALTER COLUMN area_limit SET NOT NULL;

ALTER TABLE app_user ADD COLUMN updated TIMESTAMP;
UPDATE app_user SET updated = created;
ALTER TABLE app_user ALTER COLUMN updated SET NOT NULL;