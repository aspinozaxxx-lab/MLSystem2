ALTER TABLE task_status ADD COLUMN IF NOT EXISTS results hstore;

ALTER TABLE task ADD COLUMN IF NOT EXISTS aoi_id bigint;

ALTER TABLE task ADD CONSTRAINT task_fk_area_of_interest FOREIGN KEY (aoi_id) REFERENCES area_of_interest;
