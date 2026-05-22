ALTER TABLE app_user ADD COLUMN aoi_area_limit BIGINT;
UPDATE app_user SET aoi_area_limit = 500000000;
ALTER TABLE app_user ALTER COLUMN aoi_area_limit SET NOT NULL;