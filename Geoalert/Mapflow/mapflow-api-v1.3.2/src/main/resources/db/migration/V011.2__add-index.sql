ALTER TABLE workflow ADD COLUMN IF NOT EXISTS "create_date" TIMESTAMP;
CREATE INDEX workflow_create_date_index ON workflow (create_date);
CREATE INDEX aoi_completion_date_index ON aoi (completion_date);
