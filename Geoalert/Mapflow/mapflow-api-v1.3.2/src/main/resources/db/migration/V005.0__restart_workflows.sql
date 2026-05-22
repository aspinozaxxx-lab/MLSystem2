ALTER TABLE workflow ADD COLUMN start_requested BOOLEAN;
UPDATE workflow SET start_requested = FALSE;
ALTER TABLE workflow ALTER COLUMN start_requested SET NOT NULL;