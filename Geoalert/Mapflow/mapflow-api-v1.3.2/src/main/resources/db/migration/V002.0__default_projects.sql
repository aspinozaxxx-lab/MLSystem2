ALTER TABLE project ADD COLUMN is_default BOOLEAN;
UPDATE project SET is_default = false;
ALTER TABLE project ALTER COLUMN is_default SET NOT NULL;

CREATE EXTENSION IF NOT EXISTS hstore;

ALTER TABLE processing ADD COLUMN params HSTORE;
UPDATE processing SET params = '';
ALTER TABLE processing ALTER COLUMN params SET NOT NULL;