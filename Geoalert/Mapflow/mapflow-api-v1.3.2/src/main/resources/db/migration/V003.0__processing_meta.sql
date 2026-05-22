ALTER TABLE processing ADD COLUMN meta HSTORE;
UPDATE processing SET meta = '';
ALTER TABLE processing ALTER COLUMN meta SET NOT NULL;