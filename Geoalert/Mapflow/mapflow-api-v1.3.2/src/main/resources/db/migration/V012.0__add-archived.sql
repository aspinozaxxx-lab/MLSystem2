ALTER TABLE project ADD COLUMN IF NOT EXISTS "archived" boolean default false;
ALTER TABLE processing ADD COLUMN IF NOT EXISTS "archived" boolean default false;