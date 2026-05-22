ALTER TABLE app_user DROP COLUMN is_premium;
ALTER TABLE app_user DROP COLUMN email;
ALTER TABLE app_user RENAME COLUMN login TO email;