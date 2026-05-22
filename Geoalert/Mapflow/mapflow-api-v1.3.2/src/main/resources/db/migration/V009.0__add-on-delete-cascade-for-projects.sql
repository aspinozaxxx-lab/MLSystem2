ALTER TABLE project DROP CONSTRAINT project_fk_app_user;
ALTER TABLE project ADD CONSTRAINT project_fk_app_user FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE;
