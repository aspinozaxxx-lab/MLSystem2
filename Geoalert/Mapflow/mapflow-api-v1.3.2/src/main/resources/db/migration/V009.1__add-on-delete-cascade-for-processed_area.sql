ALTER TABLE processed_area DROP CONSTRAINT processed_area_fk_app_user;
ALTER TABLE processed_area ADD CONSTRAINT processed_area_fk_app_user FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE;
