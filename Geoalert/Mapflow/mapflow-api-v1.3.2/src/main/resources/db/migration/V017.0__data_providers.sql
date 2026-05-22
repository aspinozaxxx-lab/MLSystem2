CREATE TABLE data_provider(
                               id UUID PRIMARY KEY NOT NULL,
                               name VARCHAR(256) NOT NULL,
                               display_name VARCHAR(256) NOT NULL,
                               url_template VARCHAR(2048) DEFAULT NULL,
                               price_per_mp NUMERIC DEFAULT 0,
                               credentials_username VARCHAR(1024) DEFAULT NULL,
                               credentials_password VARCHAR(1024) DEFAULT NULL,
                               credentials_token VARCHAR(1024) DEFAULT NULL,
                               is_default BOOLEAN DEFAULT false,
                               created TIMESTAMP,
                               updated TIMESTAMP,
                               archived BOOLEAN DEFAULT false
);

CREATE TABLE app_user_data_provider(
    user_id UUID NOT NULL CONSTRAINT app_user_data_providers_fk_user REFERENCES app_user ON DELETE CASCADE,
    data_provider_id UUID NOT NULL CONSTRAINT app_user_data_providers_fk_data_provider REFERENCES data_provider ON DELETE CASCADE,
    PRIMARY KEY(user_id, data_provider_id)
);
