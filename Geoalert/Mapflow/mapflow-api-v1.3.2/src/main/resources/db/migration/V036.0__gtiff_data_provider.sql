INSERT INTO data_provider (id, name, display_name, url_template, price_per_mp, credentials_username,
                           credentials_password, credentials_token, is_default, created, updated, archived, preview_url)
VALUES ('c433e83a-8da4-4a44-8257-291dbd0c2da3',
        'GTIFF',
        'Загрузить GeoTIFF',
        '',
        0,
        '',
        '',
        '',
        true,
        NOW(),
        NULL,
        false,
        '')
ON CONFLICT (id) DO NOTHING;