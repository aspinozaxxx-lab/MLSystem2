-- sequence for feature indexing
CREATE SEQUENCE IF NOT EXISTS hibernate_sequence
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1;

-- tables
CREATE TABLE IF NOT EXISTS layer(
    id UUID PRIMARY KEY NOT NULL,
    name VARCHAR(255),
    extent GEOMETRY,
    last_import_id INTEGER
);

CREATE TABLE IF NOT EXISTS feature(
    id BIGINT PRIMARY KEY NOT NULL,
    attributes JSONB,
    class_id BIGINT,
    geometry GEOMETRY,
    layer_id UUID CONSTRAINT feature_layer_fk REFERENCES layer (id),
    import_id INTEGER
);

-- indices for features
CREATE INDEX IF NOT EXISTS geometry_gix
    ON feature USING gist
    (geometry);

CREATE INDEX IF NOT EXISTS layer_id_fk
    ON feature USING btree
    (layer_id ASC NULLS LAST);