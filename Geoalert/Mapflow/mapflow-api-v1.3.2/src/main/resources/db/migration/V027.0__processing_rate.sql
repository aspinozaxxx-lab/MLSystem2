CREATE TABLE processing_rating
(
    processing_id UUID NOT NULL,
    rating        INTEGER NOT NULL CONSTRAINT rating_range CHECK ( rating > 0 and rating <= 5 ),
    feedback      TEXT    DEFAULT NULL,
    created  TIMESTAMP,
    updated  TIMESTAMP,
    PRIMARY KEY(processing_id),
    FOREIGN KEY(processing_id) REFERENCES processing (id) ON DELETE CASCADE
);
