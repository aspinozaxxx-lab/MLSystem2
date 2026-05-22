ALTER TABLE app_user ADD COLUMN review_workflow_enabled BOOLEAN NOT NULL DEFAULT FALSE;


CREATE TABLE processing_review
(
    processing_id UUID    NOT NULL,
    review_status VARCHAR(64) DEFAULT NULL
        CONSTRAINT review_status_check CHECK (review_status IN ('IN_REVIEW', 'ACCEPTED', 'NOT_ACCEPTED', 'REFUNDED')),
    comment TEXT DEFAULT NULL,
    features JSON DEFAULT NULL,
    created       TIMESTAMP,
    updated       TIMESTAMP,
    PRIMARY KEY (processing_id),
    FOREIGN KEY (processing_id) REFERENCES processing (id) ON DELETE CASCADE
);