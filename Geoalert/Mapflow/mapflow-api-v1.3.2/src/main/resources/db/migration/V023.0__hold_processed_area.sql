ALTER TABLE processed_area ADD COLUMN IF NOT EXISTS hold BOOLEAN DEFAULT FALSE;

UPDATE processed_area SET hold = TRUE WHERE processing_id IN (
    SELECT prc.id from processing prc
        JOIN aoi ON aoi.processing_id = prc.id
    WHERE aoi.status = 1 AND prc.archived = FALSE
    GROUP BY prc.id
    );

ALTER TABLE processed_area DROP CONSTRAINT processed_area_pkey;
ALTER TABLE processed_area ADD PRIMARY KEY (processing_id, user_id);