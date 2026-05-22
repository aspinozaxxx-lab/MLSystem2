DELETE FROM processed_area
WHERE processing_id in (
    SELECT processing.id
    FROM processing
    LEFT JOIN aoi ON aoi.processing_id = processing.id
    LEFT JOIN workflow ON workflow.aoi_id = aoi.id
    GROUP BY processing.id
    HAVING bool_or(workflow.status = 3) IS TRUE
);
