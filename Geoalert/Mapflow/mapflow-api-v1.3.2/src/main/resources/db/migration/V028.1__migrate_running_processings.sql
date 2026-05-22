DELETE FROM processed_area
WHERE processing_id in (
    SELECT processing.id
    FROM processing
             LEFT JOIN aoi ON aoi.processing_id = processing.id
             LEFT JOIN workflow ON workflow.aoi_id = aoi.id
    GROUP BY processing.id
    HAVING bool_or(workflow.status = 1) IS TRUE
);

INSERT INTO processed_area (processing_id, user_id, area, created, updated, hold)
SELECT processing.id, project.user_id, sum(aoi.area), processing.created, now(), TRUE
FROM processing
         LEFT JOIN project ON processing.project_id = project.id
         LEFT JOIN aoi ON aoi.processing_id = processing.id
         LEFT JOIN workflow ON workflow.aoi_id = aoi.id
GROUP BY processing.id, project.user_id
HAVING bool_or(workflow.status = 1) IS TRUE
;
