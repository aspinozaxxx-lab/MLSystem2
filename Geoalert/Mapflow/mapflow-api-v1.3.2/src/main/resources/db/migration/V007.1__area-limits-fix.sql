-- Creates entries for all existing processings, setting area to 0
-- The app will reset area to the correct value upon start-up

INSERT INTO processed_area (processing_id, user_id, area, created, updated)
    SELECT prc.id, prj.user_id, 0, prc.created, prc.created
    FROM processing prc JOIN project prj ON prc.project_id = prj.id;