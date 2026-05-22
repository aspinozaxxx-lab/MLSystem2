-- Function definition

CREATE OR REPLACE FUNCTION final_status(statuses INT[]) RETURNS VARCHAR AS $$
DECLARE
    processing_status VARCHAR;
BEGIN
    CASE
        WHEN 1 = ANY(statuses) THEN processing_status := 'IN_PROGRESS';
        WHEN 3 = ANY(statuses) THEN processing_status := 'FAILED';
        WHEN array_length(statuses, 1) = 0 THEN processing_status := 'UNPROCESSED';
        WHEN array_length(statuses, 1) = 1 AND (statuses[1] = 0 OR statuses[1] IS NULL) THEN processing_status := 'UNPROCESSED';
        WHEN 4 = ANY(statuses) THEN processing_status := 'CANCELLED';
        ELSE processing_status := 'OK';
    END CASE;

    RETURN processing_status;
END;
$$ LANGUAGE plpgsql;

--- VIEW processing_estimate

CREATE OR REPLACE VIEW processing_estimate AS
SELECT
  p.id as processing_id,
  p.created as created_at,
  final_status(ARRAY_AGG(w.status)) AS processing_status,
  COALESCE(SUM(w.area), 0) AS total_area,
  COALESCE(SUM(CASE WHEN w.status = 2 THEN w.area ELSE 0 END), 0) AS completed_area,
  COALESCE(
    LEAST(
      100,
      CASE
        WHEN COALESCE(SUM(w.area), 0) = 0 THEN 0
        ELSE (COALESCE(SUM(CASE WHEN w.status = 2 THEN w.area ELSE 0 END), 0) * 100.0) / SUM(w.area)
      END
    ),
    0
  ) AS percent_completed,
  CASE
    WHEN final_status(ARRAY_AGG(w.status)) = 'IN_PROGRESS' THEN
      ROUND(EXTRACT(EPOCH FROM NOW() - MIN(p.created))::NUMERIC *
        CASE
          WHEN COALESCE(SUM(w.area), 0) = 0 THEN null
          WHEN LEAST(
                100,
                (COALESCE(SUM(CASE WHEN w.status = 2 THEN w.area ELSE 0 END), 0) * 100.0) / SUM(w.area)
              ) = 0 THEN null
          ELSE (100 - COALESCE(
              LEAST(
                100,
                (COALESCE(SUM(CASE WHEN w.status = 2 THEN w.area ELSE 0 END), 0) * 100.0) / SUM(w.area)
              ),
              0
              )) /
              LEAST(
                100,
                (COALESCE(SUM(CASE WHEN w.status = 2 THEN w.area ELSE 0 END), 0) * 100.0) / SUM(w.area)
              )
          END
        )
      ELSE
          NULL
    END AS estimate
FROM
  processing p
  LEFT JOIN aoi a ON p.id = a.processing_id
  LEFT JOIN workflow w ON a.id = w.aoi_id
WHERE p.archived = false
GROUP BY
  p.id;

-- Function definition
CREATE OR REPLACE FUNCTION final_status_by_string(statuses VARCHAR[]) RETURNS VARCHAR AS $$
DECLARE
    processing_status VARCHAR;
BEGIN
    CASE
        WHEN 'IN_PROGRESS' = ANY(statuses) THEN processing_status := 'IN_PROGRESS';
        WHEN 'FAILED' = ANY(statuses) THEN processing_status := 'FAILED';
        WHEN array_length(statuses, 1) = 0 THEN processing_status := 'UNPROCESSED';
        WHEN array_length(statuses, 1) = 1 AND (statuses[1] = 'UNPROCESSED' OR statuses[1] IS NULL) THEN processing_status := 'UNPROCESSED';
        WHEN 'CANCELLED' = ANY(statuses) THEN processing_status := 'CANCELLED';
        ELSE processing_status := 'OK';
    END CASE;

    RETURN processing_status;
END;
$$ LANGUAGE plpgsql;

--- VIEW project_estimate
CREATE OR REPLACE VIEW project_estimate AS
select
  pr.id AS project_id,
  final_status_by_string(ARRAY_AGG(pe.processing_status)) AS project_status,
  COALESCE(
    LEAST(
      100,
      CASE
        WHEN COALESCE(SUM(pe.total_area), 0) = 0 THEN 0
        ELSE (COALESCE(SUM(pe.completed_area), 0) * 100.0) / SUM(pe.total_area)
      END
    ),
    0
  ) AS percent_completed,
  max(pe.estimate) AS estimate
from processing_estimate pe
join processing p on pe.processing_id = p.id and p.archived = false
join project pr on pr.id = p.project_id and pr.archived = false
group by pr.id ;