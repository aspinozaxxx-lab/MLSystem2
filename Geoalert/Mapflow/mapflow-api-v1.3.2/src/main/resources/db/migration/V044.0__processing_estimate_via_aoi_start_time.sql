ALTER TABLE aoi ADD COLUMN start_time TIMESTAMP DEFAULT NULL;
CREATE OR REPLACE VIEW processing_estimate AS
SELECT
  p.id as processing_id,
  p.created as created_at,
  final_status(ARRAY_AGG(DISTINCT w.status)) AS processing_status,
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
    WHEN final_status(ARRAY_AGG(DISTINCT w.status)) = 'IN_PROGRESS' THEN
      ROUND(EXTRACT(EPOCH FROM NOW() - MIN(a.start_time))::NUMERIC *
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