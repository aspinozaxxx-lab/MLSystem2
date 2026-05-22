--- VIEW project_estimate
CREATE OR REPLACE VIEW project_estimate AS
select
  pr.id AS project_id,
  final_status_by_string(ARRAY_AGG(DISTINCT pe.processing_status)) AS project_status,
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
  CASE
    WHEN MAX(CASE WHEN pe.estimate IS NULL AND pe.processing_status = 'IN_PROGRESS' THEN 1 ELSE 0 END) = 1 THEN null
    ELSE MAX(pe.estimate)
  END AS estimate
from processing_estimate pe
join processing p on pe.processing_id = p.id and p.archived = false
join project pr on pr.id = p.project_id and pr.archived = false
group by pr.id ;