CREATE INDEX IF NOT EXISTS workflow_status_update_index ON workflow (status ASC NULLS LAST, status_update_date ASC NULLS LAST) INCLUDE(status, status_update_date, required_action);
