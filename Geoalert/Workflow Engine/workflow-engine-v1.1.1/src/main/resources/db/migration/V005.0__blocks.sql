ALTER TABLE workflow_definition_ver ADD COLUMN block_config json;
ALTER TABLE workflow ADD COLUMN block_params json;