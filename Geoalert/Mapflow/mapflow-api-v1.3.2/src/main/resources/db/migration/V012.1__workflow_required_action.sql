DO
$$
    BEGIN
        IF NOT EXISTS (SELECT * FROM pg_type typ INNER JOIN pg_namespace nsp ON nsp.oid = typ.typnamespace
            WHERE nsp.nspname = current_schema() AND typ.typname = 'workflow_required_action')
            THEN
            CREATE TYPE workflow_required_action AS ENUM ('start', 'restart', 'cancel');
        END IF;
    END;
$$
LANGUAGE plpgsql;

ALTER TABLE workflow ADD COLUMN IF NOT EXISTS required_action workflow_required_action;
ALTER TABLE workflow DROP COLUMN IF EXISTS start_requested;