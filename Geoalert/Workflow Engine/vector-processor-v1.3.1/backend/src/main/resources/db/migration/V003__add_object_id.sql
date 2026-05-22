-- Create a FUNCTION to calc_area AND insert object_id
CREATE OR REPLACE FUNCTION add_uuid_and_area()
RETURNS TRIGGER AS $$
BEGIN
  NEW.attributes = jsonb_set(NEW.attributes, '{object_id}', to_jsonb(gen_random_uuid()), true);
  IF NEW.attributes->'area' IS NOT NULL THEN
    NEW.attributes = jsonb_set(NEW.attributes, '{area}', to_jsonb(ROUND(st_area(NEW.geometry::geography)::numeric, 2)), true);
  END IF;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- Update the trigger
CREATE OR REPLACE TRIGGER area_calculate BEFORE INSERT ON feature
FOR EACH ROW
EXECUTE PROCEDURE add_uuid_and_area();

-- Rename trigger for clarity
ALTER TRIGGER area_calculate ON feature RENAME TO tr_on_feature_insert;