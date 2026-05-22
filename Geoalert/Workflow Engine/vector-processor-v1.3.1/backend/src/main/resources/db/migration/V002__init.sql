-- Create a FUNCTION to calc_area
CREATE OR REPLACE FUNCTION calc_area()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.attributes->'area' IS NOT NULL THEN
    NEW.attributes = jsonb_set(NEW.attributes, '{area}', to_jsonb(ROUND(st_area(NEW.geometry::geography)::numeric, 2)), true);
  END IF;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;
-- Create the trigger
CREATE TRIGGER area_calculate BEFORE INSERT ON feature
FOR EACH ROW
EXECUTE PROCEDURE calc_area();

-- Create the UPDATE trigger
CREATE TRIGGER area_calculate_update BEFORE UPDATE ON feature
FOR EACH ROW
EXECUTE PROCEDURE calc_area();
