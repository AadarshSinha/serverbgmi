-- The prediction's own numbers, so the annotated image can be redrawn from the
-- stored screenshot instead of being stored a second time. Safe to run twice.
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS detected_center_x INTEGER;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS detected_center_y INTEGER;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS predicted_center_x INTEGER;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS predicted_center_y INTEGER;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS predicted_radius INTEGER;
