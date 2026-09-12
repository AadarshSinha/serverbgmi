-- Adds the feedback columns to a database whose prediction_logs table already
-- exists. A database created after this change gets them from create_all() and
-- does not need this file. Safe to run twice.
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS rating VARCHAR(16);
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS rated_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS ix_prediction_logs_rating ON prediction_logs (rating);
