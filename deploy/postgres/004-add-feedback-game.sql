-- One-tap answer to "which game do you play?", and message becomes optional
-- so a tap alone is a complete response. Safe to run twice.
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS game VARCHAR(16);
CREATE INDEX IF NOT EXISTS ix_feedback_game ON feedback (game);
ALTER TABLE feedback ALTER COLUMN message DROP NOT NULL;
