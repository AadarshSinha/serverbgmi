-- Runs once, the first time the postgres volume is created. Gives `make test`
-- its own database so a test run can never truncate your development data.
CREATE DATABASE zonepredictor_test;
GRANT ALL PRIVILEGES ON DATABASE zonepredictor_test TO zonepredictor;
