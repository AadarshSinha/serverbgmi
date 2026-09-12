# One set of verbs for both environments. `make dev` on a laptop and
# `make prod` on the droplet run the same image, built from the same
# Dockerfile, with the same gunicorn entrypoint — they differ only in the
# overlay file and whether the container detaches.

COMPOSE     := docker compose
DEV_FILES   := -f docker-compose.yml -f docker-compose.dev.yml
SERVICE     := api
IMAGE       ?= TestData/test3.jpg
BASE_URL    ?= http://localhost:4000

.PHONY: help dev prod update test smoke lint-config logs shell db ps restart down clean

help:
	@echo "Local:"
	@echo "  make dev       start in the foreground with live reload"
	@echo "  make test      run the test suite inside the container"
	@echo "  make smoke     upload a real screenshot and save the prediction"
	@echo "  make db        psql prompt on the development database"
	@echo "Server:"
	@echo "  make prod      build and start detached"
	@echo "  make update    git pull, rebuild, restart, prune old images"
	@echo "Both:"
	@echo "  make logs / shell / ps / restart / down"

# --- local -------------------------------------------------------------------

dev: .env
	$(COMPOSE) $(DEV_FILES) up --build

# TEST_DATABASE_URL points the suite at its own database on the same Postgres
# the app uses, so the tests exercise the engine production runs — but never
# the database your development data lives in.
test: .env
	$(COMPOSE) $(DEV_FILES) run --rm \
		-e LOAD_MODELS=false \
		-e TEST_DATABASE_URL=postgresql://zonepredictor:localdev@db:5432/zonepredictor_test \
		$(SERVICE) python -m unittest discover -s tests -t .

# End-to-end check against a running server: uploads a real screenshot and
# expects an annotated JPEG back. Deliberately curl rather than test.py, so it
# needs no Python, no venv and no packages on the host.
#   make smoke
#   make smoke IMAGE=TestData/test1.jpg
#   make smoke BASE_URL=https://api.zonepredictor.com
smoke:
	@code=$$(curl -sS -o /tmp/zp-prediction.out -w '%{http_code}' \
		-F file=@$(IMAGE) $(BASE_URL)/predict); \
	if file /tmp/zp-prediction.out | grep -qi 'jpeg\|jpg image'; then \
		mv /tmp/zp-prediction.out /tmp/zp-prediction.jpg; \
		echo "HTTP $$code — prediction saved to /tmp/zp-prediction.jpg"; \
		open /tmp/zp-prediction.jpg 2>/dev/null || true; \
	else \
		echo "HTTP $$code — $$(cat /tmp/zp-prediction.out)"; \
		rm -f /tmp/zp-prediction.out; \
		exit 1; \
	fi

# --- server ------------------------------------------------------------------

prod: lint-config
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

update: lint-config
	git pull
	$(COMPOSE) up -d --build
	docker image prune -f
	$(COMPOSE) ps

# Fail before starting rather than after, and with a sentence that says what to
# fix. The app has its own production guard; this one just catches the common
# case of deploying with the example file untouched.
lint-config:
	@if [ ! -f .env ]; then echo "No .env — run: cp .env.example .env && nano .env"; exit 1; fi
	@if ! grep -q '^APP_ENV=production' .env; then echo ".env does not set APP_ENV=production"; exit 1; fi
	@if grep -q '^JWT_SECRET_KEY=dev-only-insecure-secret' .env; then echo "JWT_SECRET_KEY is still the development default"; exit 1; fi
	@if grep -qE '^DATABASE_URL=.*(localhost|127\.0\.0\.1|@db:)' .env; then echo "DATABASE_URL points at a local database — production needs the managed one"; exit 1; fi
	@if grep -qE '^[A-Z_]+=.*[<>]' .env; then echo "Unfilled placeholders in .env:"; grep -nE '^[A-Z_]+=.*[<>]' .env | cut -d= -f1; exit 1; fi

# --- both --------------------------------------------------------------------

logs:
	$(COMPOSE) logs -f --tail=100

shell:
	$(COMPOSE) exec $(SERVICE) sh

# A psql prompt on the development database.
db:
	$(COMPOSE) $(DEV_FILES) exec db psql -U zonepredictor zonepredictor

ps:
	$(COMPOSE) ps

restart:
	$(COMPOSE) restart

down:
	$(COMPOSE) down

# Also removes the uploads and Postgres volumes. Local housekeeping only.
clean:
	$(COMPOSE) $(DEV_FILES) down -v

# Created on first `make dev`; `make prod` refuses to invent one, because
# guessing at production settings is how a server ends up serving the
# development defaults.
.env:
	cp .env.example .env
	@echo "Created .env from .env.example — review it before running anything real."
