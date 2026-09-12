# ZonePredictor API

Flask service behind [zonepredictor.com](https://zonepredictor.com). It takes a
BGMI/PUBG minimap screenshot and returns the same image with the predicted next
safe zone drawn on it.

The frontend lives in a separate repository (`zonepredictor/`).

## How a prediction works

1. **YOLOv8** (`Models/bestFull.pt`) finds the safe-zone circle in the upload.
2. **SIFT + FLANN + homography** aligns the screenshot against the stitched
   reference map in `Map/combine_new.png`, converting the circle into template
   coordinates.
3. The quadrant the circle lands in identifies the map; the circle's radius
   identifies the zone phase (thresholds in `constants.json`).
4. A **Keras model per map per phase** (`Models/<map><n><n+1>/model.h5` plus its
   `scaler.pkl`) predicts the next circle's centre.
5. The prediction is transformed back into image space and drawn.

Only **Erangel** and **Miramar** are supported. Vikendi and Sanhok were retired —
their models were trained on too few matches to be trustworthy. The reference
map still contains all four quadrants because that layout defines the coordinate
system, but screenshots from the retired quadrants return a clear
`MAP_NOT_SUPPORTED` error instead of a bad guess.

## Running it

Both environments run the same image, built from the same `Dockerfile`, with the
same gunicorn entrypoint. They differ in one overlay file and nothing else.

| | Local | Server |
| --- | --- | --- |
| Command | `make dev` | `make prod` |
| Compose files | `docker-compose.yml` + `docker-compose.dev.yml` | `docker-compose.yml` |
| `APP_ENV` | `development` | `production` |
| Database | Postgres in the `db` container | managed Postgres |
| Source | mounted from the working tree, `--reload` on | baked into the image |
| Container | foreground, no restart policy | detached, `restart: unless-stopped` |

```bash
cp .env.example .env     # make dev creates this for you if it is missing
make dev                 # http://localhost:4000
```

`make help` lists the rest. `logs`, `shell`, `ps`, `restart` and `down` work the
same in both places, so muscle memory from your laptop transfers to the droplet.

### The database

Postgres in every environment — there is no SQLite in the normal path any more.
`make dev` starts a `db` container alongside the app and publishes it on
`127.0.0.1:5432`, so the container, your venv and any GUI client all talk to the
same database:

```bash
make db                      # psql prompt
psql postgresql://zonepredictor:localdev@localhost:5432/zonepredictor
```

Production points `DATABASE_URL` at a managed Postgres instead. Same engine,
same SQL, same behaviour — only the host differs, which is the entire point:
a query that works locally works on the droplet.

Two things this does not give you. The data is still per-environment: a local
account is not a production account. And `make prod` now refuses a
`DATABASE_URL` containing `localhost` or `@db:`, because that is a development
URL that would only fail at connection time on the server.

> **No container runtime on this machine?** Point `DATABASE_URL` at a free Neon
> or Supabase dev database instead. You get real Postgres with no Docker, which
> keeps local and production on the same engine. `.env.example` has the line.

> **macOS 12 note:** containers do not run on this machine. Docker Desktop needs
> macOS 13, and Colima needs qemu, which has no Monterey bottle and fails to
> build from source. Use the venv below with a hosted Postgres, and prove the
> container on the droplet — `docker compose build` there costs no downtime.

### Tests

```bash
make test       # in the container, against Postgres
```

The suite runs against its own `zonepredictor_test` database on the same
Postgres, so it exercises the engine production uses and can never touch your
development data. Run from the venv with no `TEST_DATABASE_URL` set and it
falls back to a temporary SQLite file per test, so the tests still work with
nothing installed but the venv.

65 tests covering the HTTP contract: auth, prediction feedback, upload storage,
the error codes the frontend branches on, request logging, the billing scaffold
and the production config guard. The ML pipeline is stubbed, so no model files
are needed.

There is also `test.py`, a CLI smoke test against a running server:

```bash
make smoke                                          # curl, no Python needed
./myenv/bin/python test.py TestData/test3.jpg       # same thing via requests
```

### Without Docker

The venv path still works and is the faster loop if you already have one — the
container is not required for local development, just for making local match
production:

```bash
python -m venv myenv
./myenv/bin/python -m pip install -r requirements.txt
./myenv/bin/python app.py                                  # http://localhost:4000
./myenv/bin/python -m unittest discover -s tests -t .
```

`requirements.txt` is the single source of truth for both, so a pin you change
lands in the venv and the image together.

> **macOS note:** if `pip install` fails building `cryptography` from source
> (a Rust toolchain error), rerun with `--only-binary=cryptography`. It is a
> transitive dependency of TensorBoard and is never imported at runtime.

## Deploying with Docker

The app runs in a container; **nginx and certbot stay on the host** so the
existing TLS certificate and its renewal cron keep working untouched.

```
internet ──443──> nginx (host, TLS) ──> 127.0.0.1:4000 ──> container ──> /data volume
```

### First deploy on a fresh droplet

```bash
ssh root@142.93.217.78
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh          # engine + compose plugin

ssh-keygen -t ed25519 -C "droplet"
cat ~/.ssh/id_ed25519.pub                       # paste into GitHub deploy keys
git clone git@github.com:AadarshSinha/serverbgmi.git
cd serverbgmi

cp .env.example .env
nano .env      # APP_ENV=production + JWT_SECRET_KEY + DATABASE_URL + ALLOWED_ORIGINS

make prod                                       # first build ~10-15 min
make logs                                       # wait for "Booting worker"
curl localhost:4000/health
```

`make prod` refuses to start if `.env` is missing, still says
`APP_ENV=development`, or still carries the example JWT secret — the same three
mistakes the app's own production guard would catch, caught before the build
instead of after it.

No `python3-venv`, no `libgl1`, no torch reinstall, no systemd unit — all of
that is in the `Dockerfile` now. `restart: unless-stopped` plus the Docker
daemon's own service is what survives a reboot.

### nginx and TLS (host, once)

```bash
apt install -y nginx certbot python3-certbot-nginx
cp deploy/nginx/api.zonepredictor.com.conf /etc/nginx/sites-available/api.zonepredictor.com
ln -sf /etc/nginx/sites-available/api.zonepredictor.com /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
certbot --nginx -d api.zonepredictor.com        # rewrites the file to add :443
```

Firewall: `ufw allow OpenSSH`, `ufw allow 80,443/tcp`. **Do not open 4000.**
Compose publishes the port as `127.0.0.1:4000:4000`; if you drop that prefix,
Docker's iptables rules jump the queue ahead of ufw and the API is exposed to
the internet regardless of what ufw is told.

### Updating

```bash
cd serverbgmi && make update     # git pull, rebuild, restart, prune old images
```

TensorFlow plus torch make this a large image (expect 4-5GB). Check `df -h`
before the first build. If the droplet is small enough that `pip install`
runs out of memory during the build, build the image somewhere larger, push it
to a registry, and replace `build: .` with `image: ghcr.io/...` on the droplet —
nothing else in the compose file changes.

### Day to day

| Task | Command | Works locally too |
| --- | --- | --- |
| Logs | `make logs` | yes |
| Restart | `make restart` | yes |
| Health | `make ps` (shows healthy/unhealthy) | yes |
| Shell | `make shell` | yes |
| Stop | `make down` | yes |
| Memory / CPU | `docker stats zonepredictor-api` | yes |

Each worker holds its own copy of TensorFlow, the Keras models, the YOLO
detector and the stitched map. `GUNICORN_WORKERS` defaults to 1; raise it in
`.env` only after watching `docker stats`, since two workers also means two
predictions can run at once instead of queueing.

### Migrating off the old venv + systemd setup

```bash
systemctl disable --now zonepredictor    # or it keeps port 4000 and the container never binds
make prod
```

The old `zonepredictor.service` ran `python app.py` — the Flask **development**
server, not gunicorn, despite the runbook's gunicorn line. The container runs
gunicorn properly. The old `gunicorn app:app` invocation no longer works either:
the entrypoint is the factory `app:create_app()`.

Existing data on the droplet needs moving before the old directory is deleted:

```bash
# screenshots collected so far -> the container's volume
docker cp ~/serverbgmi/uploads/. zonepredictor-api:/data/uploads/
docker compose exec api sh -c 'ls /data/uploads | wc -l'
```

The accounts in the old `zonepredictor.db` are SQLite rows; export them into the
managed Postgres before switching `APP_ENV=production`, because the guard will
(correctly) refuse to boot pointing at that file. Note that setting a real
`JWT_SECRET_KEY` invalidates every token issued under the old dev secret —
signed-in users are simply asked to sign in again, which the frontend handles.

### Configuration guards

With `APP_ENV=production` the app **refuses to start** if any of these are wrong,
rather than serving insecurely:

- `JWT_SECRET_KEY` is still the development default
- `DATABASE_URL` still points at SQLite
- `ALLOWED_ORIGINS` is still `*`
- `BILLING_ENABLED` is on without Cashfree credentials

**Put the database and the uploads outside the droplet.** Set `DATABASE_URL` to
a managed Postgres (Neon / Supabase / RDS) and `S3_BUCKET` to a bucket. The
screenshots are the half that cannot be regenerated — the database rows are
metadata about images that would be gone.

Uploads work with any S3-compatible store, not just AWS. Neon Object Storage
lives in the same project as the database and gives 5GB free; Cloudflare R2 and
DigitalOcean Spaces work the same way. Set `S3_ENDPOINT_URL` alongside
`S3_BUCKET` and the keys, and nothing else changes — the same `image_key` that
addressed a file on disk addresses the object in the bucket.

Storing the images **in Postgres** is the one option to avoid: Neon's free plan
is 0.5GB total, shared with your actual data, and inserts start failing when it
fills. At roughly 600KB a screenshot that is about 800 uploads before signups
break. Object storage is 5GB free and, when it is billed, $0.023/GB-month
against $0.35/GB-month for database storage. The
`zp-data` volume survives `docker compose down` and rebuilds, but it does not
survive destroying the droplet — and the production guard will refuse to boot on
SQLite precisely to stop that from becoming the permanent arrangement.

## Configuration

See `.env.example` for the full list with comments. The ones that matter:

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_ENV` | `development` | `production` turns on the startup checks above |
| `PORT` | `4000` | |
| `DATABASE_URL` | SQLite fallback | Postgres everywhere; managed Postgres in production |
| `JWT_SECRET_KEY` | dev default | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ALLOWED_ORIGINS` | `*` | Comma-separated origins in production |
| `MAX_UPLOAD_BYTES` | `10485760` | Returns `FILE_TOO_LARGE` above this |
| `STORE_UPLOADS` | `true` | Keeps screenshots as training data |
| `S3_BUCKET` | unset | Unset stores uploads on local disk |
| `S3_ENDPOINT_URL` | unset | Set for Neon Object Storage / R2 / Spaces; unset means AWS |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | unset | Falls back to the AWS credential chain |
| `BILLING_ENABLED` | `false` | `/billing` endpoints return 503 while off |
| `LOAD_MODELS` | `true` | Tests turn this off |

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/` `/health` | — | Liveness; the frontend polls this for its offline banner |
| POST | `/auth/signup` | — | Create an account, returns a JWT |
| POST | `/auth/login` | — | Returns a JWT |
| GET | `/auth/me` | JWT | Current user |
| POST | `/predict` | optional JWT | The prediction. Returns `image/jpeg` plus an `X-Prediction-Id` header |
| POST | `/predict/<id>/feedback` | optional JWT | Rate a prediction: `spot_on` / `close` / `way_off` |
| GET | `/billing/plans` | — | Plan catalogue |
| POST | `/billing/checkout` | JWT | Cashfree order — 503 until billing is on |
| POST | `/billing/webhook` | signature | Cashfree payment updates |

Accounts are **optional**: predictions work signed out and are unlimited for
everyone. Auth exists so usage can be attributed and so the paid tier has
something to hang off later.

### Error contract

Every failure is JSON — never an HTML error page — shaped like:

```json
{ "code": "ZONE_NOT_DETECTED", "error": "No zone circle was found in that screenshot." }
```

The frontend shows `error` to the user and branches on `code` for extra
guidance.

| Status | Codes |
| --- | --- |
| 400 | `NO_FILE` `EMPTY_FILE` `INVALID_IMAGE` `VALIDATION_ERROR` `INVALID_PLAN` `INVALID_PAYLOAD` `INVALID_RATING` |
| 401 | `INVALID_CREDENTIALS` `INVALID_SIGNATURE` |
| 403 | `ACCOUNT_DISABLED` `NOT_YOUR_PREDICTION` |
| 404 | `UNKNOWN_PREDICTION` |
| 409 | `EMAIL_TAKEN` `ALREADY_RATED` |
| 413 | `FILE_TOO_LARGE` |
| 422 | `ZONE_NOT_DETECTED` `MAP_NOT_MATCHED` `UNKNOWN_MAP` `UNKNOWN_ZONE` `FINAL_ZONE` `MAP_NOT_SUPPORTED` `FEEDBACK_WINDOW_CLOSED` |
| 500 | `INTERNAL_ERROR` `ENCODE_FAILED` |
| 503 | `MODEL_UNAVAILABLE` `BILLING_DISABLED` `BILLING_NOT_IMPLEMENTED` |

## Data collected

Every `/predict` call writes one `PredictionLog` row: timestamp, user (if signed
in), outcome and error code, map and zone phase, duration, stored image size, a
**hashed** client IP, the storage key of the stored result image, the geometry
of the prediction itself, and — if the user answered — how good they said it
was. The
screenshots themselves are retained as training data, which the website
discloses. Raw IPs are never stored.

## Prediction quality

After a successful prediction the site asks one question with three answers:
**Spot on**, **Close**, **Way off**. One click, and every answer means
something — a five-point scale mostly collects 3s, and each extra click costs
response rate.

The answer lands on the prediction's own row, so it is already joined to the
map, the zone phase, the model version in use at the time and the stored
screenshot. Useful queries:

```sql
-- Where is the model weakest?
SELECT map_type, zone_number,
       COUNT(*) FILTER (WHERE rating = 'spot_on') AS spot_on,
       COUNT(*) FILTER (WHERE rating = 'close')   AS close,
       COUNT(*) FILTER (WHERE rating = 'way_off') AS way_off
FROM prediction_logs
WHERE rating IS NOT NULL
GROUP BY map_type, zone_number
ORDER BY map_type, zone_number;

-- How many people actually answer?
SELECT COUNT(*) AS predictions,
       COUNT(rating) AS rated,
       ROUND(100.0 * COUNT(rating) / NULLIF(COUNT(*), 0), 1) AS percent_rated
FROM prediction_logs
WHERE succeeded;

-- The screenshots worth retraining on first.
SELECT id, created_at, map_type, zone_number, image_key
FROM prediction_logs
WHERE rating = 'way_off'
ORDER BY created_at DESC
LIMIT 50;
```

That last one is the point of the whole feature: a queue of real failures with
the original screenshot attached.

### What gets stored

One image per successful prediction: the **annotated result** — the screenshot
with the predicted circle and direction arrow drawn on, exactly what the user
was shown. Fetch any of them by row id:

```bash
./myenv/bin/python scripts/get_prediction_image.py 42
```

A failed prediction has no result image, so it stores none; the row is still
written with its error code.

Each row also carries the geometry the drawing was made from —
`detected_center_x/y`, `predicted_center_x/y`, `predicted_radius`. Those are
what you query: average error by zone phase, compare model versions on the same
inputs, or join against the rating to see how far off "way off" really was. An
image cannot answer any of that.

> One consequence worth knowing: because the stored image has the prediction
> painted over the pixels, it cannot be used as-is to retrain. Feature matching
> and the zone detector would be reading our own overlay. If retraining on real
> traffic becomes the priority, store the upload instead — a one-line change in
> `routes/predict.py` — or store both under separate keys.

A prediction can be rated once, within 24 hours, and only if it succeeded. One
made while signed in can only be rated by that same user; anonymous predictions
are open, because most predictions are made signed out and requiring an account
would throw away most of the signal.

## Layout

```
Makefile                 dev / prod / test / logs — same verbs in both places
Dockerfile               CPU-only torch, opencv system libs, gunicorn entrypoint
docker-compose.yml       the service, its volume and the loopback port binding
docker-compose.dev.yml   local overlay: mounted source, reload, Postgres
deploy/nginx/            host reverse-proxy config (certbot edits it in place)
deploy/postgres/         creates the test database on first volume init

app.py           application factory, error handlers
config.py        environment-driven config + production guard
predictor.py     the ML pipeline (the only module that imports TF/YOLO)
models.py        User, PredictionLog, Payment
storage.py       uploads to any S3-compatible store or local disk
scripts/         database migration, fetching stored result images
errors.py        ApiError / PredictionError
extensions.py    db, migrate, jwt, cors singletons
routes/          health, auth, predict, billing blueprints
tests/           unittest suite
```

## Enabling payments later

The Cashfree scaffold is deliberately inert. To switch it on: set
`BILLING_ENABLED=true` plus the `CASHFREE_*` credentials, then fill in the single
`TODO(billing)` in `routes/billing.py` that creates the Cashfree order and
returns its `payment_session_id`. The order table, plan catalogue and webhook
signature verification already work.
