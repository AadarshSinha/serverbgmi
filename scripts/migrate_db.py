"""Copy every row from one database into another.

Written for the one-off move off SQLite:

    ./myenv/bin/python scripts/migrate_db.py \
        --source sqlite:///zonepredictor.db \
        --target "postgresql://user:pass@host/dbname?sslmode=require"

It copies users, prediction_logs and payments in foreign-key order, preserves
primary keys, and resets Postgres sequences afterwards -- skip that last part
and the next signup collides with an existing id.

Safe by default: it refuses to write into a target that already holds rows.
"""
import argparse
import os
import sys

os.environ.setdefault("LOAD_MODELS", "false")
os.environ.setdefault("APP_ENV", "development")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, func, inspect, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from models import Payment, PredictionLog, User  # noqa: E402
from extensions import db  # noqa: E402

# Parents before children: prediction_logs and payments both reference users.
MODELS = [User, PredictionLog, Payment]


def normalise(url):
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url


def row_counts(engine):
    counts = {}
    with Session(engine) as session:
        for model in MODELS:
            if inspect(engine).has_table(model.__tablename__):
                counts[model.__tablename__] = session.scalar(
                    select(func.count()).select_from(model)
                )
            else:
                counts[model.__tablename__] = 0
    return counts


def copy(source_url, target_url, force=False, dry_run=False):
    source = create_engine(normalise(source_url))
    target = create_engine(normalise(target_url))

    before = row_counts(source)
    print("source:", ", ".join(f"{t}={n}" for t, n in before.items()))

    db.metadata.create_all(target)

    existing = row_counts(target)
    print("target:", ", ".join(f"{t}={n}" for t, n in existing.items()))

    if any(existing.values()) and not force:
        sys.exit(
            "Target already contains rows. Re-run with --force only if you are "
            "certain you want to add to it."
        )

    if dry_run:
        print("\ndry run — nothing written")
        return

    with Session(source) as src, Session(target) as dst:
        for model in MODELS:
            rows = src.scalars(select(model)).all()
            for row in rows:
                values = {
                    column.name: getattr(row, column.name)
                    for column in model.__table__.columns
                }
                dst.execute(model.__table__.insert().values(**values))
            print(f"copied {len(rows):>5} -> {model.__tablename__}")
        dst.commit()

    # Postgres sequences do not know about ids inserted explicitly.
    if target.dialect.name == "postgresql":
        with Session(target) as dst:
            for model in MODELS:
                dst.execute(
                    db.text(
                        "SELECT setval(pg_get_serial_sequence(:t, 'id'), "
                        "COALESCE((SELECT MAX(id) FROM " + model.__tablename__ + "), 1))"
                    ),
                    {"t": model.__tablename__},
                )
            dst.commit()
        print("sequences reset")

    print("\nfinal:", ", ".join(f"{t}={n}" for t, n in row_counts(target).items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="SQLAlchemy URL to read from")
    parser.add_argument("--target", required=True, help="SQLAlchemy URL to write to")
    parser.add_argument("--force", action="store_true", help="write even if the target has rows")
    parser.add_argument("--dry-run", action="store_true", help="report counts and stop")
    args = parser.parse_args()
    copy(args.source, args.target, force=args.force, dry_run=args.dry_run)
