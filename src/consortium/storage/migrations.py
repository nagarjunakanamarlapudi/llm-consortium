"""Schema versioning and migrations."""

from __future__ import annotations

import structlog

from consortium.storage.database import SCHEMA_VERSION, Database

logger = structlog.get_logger()

# Add migration functions here as the schema evolves.
# Each migration is a function that takes a Database and applies changes.
# Keyed by the target version number.
MIGRATIONS: dict[int, str] = {
    # version 1 is the initial schema — no migration needed
}


def check_schema(db: Database) -> bool:
    """Check if the database schema is up to date."""
    current = db.get_schema_version()
    if current is None:
        return False
    return current >= SCHEMA_VERSION


def migrate(db: Database) -> None:
    """Apply any pending migrations to bring the schema up to date."""
    current = db.get_schema_version()
    if current is None:
        logger.info("schema_not_initialized", action="init")
        db.init_schema()
        return

    if current >= SCHEMA_VERSION:
        logger.debug("schema_up_to_date", version=current)
        return

    for target_version in range(current + 1, SCHEMA_VERSION + 1):
        if target_version in MIGRATIONS:
            logger.info("applying_migration", from_version=current, to_version=target_version)
            db.conn.executescript(MIGRATIONS[target_version])
            db.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (target_version,),
            )
            db.conn.commit()
            logger.info("migration_applied", version=target_version)
        else:
            logger.warning("migration_missing", version=target_version)
