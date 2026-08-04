"""Command-line health check for the configured RAG database."""

from matagent.rag.database import (
    DatabaseConfigurationError,
    DatabaseHealthError,
    check_database,
    settings_from_environment,
)


def main() -> None:
    try:
        health = check_database(settings_from_environment())
    except (DatabaseConfigurationError, DatabaseHealthError) as error:
        raise SystemExit(f"Database check failed: {error}") from error

    print("Database connection: OK")
    print(f"Database: {health.database_name}")
    print(f"PostgreSQL: {health.postgres_version}")
    print(f"pgvector: {health.vector_version}")


if __name__ == "__main__":
    main()
