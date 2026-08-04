"""Run one safe hosted-embedding request without printing the vector."""

from matagent.rag.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
)


def main() -> None:
    try:
        settings = EmbeddingSettings.from_environment()
        vectors = OpenAICompatibleEmbeddingProvider(settings).embed(
            ["wide-bandgap semiconductor materials"]
        )
    except (EmbeddingConfigurationError, EmbeddingError, ValueError) as error:
        raise SystemExit(f"Embedding check failed: {error}") from error

    print("Embedding API connection: OK")
    print(f"Model: {settings.model}")
    print(f"Dimension: {len(vectors[0])}")


if __name__ == "__main__":
    main()
