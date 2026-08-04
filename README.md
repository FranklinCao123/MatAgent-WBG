# MatAgent-WBG

MatAgent-WBG is a lightweight, tool-using scientific Agent for screening
wide-bandgap semiconductor materials. It combines LLM reasoning, validated tool
calls, Materials Project data, scientific RAG, persistent LangGraph state, and a
citation-checked final report without local models, a GPU, or a large dataset.

## What this project demonstrates

- Natural-language requirements → strict Pydantic objects.
- LangGraph planning, conditional routing, state, and SQLite checkpoints.
- OpenAI-compatible DeepSeek JSON output and function calling.
- Allow-listed tools with locally validated arguments and results.
- Real Materials Project candidate search with explicit hard filters.
- Semantic Scholar ingestion, hosted BGE-M3 embeddings, and Supabase pgvector.
- Candidate-specific evidence retrieval with exact material-tag filtering.
- Grounded LLM synthesis that rejects unknown candidates and invented DOI values.
- Deterministic offline fallback, bounded retries, trace output, and retrieval metrics.

This is intentionally a single workflow Agent, not an open-ended multi-agent
system. Scientific filtering and ranking remain deterministic and auditable;
the LLM handles language, tool selection, and evidence-bounded communication.

## Workflow

```mermaid
flowchart TD
    START([User query]) --> INIT[initialize_run]
    INIT --> PARSE[parse_requirements]
    PARSE -->|valid requirements| PLAN[plan_screening]
    PARSE -->|parsing error| REPORT[generate_report]
    PLAN --> DECIDE[decide_tools]
    DECIDE -->|tool selected| EXECUTE[execute_tools]
    DECIDE -->|no tool or error| REPORT
    EXECUTE -->|candidates found| RANK[rank_candidates]
    EXECUTE -->|no candidates or error| REPORT
    RANK --> RAG[retrieve_evidence]
    RAG --> REPORT
    REPORT --> END([Final report])
```

Every stage writes a compact execution record to `tool_history`. Unknown tools,
invalid arguments, malformed API data, and invented citations are rejected
before they reach the final answer.

## Project structure

```text
matagent/
├── cli.py                 # Main Agent CLI
├── config.py              # Environment configuration and dependency wiring
├── graph.py               # LangGraph assembly and routing
├── schemas.py             # Domain and final-report contracts
├── state.py               # Shared Agent state
├── persistence.py         # SQLite checkpoint helpers
├── material_names.py      # Formula/phase alias normalization
├── llm/                   # Offline + DeepSeek reasoning components
├── tools/                 # Tool registry and scientific tool adapters
├── workflow/              # Parsing, planning, ranking, RAG, and reporting nodes
├── rag/                   # Supabase, embedding, ingestion, and literature clients
└── evaluation/            # Precision@k, Recall@k, MRR, and HitRate@k

sql/                       # Three idempotent Supabase setup scripts
tests/                     # Network-free unit and workflow tests
mock_materials.json        # Explicitly illustrative offline data
```

## Install

Python 3.10+ is supported. On the existing laptop environment:

```powershell
conda activate agentenv
python -m pip install --no-cache-dir -r requirements.txt
```

Copy `.env.example` to `.env` and add keys locally. `.env` is ignored by Git.

```powershell
Copy-Item .env.example .env
notepad .env
git check-ignore .env
```

Required keys depend on the selected features:

| Feature | Setting |
|---|---|
| DeepSeek reasoning and final synthesis | `MATAGENT_LLM_API_KEY` |
| Materials Project search | `MATAGENT_MP_API_KEY` |
| Literature discovery | `MATAGENT_S2_API_KEY` |
| Hosted BGE-M3 embeddings | `MATAGENT_EMBEDDING_API_KEY` |
| Supabase RAG | `MATAGENT_SUPABASE_URL`, `MATAGENT_SUPABASE_SECRET_KEY` |

Never put secrets in source code, CLI arguments, state, traces, or commits.

## Run

Fast, free, deterministic smoke test:

```powershell
python -m matagent.cli --mode offline --material-backend mock --show-trace `
  "寻找带隙大于3 eV、适合高温功率器件并优先考虑高热导率的材料"
```

Complete real-data Agent run:

```powershell
python -m matagent.cli --mode deepseek `
  --material-backend materials-project --fetch-limit 100 --report-limit 10 `
  --rag --evidence-top-k 2 --evidence-candidate-limit 5 --show-trace `
  "寻找带隙大于3 eV、适合高温功率器件并优先考虑高热导率的材料"
```

DeepSeek mode makes three bounded LLM calls: requirement parsing, tool
selection, and a concise Top-3 structured synthesis. If RAG or synthesis fails,
the deterministic screening report is still returned with an explicit diagnostic.

## RAG setup

Run these files in the Supabase SQL Editor, in order:

1. `sql/001_supabase_rag.sql` — restricted health RPC;
2. `sql/002_rag_evidence_store.sql` — documents, chunks, HNSW index, search RPC;
3. `sql/003_rag_ingestion.sql` — DOI-idempotent atomic ingestion RPC.

Then verify the two external layers without printing secrets or vectors:

```powershell
python -m matagent.rag.check_database
python -m matagent.rag.check_embedding
```

The database stores 1024-dimensional BGE-M3 vectors. Re-ingesting a DOI updates
its chunks while preserving the union of existing and new material tags.

## Build a small literature corpus

Search only, with no database write:

```powershell
python -m matagent.rag.literature_cli search `
  "wide bandgap semiconductor thermal conductivity" --limit 5
```

Controlled multi-material dry-run:

```powershell
python -m matagent.rag.literature_cli bootstrap `
  --material Diamond --material AlN --material beta-Ga2O3 `
  --material 4H-SiC --material GaN `
  --papers-per-material 2 --search-limit 10
```

The planner requires DOI-backed abstracts, checks material mentions and year,
normalizes aliases, spaces requests to respect the 1 RPS introductory limit,
retries HTTP 429 with bounded backoff, and deduplicates by DOI. Review the
venue/citation preview, then repeat the same command with `--write`.

For a portfolio demo, a reviewed corpus of roughly 10–15 relevant abstracts is
preferable to thousands of uncurated records.

## Checkpointing

```powershell
python -m matagent.cli --mode offline --thread-id demo-001 `
  --show-checkpoints "寻找高温功率半导体材料"
```

The default `.matagent/checkpoints.sqlite3` stores execution state and history,
not API keys. Thread IDs are isolated. This is durable workflow memory, not a
chat-history feature.

## Lightweight web interface

The browser UI is a thin layer over the same LangGraph runtime used by the CLI.
It keeps API keys on the server and exposes only a screening request, ranked
candidates, retrieved evidence, the final report, and an optional execution
trace.

```powershell
python -m uvicorn matagent.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`. The interface intentionally has no authentication,
file upload, background queue, or frontend framework; bind it to localhost for
development.

## Tests

Tests use fake HTTP and LLM clients and make no paid calls:

```powershell
python -m unittest discover -s tests -v
```

Retrieval evaluation is available through
`matagent.evaluation.evaluate_retrieval` using reviewed query/DOI labels.

## Scientific boundaries

- Mock properties and weighted scores are demonstration-only.
- Materials Project values are computed screening data, not device guarantees.
- The live backend does not currently provide thermal conductivity or breakdown
  field, so those properties are not fabricated or silently ranked.
- Retrieved abstracts support traceability but do not replace full-paper review.
- Evidence count is not used as a material-performance score.
- Manufacturability, defects, doping, contacts, cost, and uncertainty remain
  explicit limitations of this lightweight screening stage.

The project is complete as an auditable Agent engineering demonstration, while
remaining honest about the boundary between software orchestration and validated
materials science.
