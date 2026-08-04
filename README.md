# MatAgent-WBG

Lightweight local prototype for an LLM-based scientific agent that screens
wide-bandgap semiconductor materials.

## Current scope

The current version supports offline or DeepSeek reasoning and mock or live
Materials Project search backends:

1. Parse a natural-language request into validated screening requirements.
2. Enrich the request with a transparent demonstration domain policy.
3. Build a request-aware ranking plan.
4. Ask an offline or DeepSeek selector to propose a tool call.
5. Validate the tool name and arguments through an allow-listed registry.
6. Search a tiny local mock dataset or a capped Materials Project summary query
   using exact `>` or `>=` semantics.
7. Route conditionally based on tool results.
8. Rank candidates using the generated plan when results exist.
9. Generate a Markdown report, including inferred requirements and errors.

Offline reasoning with the mock backend requires no API, GPU, scientific model,
vector database, or server. All mock property values are illustrative and are
not suitable for scientific or engineering decisions.

Requirement parsing and tool selection support two modes:

- `offline`: deterministic local implementations with no network or API cost.
- `deepseek`: DeepSeek JSON Output plus function calling, both followed by
  local Pydantic validation.

## Project structure

```text
matagent/
|-- schemas.py               # Screening and ranking data contracts
|-- config.py                # Environment-based LLM configuration
|-- domain_policy.py         # Auditable conventional-device filter defaults
|-- state.py                 # Shared LangGraph state
|-- graph.py                 # Graph construction and conditional routing
|-- cli.py                   # Command-line interface
|-- workflow/
|   |-- core.py              # Parsing, tool selection, and execution nodes
|   |-- screening.py         # Screening policy and candidate ranking
|   `-- report.py            # Markdown report generation
|-- llm/
|   |-- base.py              # Parser interface and shared client setup
|   |-- rule_based.py        # Offline requirement parser
|   |-- deepseek.py          # DeepSeek JSON requirement parser
|   `-- tool_selector.py     # Offline and DeepSeek tool selectors
`-- tools/
    |-- material_search.py   # Replaceable material-search implementation
    |-- materials_project.py # Lightweight Materials Project REST backend
    |-- registry.py          # Allow-listed execution registry
    `-- schemas.py           # Validated tool arguments and results

tests/
|-- test_workflow.py         # Standard-library workflow tests
|-- test_persistence.py      # SQLite checkpoint tests
`-- test_materials_project.py # Remote-backend tests with fake HTTP
```

## Install

Activate the existing Conda environment and install the small Python
dependencies:

```powershell
conda activate agentenv
python -m pip install --no-cache-dir -r requirements.txt
```

## Run offline

```powershell
python -m matagent.cli --mode offline --show-trace `
  "寻找适合高温功率电子器件的宽禁带半导体材料"
```

## Run with DeepSeek

Copy `.env.example` to `.env`, insert a newly created API key locally, and
never commit `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
git check-ignore .env
```

Then run:

```powershell
python -m matagent.cli --mode deepseek --show-trace `
  "寻找带隙大于3 eV、适合高温功率器件并优先考虑高热导率的材料"
```

Configuration variables:

```text
MATAGENT_LLM_MODE=deepseek
MATAGENT_LLM_API_KEY=<local secret>
MATAGENT_LLM_MODEL=deepseek-v4-flash
MATAGENT_LLM_BASE_URL=https://api.deepseek.com
MATAGENT_MATERIAL_BACKEND=mock
MATAGENT_MP_API_KEY=<local secret>
MATAGENT_MP_BASE_URL=https://api.materialsproject.org
MATAGENT_MP_FETCH_LIMIT=100
MATAGENT_MP_TIMEOUT_SECONDS=20
```

## Search Materials Project

Add `MATAGENT_MP_API_KEY` to the local `.env` file, then run:

```powershell
python -m matagent.cli --mode deepseek `
  --material-backend materials-project --fetch-limit 100 `
  --report-limit 10 --show-trace `
  "寻找带隙大于3 eV、适合高温功率器件的材料"
```

The lightweight REST backend adds no new third-party dependency. It requests
only material ID, formula, band gap, stability, energy above hull, and formation
energy. The API key is sent in the `X-API-KEY` header and is never put in graph
state, trace output, or the request URL. The API candidate pool is capped at 100
lightweight records by default, while the report displays 10.

The live planner enforces a transparent first-pass quality policy before
ranking: exclude elements without conventionally stable isotopes for the
default non-nuclear device workflow, require `is_metal=false`, and cap energy
above hull at 0.1 eV/atom. Materials Project applies the supported subset on the
server, then local Pydantic-validated code repeats the exact checks and records
rejection reasons. The API limits the `exclude_elements` string to 60
characters, so the server receives a high-priority subset while local checking
always uses the complete policy list.

Live results use a transparent lexicographic order: stable entries first, then
lower energy above hull, then higher band gap. Thermal conductivity and
breakdown field are unavailable in this first integration, so they are reported
as missing and are not used for ranking. The mock backend retains its
demonstration-only weighted ranking.

## Workflow

```text
parse_requirements
|-- success -> plan_screening -> decide_tools
|                                  |-- tool call -> execute_tools
|                                  |                 |-- candidates -> rank_candidates
|                                  |                 `-- empty/error ---------.
|                                  `-- no call/error ------------------------|
`-- error -------------------------------------------------------------------|
                                                                             v
                                                                      generate_report
```

DeepSeek proposes calls but never executes Python directly. `ToolRegistry`
rejects unknown names, validates arguments with Pydantic, and invokes only
locally registered read-only tools. The first implementation allows one tool
selection round per run.

`plan_screening` is a transparent demonstration policy, not a validated
scientific ranking method. It records inferred domain requirements and creates
normalized weights that sum to 1.0. The ranking node consumes this plan instead
of fixed weights.

## Local checkpoint persistence

Pass a thread ID to save every LangGraph super-step in a lightweight local
SQLite database:

```powershell
python -m matagent.cli --mode offline --thread-id demo-001 `
  --show-checkpoints "寻找适合高温功率电子器件的材料"
```

The default database path is:

```text
.matagent/checkpoints.sqlite3
```

The `.matagent/` directory is ignored by Git. A custom path can be supplied:

```powershell
python -m matagent.cli --thread-id demo-001 `
  --checkpoint-db D:\path\to\checkpoints.sqlite3 "query"
```

Each checkpoint records the current status and the next node. Reusing a thread
preserves its checkpoint history, while `initialize_run` clears transient
candidate, tool-call, error, and report state before a new run. Different
thread IDs remain isolated.

Checkpointing currently provides durable execution history, not conversational
follow-up understanding. Message history and requirement merging will be added
separately. The local SQLite file is not encrypted and can contain user queries,
tool results, and generated reports; it never stores the LLM API key.

## Test

Tests use Python's standard library and fake DeepSeek clients, so they do not
make paid API calls:

```powershell
python -m unittest discover -s tests -v
```

## RAG database health check

This prototype reaches Supabase PostgreSQL through the HTTPS Data API, avoiding
a direct port-5432 dependency. First run `sql/001_supabase_rag.sql` in the
Supabase SQL Editor. Then configure these server-only values in `.env`:

```dotenv
MATAGENT_SUPABASE_URL=https://your-project.supabase.co
MATAGENT_SUPABASE_SECRET_KEY=sb_secret_your_key
```

Run:

```powershell
python -m matagent.rag.check_database
```

The command calls a restricted, read-only RPC and prints only the database name,
PostgreSQL version, and pgvector version. It never prints the URL or secret key.

## RAG evidence schema

After the health check succeeds, run `sql/002_rag_evidence_store.sql` in the
Supabase SQL Editor. It creates:

- `rag_documents` for paper-level source metadata;
- `rag_document_chunks` for traceable passages and 1024-dimensional vectors;
- an HNSW cosine-distance index for approximate nearest-neighbor search;
- `match_rag_chunks(...)`, a server-only similarity-search RPC.

The dimension is fixed to the planned `BAAI/bge-m3` dense embedding contract.
Embedding generation and document ingestion are separate modules and are not
implemented in this migration.

## Hosted embeddings

Configure the server-only embedding settings shown in `.env.example`. The
default uses SiliconFlow's OpenAI-compatible `BAAI/bge-m3` endpoint and returns
1024-dimensional vectors without downloading the model locally.

Verify the configured service with one small request:

```powershell
python -m matagent.rag.check_embedding
```

The command prints only the model name and returned dimension. It never prints
the API key or embedding values. The provider batches inputs, preserves response
order, and rejects malformed or dimensionally incompatible vectors before they
reach PostgreSQL.

## Document ingestion

Run `sql/003_rag_ingestion.sql` in the Supabase SQL Editor. The RPC ingests one
document and all of its chunks in a single PostgreSQL transaction. Re-ingesting
the same DOI updates the document and replaces its old chunks.

Ingest a UTF-8 text or Markdown document:

```powershell
python -m matagent.rag.ingest_document `
  --file .\data\paper.txt `
  --title "Paper title" `
  --doi "10.xxxx/example" `
  --source-url "https://example.org/paper" `
  --year 2025 `
  --material "4H-SiC"
```

The default chunker uses 1,800-character windows with 200-character overlap and
prefers paragraph or sentence boundaries. PDF parsing is intentionally outside
this module; source text must already be available as UTF-8 `.txt` or `.md`.

## Literature discovery

Preview a small, relevance-ranked set of Semantic Scholar records before any
database write:

```powershell
python -m matagent.rag.literature_cli search `
  "wide bandgap semiconductor thermal conductivity" --limit 5
```

Only records containing both an abstract and DOI are shown as usable. After
reviewing the title, DOI, year, and open-access status, ingest one selected paper:

```powershell
python -m matagent.rag.literature_cli ingest PAPER_ID `
  --material "4H-SiC"
```

The API key is optional at the protocol level, but unauthenticated requests can
receive HTTP 429 rate limits. Configure `MATAGENT_S2_API_KEY` for reliable
authenticated access. Search never writes to the database; only the explicit
`ingest` subcommand generates embeddings and writes evidence.

## Agent RAG retrieval

Enable query-time evidence retrieval explicitly:

```powershell
python -m matagent.cli --mode deepseek `
  --material-backend materials-project `
  --rag --evidence-top-k 5 `
  "寻找适合高温功率器件的宽禁带材料"
```

After material ranking, the graph calls the allow-listed
`retrieve_scientific_evidence` tool. It embeds the user query together with the
leading candidate names, retrieves pgvector matches, and adds source title, DOI,
URL, passage text, and similarity to the report. Retrieval failure is isolated:
the material-screening result is still produced with an explicit evidence error.
