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
