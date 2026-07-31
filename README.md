# MatAgent-WBG

Lightweight local prototype for an LLM-based scientific agent that screens
wide-bandgap semiconductor materials.

## Current scope

The current version validates a deterministic LangGraph workflow:

1. Parse a natural-language request.
2. Enrich the parsed request with an auditable demonstration domain policy.
3. Build a request-aware ranking plan.
4. Search a tiny local mock dataset.
5. Route conditionally based on tool results.
6. Rank candidates using the generated plan when results exist.
7. Generate a Markdown report, including inferred requirements and errors.

No LLM API, GPU, scientific model, vector database, or server is required.
All material property values are illustrative mock data and are not suitable
for scientific or engineering decisions.

The requirement-parsing node now supports two modes:

- `offline`: deterministic local parser; no network or API cost.
- `deepseek`: DeepSeek JSON Output followed by Pydantic validation.

## Project structure

```text
matagent/
├── schemas.py               # Strict Pydantic data contracts
├── config.py                # Environment-based LLM configuration
├── llm/                     # Offline and DeepSeek requirement parsers
├── state.py                 # Shared LangGraph state
├── nodes.py                 # Workflow node logic
├── graph.py                 # Graph construction and conditional routing
├── cli.py                   # Command-line interface
└── tools/
    └── material_search.py   # Replaceable material-search tool

tests/
└── test_workflow.py         # Standard-library automated tests
```

`prototype.py` remains as a backward-compatible entry point.

## Run

Activate the existing Conda environment:

```powershell
conda activate agentenv
```

Run with the default query:

```powershell
python prototype.py
```

The default mode is offline. To use DeepSeek, copy `.env.example` to `.env`,
insert a newly created API key locally, and never commit `.env`:

```powershell
Copy-Item .env.example .env
```

Then run:

```powershell
python -m matagent.cli --mode deepseek `
  "寻找带隙大于 3 eV、适合高温功率器件并优先考虑高热导率的材料"
```

Configuration variables:

```text
MATAGENT_LLM_MODE=deepseek
MATAGENT_LLM_API_KEY=<local secret>
MATAGENT_LLM_MODEL=deepseek-v4-flash
MATAGENT_LLM_BASE_URL=https://api.deepseek.com
```

Run with a custom query and display the execution trace:

```powershell
python prototype.py "寻找适合高温功率电子器件的宽禁带半导体材料" --show-trace
```

The formal module entry point is equivalent:

```powershell
python -m matagent.cli "寻找适合高温功率电子器件的宽禁带半导体材料" --show-trace
```

## Test

The tests use Python's standard library, so no additional package is needed:

```powershell
python -m unittest discover -s tests -v
```

The graph now contains a conditional edge:

```text
parse_requirements
├── success → plan_screening → search_materials
│                              ├── candidates → rank_candidates ─┐
│                              └── empty/error ──────────────────┤
└── error ──────────────────────────────────────────────────────┤
                                                               ↓
                                                        generate_report
```

Screening requirements are represented by a strict Pydantic model. Invalid
values and unexpected fields are rejected before they can reach a scientific
tool. The same schema can later be used as the contract for structured LLM
output.

`plan_screening` is currently a transparent demonstration policy rather than
an LLM call or a validated scientific ranking method. It records any domain
requirements inferred from the application and produces normalized weights
that sum to 1.0. The ranking node consumes this plan instead of fixed weights.
