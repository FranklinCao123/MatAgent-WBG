# MatAgent-WBG

Lightweight local prototype for an LLM-based scientific agent that screens
wide-bandgap semiconductor materials.

## Current scope

The current version validates a deterministic LangGraph workflow:

1. Parse a natural-language request.
2. Search a tiny local mock dataset.
3. Rank candidates with a transparent weighted rule.
4. Generate a Markdown report.

No LLM API, GPU, scientific model, vector database, or server is required.
All material property values are illustrative mock data and are not suitable
for scientific or engineering decisions.

## Project structure

```text
matagent/
├── state.py                 # Shared LangGraph state
├── nodes.py                 # Workflow node logic
├── graph.py                 # Graph construction
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
