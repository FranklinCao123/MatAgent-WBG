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
