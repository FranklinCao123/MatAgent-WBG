"""Minimal FastAPI and single-page interface for MatAgent-WBG."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from matagent.config import ConfigurationError
from matagent.runtime import RuntimeOptions, build_runtime_graph


INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class ScreenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=1000)
    use_rag: bool = True


class ScreenResponse(BaseModel):
    status: str
    report: str
    candidates: list[dict[str, Any]]
    evidence: dict[str, list[dict[str, Any]]]
    trace: list[dict[str, Any]]


app = FastAPI(
    title="MatAgent-WBG",
    description="Lightweight wide-bandgap material-screening Agent API.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    """Process health only; external services are checked during a real run."""

    return {"status": "ok", "service": "MatAgent-WBG"}


def _run_agent(request: ScreenRequest) -> ScreenResponse:
    graph = build_runtime_graph(RuntimeOptions(use_rag=request.use_rag))
    result = graph.invoke({"user_query": request.query})
    return ScreenResponse(
        status=result.get("status", "unknown"),
        report=result.get("final_report", ""),
        candidates=result.get("ranked_candidates", [])[:10],
        evidence=result.get("candidate_evidence", {}),
        trace=result.get("tool_history", []),
    )


@app.post("/api/screen", response_model=ScreenResponse)
async def screen(request: ScreenRequest) -> ScreenResponse:
    """Run blocking scientific APIs outside the ASGI event-loop thread."""

    try:
        return await run_in_threadpool(_run_agent, request)
    except (ConfigurationError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Agent execution failed ({type(error).__name__}).",
        ) from error
