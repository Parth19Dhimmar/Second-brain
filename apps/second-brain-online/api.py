import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import litellm
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from second_brain_online.application.agents import get_agent
from second_brain_online.utilities import opik_utils
from second_brain_online.response import error_response, success_response
from second_brain_online.config import settings
from second_brain_online.exceptions import (
    LLMGenerationError, LLMQuotaError, AgentError
)

# we have set it static
RETRIEVER_CONFIG_PATH = Path("configs/compute_rag_vector_index_huggingface_contextual_none.yaml")

state = {"agent": None}

# ── Rate limiter ───────────────────────────────────────────────────────────────
# Uses your existing Redis so limits are consistent even if you run
# multiple uvicorn workers. Falls back gracefully if Redis is down.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing agent...")
    state["agent"] = get_agent(retriever_config_path=RETRIEVER_CONFIG_PATH)
    logger.info("Agent initialized successfully")
    logger.info("Configuring Opik...")
    opik_utils.configure_opik()
    logger.info("Opik Configuration Successfull.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Second Brain API", lifespan=lifespan)

# Attach limiter to app state so slowapi can find it
app.state.limiter = limiter


# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom 429 response with Retry-After header."""
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address(request)}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down and try again shortly.",
            "retry_after_seconds": 60,
        },
        headers={"Retry-After": "60"},
    )

@app.exception_handler(LLMQuotaError)
async def quota_error_handler(request: Request, exc: LLMQuotaError):
    return error_response(
        message=str(exc),
        status_code=429,
        error_code="LLM_QUOTA_EXCEEDED"
    )


@app.exception_handler(LLMGenerationError)
async def llm_error_handler(request: Request, exc: LLMGenerationError):
    return error_response(
        message=str(exc),
        status_code=500,
        error_code="LLM_GENERATION_FAILED"
    )


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    return error_response(
        message=str(exc),
        status_code=500,
        error_code="AGENT_ERROR"
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return error_response(
        message="Something went wrong. Please try again later.",
        status_code=500,
        error_code="SERVER_ERROR"
    )

### PYDANTIC REQ-RES MODELS

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
@limiter.limit("1/minute")
async def query_agent(request: Request, body: QueryRequest):
    """
    Query the Second Brain agent.
    Rate limited to 10 requests per minute per IP.
    Cache hits (Redis/MongoDB) do not count against LLM quota but still
    count against rate limit — protects the server regardless.
    """
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    logger.info(f"Processing query: {body.query}")
    result = agent.run(body.query)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {"answer": result}
    
    return success_response(
        data=result,
        message="Query processed successfully"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_loaded": state.get("agent") is not None}
