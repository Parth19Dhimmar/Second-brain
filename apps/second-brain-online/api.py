import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import litellm
from contextlib import asynccontextmanager

from second_brain_online.application.agents import get_agent
from second_brain_online import opik_utils
from response import error_response, success_response

from exceptions import (
    LLMGenerationError, LLMQuotaError, AgentError
)

# we have set it static
RETRIEVER_CONFIG_PATH = Path("configs/compute_rag_vector_index_huggingface_contextual_none.yaml")

state = {"agent": None}

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


### GLOBAL EXCEPTION HANDLERS

@app.exception_handler(LLMQuotaError)
async def quota_error_handler(request, exc: LLMQuotaError):
    return error_response(
        message=str(exc),
        status_code=429,
        error_code="LLM_QUOTA_EXCEEDED"
    )


@app.exception_handler(LLMGenerationError)
async def llm_error_handler(request, exc: LLMGenerationError):
    return error_response(
        message=str(exc),
        status_code=500,
        error_code="LLM_GENERATION_FAILED"
    )


@app.exception_handler(AgentError)
async def agent_error_handler(request, exc: AgentError):
    return error_response(
        message=str(exc),
        status_code=500,
        error_code="AGENT_ERROR"
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
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
    
### RAG API

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the Second Brain agent
    """
    agent = state.get("agent")
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    logger.info(f"Processing query: {request.query}")
    result = agent.run(request.query)
    
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
