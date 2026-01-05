from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
from contextlib import asynccontextmanager

from second_brain_online.application.agents import get_agent
from second_brain_online import opik_utils

opik_utils.configure_opik()

# we have set it static
RETRIEVER_CONFIG_PATH = Path("configs/compute_rag_vector_index_huggingface_contextual_none.yaml")

state = {"agent": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing agent...")
    state["agent"] = get_agent(retriever_config_path=RETRIEVER_CONFIG_PATH)
    logger.info("Agent initialized successfully")
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

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the Second Brain agent
    """
    try:
        agent = state.get("agent")
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        logger.info(f"Processing query: {request.query}")
        result = agent.run(request.query)
        return str(result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_loaded": state.get("agent") is not None}
