from pathlib import Path
from loguru import logger
from second_brain_online.config import settings
from opik.evaluation.evaluator import evaluate
from opik.evaluation.metrics import Hallucination, AnswerRelevance, Moderation
from opik.evaluation import models

from second_brain_online.application.agents import get_agent, extract_tool_response
from second_brain_online import opik_utils
from .summary_density_heuristics import SummaryDensityHeuristic
from .summary_density_judge import SummaryDensityJudge


opik_utils.configure_opik()

judge_model = models.LiteLLMChatModel(
    model_name="gemini/gemini-2.5-flash-lite"
)

def evaluate_agent(
    prompts: list[str],
    retriever_config_path: Path,
) -> None:
    
    assert settings.COMET_API_KEY, "Please add comet api key in config file."
    
    logger.info("Starting Agent evaluation...")
    logger.info(f"Evaluating agent with {len(prompts)} prompts.")
    
    def evaluation_task(x: dict) -> dict:
        
        agent = get_agent(retriever_config_path=retriever_config_path)
        response = agent.run(task=x["input"])
        tool_response = extract_tool_response(agent)
        
        return {
            "input" : x["input"],
            "output" : response,
            "context" : tool_response
        }
        
    dataset_name = "second_brain_ai_agentic_evaluation_dataset"
    dataset = opik_utils.get_or_create_dataset(name=dataset_name, prompts=prompts)
    
    # experiment configs 
    
    agent = get_agent(retriever_config_path=retriever_config_path)
    
    experiment_config = {
        "dataset_name" : dataset_name,
        "model_id" : settings.GEMINI_MODEL_ID,
        "retriever_config_path" : retriever_config_path,
        "agent_config" : {
            "agent_name" : agent.agent_name,
            "max_steps" : agent.max_steps,
        }
    }
    
    scoring_metrics = [
        Hallucination(model=judge_model),
        AnswerRelevance(model=judge_model),
        Moderation(model=judge_model),
        SummaryDensityHeuristic(),
        SummaryDensityJudge(),
    ]   
    
    if dataset:
        logger.info("Evaluation details:")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Metrics: {[m.__class__.__name__ for m in scoring_metrics]}")
        evaluate(
            dataset=dataset,
            task=evaluation_task,
            experiment_config=experiment_config,
            scoring_metrics=scoring_metrics,
            task_threads=1,
        )
    else:
        logger.error("Can't run the evaluation as the dataset items are empty.")
