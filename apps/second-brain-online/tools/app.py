from pathlib import Path
import click

from smolagents import GradioUI
from loguru import logger

from second_brain_online.application.agents import get_agent
from second_brain_online import opik_utils

opik_utils.configure_opik()

@click.command()
@click.option(
    "--retriever-config-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the retriever config file",
)
@click.option(
    "--ui",
    is_flag=True,
    default=False,
    help="If want to open Gradio UI or continue on CLI."
)
@click.option(
    "--query",
    "-q",
    type=str,
    default="What is the feature/training/inference (FTI) pipelines architecture?",
    help="Query to run in CLI mode",
)
def main(retriever_config_path: Path, ui: bool, query: str) -> None:
    """Run the agent either in Gradio UI or CLI mode.

    Args:
        ui: If True, launches Gradio UI. If False, runs in CLI mode
        query: Query string to run in CLI mode
    """
    if retriever_config_path:
        # logger.info(f"retriever_config_path : {retriever_config_path}, type of config path : {type(retriever_config_path)}")
        agent = get_agent(retriever_config_path=Path(retriever_config_path))
    
    if ui:
        GradioUI(agent).launch()
    else:
        assert query, "Query is required to run agent in CLI mode."
        
        result = agent.run(query)
        
        print(result)
    
if __name__ == "__main__":
    main()