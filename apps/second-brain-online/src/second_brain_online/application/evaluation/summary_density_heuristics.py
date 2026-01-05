from typing import Any
from opik.evaluation.metrics import base_metric, score_result


class SummaryDensityHeuristic(base_metric.BaseMetric):
    def __init__(
        self,
        name: str = "summary_density_heuristic",
        min_length: int = 512,
        max_length: int = 1024
    ) -> None:
        self.name = name
        self.min_length = min_length
        self.max_length = max_length
        
    def score(self, input: str, output: str, **ignored_kwargs: Any) -> score_result.ScoreResult:
        """_summary_

        Args:
            input (str): _description_
            output (str): _description_

        Returns:
            score_result.ScoreResult: _description_
        """
        
        length_score = self._compute_length_score(output)
        
        reason = f"output_length : {len(output)}."
        
        if length_score == 1.0:
            reason += f"The output length is in ideal range."
        if 1.0 > length_score > 0.5:
            reason += f"The output length is slightly outside ideal range."
        else:
            reason += f"The output length is significantly outside ideal range." 
        
        return score_result(
            name=self.name,
            value=length_score,
            reason=reason,
        )
        
    
    def _compute_length_score(self, text: str) -> float:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            float: _description_
        """
        
        text_length = len(str)
        
        if self.min_length < text_length < self.max_length:
            return 1.0
        
        if text_length < self.min_length:
            deviation = (self.min_length - text_length) / self.min_length
            
        else:
            deviation = (text_length - self.max_length) / self.max_length
            
        score = max(0.0, 1.0 - deviation)
        
        return score