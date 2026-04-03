### Global Error Handling ###

class AgentError(Exception):
    pass

class LLMGenerationError(AgentError):
    pass

class LLMQuotaError(AgentError):
    pass