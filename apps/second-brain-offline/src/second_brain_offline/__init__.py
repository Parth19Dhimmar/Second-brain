from .config import settings

# only imports settings and not anything starts with "_" or not even any other variables
# which are not mentioned in list, even when wilcard imports done
__all__ = ["settings"] 
