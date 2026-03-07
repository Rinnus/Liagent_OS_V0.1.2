"""Agent package — lazy imports to avoid heavy dependency chains at import time."""

# Deferred imports via PEP 562 __getattr__ to prevent side effects
# (brain.py triggers tool registration, engine loading, etc.)

_LAZY_IMPORTS = {
    "AgentBrain": ".brain",
    "ConversationMemory": ".memory",
    "LongTermMemory": ".memory",
    "TaskPlanner": ".planner",
    "PromptBuilder": ".prompt_builder",
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib
        mod = importlib.import_module(module_path, __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
