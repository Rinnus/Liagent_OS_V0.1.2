"""Autonomous task tools — create, delete, and list scheduled/triggered tasks."""

from . import ToolCapability, tool

# Runtime references — set by main.py when the task system is initialized.
_task_store = None
_trigger_manager = None
_engine = None


def configure(store, trigger_manager, engine):
    """Called by main.py to inject runtime references."""
    global _task_store, _trigger_manager, _engine
    _task_store = store
    _trigger_manager = trigger_manager
    _engine = engine


# ─── create_task ──────────────────────────────────────────────────────────────

def _validate_create_task(args: dict) -> tuple[bool, str]:
    desc = str(args.get("description", "")).strip()
    if not desc:
        return False, "description is required"
    if len(desc) > 500:
        return False, "description too long (max 500 chars)"
    return True, "ok"


@tool(
    name="create_task",
    description=(
        "Create delayed or scheduled tasks. Use this tool when the request contains time intent. "
        "Keywords: 'in X minutes', 'in X hours', 'later', 'every day', 'every hour', "
        "'schedule', 'remind me'. "
        "Examples: 'remind me to drink water in 3 minutes' -> one-shot delayed task; "
        "'check GOOG in 5 minutes and send me the result' -> one-shot delayed query task; "
        "'search AI news at 8 every day' -> recurring scheduled task. "
        "Important: when user says 'do X in N minutes', do not execute now; create a delayed task."
    ),
    risk_level="medium",
    capability=ToolCapability(
        data_classification="internal",
        network_access=False,
        filesystem_access=False,
        max_output_chars=600,
        idempotent=False,
    ),
    validator=_validate_create_task,
    parameters={
        "properties": {
            "description": {
                "type": "string",
                "description": "Natural-language task description containing schedule and action details",
            },
        },
        "required": ["description"],
    },
)
async def create_task(description: str, **kwargs) -> str:
    """Parse a natural language task description and create an autonomous task."""
    if _task_store is None or _engine is None:
        return "[Error] Task system is not initialized. Run in web mode."

    from ..agent.task_parser import parse_task_description

    parsed = await parse_task_description(_engine, description)
    if parsed is None:
        return f"[Error] Unable to parse task description: {description}"

    task = _task_store.create_task(
        name=parsed["name"],
        trigger_type=parsed["trigger_type"],
        trigger_config=parsed.get("trigger_config", {}),
        prompt_template=parsed["prompt_template"],
    )

    # Register trigger
    if _trigger_manager:
        if parsed["trigger_type"] == "cron":
            schedule = parsed.get("trigger_config", {}).get("schedule", "")
            if schedule:
                await _trigger_manager.register_cron(task["id"], schedule)
        elif parsed["trigger_type"] == "once":
            delay = parsed.get("trigger_config", {}).get("delay_seconds", 60)
            await _trigger_manager.register_once(task["id"], delay)

    # Build human-readable response
    trigger_info = ""
    if parsed["trigger_type"] == "cron":
        schedule = parsed.get("trigger_config", {}).get("schedule", "")
        trigger_info = f", recurring schedule: {schedule}"
    elif parsed["trigger_type"] == "once":
        delay = parsed.get("trigger_config", {}).get("delay_seconds", 0)
        if delay >= 3600:
            trigger_info = f", will run in {delay // 3600} hour(s)"
        elif delay >= 60:
            trigger_info = f", will run in {delay // 60} minute(s)"
        else:
            trigger_info = f", will run in {delay} second(s)"
    return (
        f"[Task created] ID: {task['id']}\n"
        f"Name: {parsed['name']}\n"
        f"Type: {parsed['trigger_type']}{trigger_info}\n"
        f"Execution prompt: {parsed['prompt_template'][:100]}"
    )


# ─── delete_task ──────────────────────────────────────────────────────────────

def _validate_delete_task(args: dict) -> tuple[bool, str]:
    task_id = str(args.get("task_id", "")).strip()
    if not task_id:
        return False, "task_id is required"
    return True, "ok"


@tool(
    name="delete_task",
    description=(
        "Cancel/delete one scheduled task. Use when user asks to cancel a task or stop reminders. "
        "Requires task ID. If unknown, call `list_tasks` first."
    ),
    risk_level="low",
    capability=ToolCapability(
        data_classification="internal",
        network_access=False,
        filesystem_access=False,
        max_output_chars=300,
        idempotent=False,
        failure_modes=("not_found",),
    ),
    validator=_validate_delete_task,
    parameters={
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task ID to cancel",
            },
        },
        "required": ["task_id"],
    },
)
async def delete_task(task_id: str, **kwargs) -> str:
    """Delete an autonomous task by ID."""
    if _task_store is None:
        return "[Error] Task system is not initialized."

    task = _task_store.get_task(task_id)
    if not task:
        return f"[Error] Task ID not found: {task_id}"

    name = task.get("name", "")
    _task_store.delete_task(task_id)
    if _trigger_manager:
        await _trigger_manager.unregister(task_id)

    return f"[Cancelled] Task '{name}' (ID: {task_id}) was deleted."


# ─── delete_all_tasks ────────────────────────────────────────────────────────

@tool(
    name="delete_all_tasks",
    description=(
        "Bulk-cancel/delete all scheduled tasks (excluding filewatch tasks). "
        "Optional `name_filter` deletes only tasks whose names contain the filter string."
    ),
    risk_level="low",
    capability=ToolCapability(
        data_classification="internal",
        network_access=False,
        filesystem_access=False,
        max_output_chars=600,
        idempotent=False,
    ),
    parameters={
        "properties": {
            "name_filter": {
                "type": "string",
                "description": "Optional: only delete tasks whose names contain this text. Empty means all non-filewatch tasks.",
            },
        },
        "required": [],
    },
)
async def delete_all_tasks(name_filter: str = "", **kwargs) -> str:
    """Delete all user-created tasks (skip filewatch presets)."""
    if _task_store is None:
        return "[Error] Task system is not initialized."

    tasks = _task_store.list_tasks()
    deleted = []
    for t in tasks:
        # Skip built-in filewatch presets
        if t.get("trigger_type") == "filewatch":
            continue
        # Apply optional name filter
        if name_filter and name_filter not in t.get("name", ""):
            continue
        name = t.get("name", "")
        _task_store.delete_task(t["id"])
        if _trigger_manager:
            await _trigger_manager.unregister(t["id"])
        deleted.append(f"  - {name} ({t['id'][:8]}...)")

    if not deleted:
        return "No deletable tasks found."
    return f"[Bulk cancelled] Deleted {len(deleted)} task(s):\n" + "\n".join(deleted)


# ─── list_tasks ───────────────────────────────────────────────────────────────

@tool(
    name="list_tasks",
    description=(
        "List all scheduled tasks. Use when user asks what tasks exist or needs task IDs for cancellation."
    ),
    risk_level="low",
    capability=ToolCapability(
        data_classification="internal",
        network_access=False,
        filesystem_access=False,
        max_output_chars=1200,
    ),
    parameters={"properties": {}},
)
async def list_tasks(**kwargs) -> str:
    """List all active autonomous tasks."""
    if _task_store is None:
        return "[Error] Task system is not initialized."

    tasks = _task_store.list_tasks()
    if not tasks:
        return "There are currently no scheduled tasks."

    lines = [f"{len(tasks)} task(s):\n"]
    for t in tasks:
        status_icon = {"active": "▶", "paused": "⏸", "done": "✓"}.get(t.get("status", ""), "?")
        trigger_type = t.get("trigger_type", "")
        config = t.get("trigger_config", "{}")
        if isinstance(config, str):
            import json
            try:
                config = json.loads(config)
            except Exception:
                config = {}
        detail = ""
        if trigger_type == "cron":
            detail = f" ({config.get('schedule', '')})"
        elif trigger_type == "once":
            detail = f" ({config.get('delay_seconds', 0)}s delay)"
        elif trigger_type == "filewatch":
            detail = f" ({config.get('watch_dir', '').split('/')[-1]})"

        lines.append(
            f"{status_icon} [{t['id']}] {t.get('name', '')} - {trigger_type}{detail} - {t.get('status', '')}"
        )
    return "\n".join(lines)
