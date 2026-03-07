"""Interest / Watch REST endpoints — /api/interests/*."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import shared_state as _shared
from .shared_state import (
    _allow_rate,
    _authorized_http,
    _client_id_from_request,
    _http_rate,
    _log,
)

router = APIRouter()


@router.post("/api/interests")
async def create_interest_endpoint(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    query = str(data.get("query", "")).strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    from ..agent.interest import (
        InterestStore, create_interest_from_query, build_coverage_summary,
    )

    store = InterestStore()
    try:
        interest = await create_interest_from_query(query, _shared._engine, store)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        _log.error("interest_api", e, action="create_interest")
        return JSONResponse({"error": f"internal error: {e}"}, status_code=500)

    # Notify signal poller to start polling this interest's factors
    if _shared._signal_poller is not None:
        try:
            _shared._signal_poller.add_interest(interest)
        except Exception as e:
            _log.error("interest_api", e, action="poller_add_interest")

    summary = build_coverage_summary(interest)
    return JSONResponse({"interest": interest, "coverage": summary})


@router.get("/api/interests")
async def list_interests_endpoint(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    return JSONResponse({"interests": store.list_interests()})


@router.get("/api/interests/backlog")
async def blind_backlog_endpoint(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    return JSONResponse({"backlog": store.get_blind_backlog()})


@router.get("/api/interests/{interest_id}")
async def get_interest_endpoint(interest_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore, build_coverage_summary
    store = InterestStore()
    interest = store.get_interest(interest_id)
    if interest is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({
        "interest": interest,
        "coverage": build_coverage_summary(interest),
    })


@router.post("/api/interests/{interest_id}/thread")
async def set_thread_endpoint(interest_id: str, data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    thread_id = str(data.get("discord_thread_id", "")).strip()
    if not thread_id:
        return JSONResponse({"error": "discord_thread_id is required"}, status_code=400)
    result = store.update_interest(interest_id, discord_thread_id=thread_id)
    if result is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"status": "ok"})


@router.post("/api/interests/{interest_id}/pause")
async def pause_interest_endpoint(interest_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    if not store.pause_interest(interest_id):
        return JSONResponse({"error": "not found or not active"}, status_code=404)
    if _shared._signal_poller is not None:
        _shared._signal_poller.remove_interest(interest_id)
    return JSONResponse({"status": "paused"})


@router.post("/api/interests/{interest_id}/resume")
async def resume_interest_endpoint(interest_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    if not store.resume_interest(interest_id):
        return JSONResponse({"error": "not found or not paused"}, status_code=404)
    # Re-add interest factors to poller
    if _shared._signal_poller is not None:
        interest = store.get_interest(interest_id)
        if interest:
            _shared._signal_poller.add_interest(interest)
    return JSONResponse({"status": "resumed"})


@router.delete("/api/interests/{interest_id}")
async def archive_interest_endpoint(interest_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from ..agent.interest import InterestStore
    store = InterestStore()
    if not store.archive_interest(interest_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    if _shared._signal_poller is not None:
        _shared._signal_poller.remove_interest(interest_id)
    return JSONResponse({"status": "archived"})
