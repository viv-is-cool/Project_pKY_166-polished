"""AI Harness backend — Stages 1-4 + Schedules, Evals & Settings.

Includes:
- Ollama model registry with robust context length detection
- Projects CRUD with JSON swarm graph persistence
- Asynchronous run execution & WebSocket live streaming
- Persistent Knowledge & Self-improving PostRunReflector
- Recurring Task Scheduler (crons & intervals)
- Automated Evals Harness (rubric scoring & metrics)
- Harness Configuration Settings
"""

import asyncio
import json
import logging
import os
import re
import inspect
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import ollama
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from models import (
    Project, engine, get_session, init_db,
    Run, RunStatus, Step, KnowledgeEntry,
    ScheduledTask, EvalSuite, EvalTestCase, EvalRun, EvalResult, AppSettings
)
from agent_core import (
    AgentLoop, StepEvent, PlanEvent, PlanUpdateEvent, ThoughtEvent,
    ActionEvent, ApprovalRequiredEvent, ApprovalResolvedEvent,
    ObservationEvent, CompactionEvent, NudgeEvent, FinalAnswerEvent, ErrorEvent
)
from knowledge import RunMemory, ProjectKnowledge, PostRunReflector
from tools import ToolRegistry
from graph_executor import GraphExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Background task tracking & active runs
# ---------------------------------------------------------------------------

_active_runs: dict[int, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Initialize default settings if not present
    with Session(engine) as session:
        settings = session.exec(select(AppSettings)).first()
        if not settings:
            session.add(AppSettings())
            session.commit()

    # Start recurring task scheduler
    try:
        from scheduler import scheduler
        scheduler.start()
    except Exception as e:
        logger.warning(f"Scheduler failed to start: {e}")

    yield

    try:
        scheduler.stop()
    except Exception:
        pass


app = FastAPI(title="AI Harness API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model registry — with robust context length detection
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    size_bytes: Optional[int] = None
    family: Optional[str] = None
    modified_at: Optional[str] = None


class ModelRegistryResponse(BaseModel):
    ollama_available: bool
    models: list[ModelInfo]
    error: Optional[str] = None


def _extract_context_length(show_response: Any, model_name: str = "") -> Optional[int]:
    """
    Extracts context length robustly across Ollama versions and python client types.
    Handles dict, pydantic objects, parameters, modelfile, and sensible model defaults.
    """
    show_dict: dict = {}
    if isinstance(show_response, dict):
        show_dict = show_response
    elif hasattr(show_response, "model_dump"):
        try:
            show_dict = show_response.model_dump()
        except Exception:
            show_dict = getattr(show_response, "__dict__", {})
    elif hasattr(show_response, "__dict__"):
        show_dict = show_response.__dict__

    # 1. Scan model_info for *.context_length
    model_info = show_dict.get("model_info") or getattr(show_response, "model_info", None)
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if str(key).endswith(".context_length"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

    # 2. Check parameters string
    parameters = show_dict.get("parameters") or getattr(show_response, "parameters", "") or ""
    if isinstance(parameters, str):
        match = re.search(r"num_ctx\s+(\d+)", parameters)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass

    # 3. Check modelfile
    modelfile = show_dict.get("modelfile") or getattr(show_response, "modelfile", "") or ""
    if isinstance(modelfile, str):
        match = re.search(r"PARAMETER\s+num_ctx\s+(\d+)", modelfile, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass

    # 4. Fallback based on model family name
    name_lower = model_name.lower()
    if "qwen2.5" in name_lower or "qwen2" in name_lower:
        return 32768
    elif "llama3.2" in name_lower or "llama3.1" in name_lower:
        return 131072
    elif "llama3" in name_lower:
        return 8192
    elif "mistral" in name_lower or "mixtral" in name_lower:
        return 32768
    elif "gemma2" in name_lower or "gemma" in name_lower:
        return 8192
    elif "phi3" in name_lower:
        return 128000
    elif "deepseek" in name_lower:
        return 65536
    elif "smollm" in name_lower:
        return 8192

    return 4096


@app.get("/api/models", response_model=ModelRegistryResponse)
def list_models() -> ModelRegistryResponse:
    # Read AppSettings to see if an external Ollama host is configured
    try:
        with Session(engine) as session:
            settings = session.exec(select(AppSettings)).first()
            if settings and settings.ollama_host:
                os.environ.setdefault("OLLAMA_HOST", settings.ollama_host)
    except Exception:
        pass

    try:
        raw = ollama.list()
    except Exception as exc:
        return ModelRegistryResponse(
            ollama_available=False,
            models=[],
            error=f"Couldn't reach Ollama ({exc}). Set the Ollama host in settings or OLLAMA_HOST env.",
        )

    models_list = []
    # Handle list response as dict or object
    raw_models = raw.get("models", []) if isinstance(raw, dict) else getattr(raw, "models", [])

    for entry in raw_models:
        if isinstance(entry, dict):
            name = entry.get("model") or entry.get("name", "unknown")
            details = entry.get("details", {}) or {}
            size_bytes = entry.get("size")
            modified_at = entry.get("modified_at")
        else:
            name = getattr(entry, "model", None) or getattr(entry, "name", "unknown")
            details = getattr(entry, "details", {}) or {}
            size_bytes = getattr(entry, "size", None)
            modified_at = getattr(entry, "modified_at", None)

        if not isinstance(details, dict) and hasattr(details, "__dict__"):
            details = details.__dict__

        context_length = None
        family = details.get("family") if isinstance(details, dict) else None
        parameter_size = details.get("parameter_size") if isinstance(details, dict) else None
        quantization = details.get("quantization_level") if isinstance(details, dict) else None

        try:
            show = ollama.show(name)
            context_length = _extract_context_length(show, name)
        except Exception:
            context_length = _extract_context_length({}, name)

        models_list.append(
            ModelInfo(
                name=name,
                parameter_size=parameter_size,
                quantization=quantization,
                context_length=context_length,
                size_bytes=size_bytes,
                family=family,
                modified_at=str(modified_at) if modified_at else None,
            )
        )

    return ModelRegistryResponse(ollama_available=True, models=models_list)


# ---------------------------------------------------------------------------
# Projects CRUD
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    workspace_path: str
    auto_approve: bool = False


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workspace_path: Optional[str] = None
    auto_approve: Optional[bool] = None
    graph: Optional[dict] = None


@app.get("/api/projects", response_model=list[Project])
def list_projects(session: Session = Depends(get_session)) -> list[Project]:
    return session.exec(select(Project)).all()


@app.post("/api/projects", response_model=Project)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> Project:
    project = Project(**payload.model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.get("/api/projects/{project_id}", response_model=Project)
def get_project(project_id: int, session: Session = Depends(get_session)) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.patch("/api/projects/{project_id}", response_model=Project)
def update_project(
    project_id: int, payload: ProjectUpdate, session: Session = Depends(get_session)
) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    data_dict = payload.model_dump(exclude_unset=True)
    for field, value in data_dict.items():
        setattr(project, field, value)
    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, session: Session = Depends(get_session)) -> dict:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Clean up dependent runs and steps
    runs = session.exec(select(Run).where(Run.project_id == project_id)).all()
    for r in runs:
        steps = session.exec(select(Step).where(Step.run_id == r.id)).all()
        for s in steps:
            session.delete(s)
        session.delete(r)

    # Clean up knowledge entries
    k_entries = session.exec(select(KnowledgeEntry).where(KnowledgeEntry.project_id == project_id)).all()
    for k in k_entries:
        session.delete(k)

    # Clean up scheduled tasks
    s_tasks = session.exec(select(ScheduledTask).where(ScheduledTask.project_id == project_id)).all()
    for st in s_tasks:
        session.delete(st)

    # Clean up eval suites
    e_suites = session.exec(select(EvalSuite).where(EvalSuite.project_id == project_id)).all()
    for es in e_suites:
        cases = session.exec(select(EvalTestCase).where(EvalTestCase.suite_id == es.id)).all()
        for c in cases:
            session.delete(c)
        e_runs = session.exec(select(EvalRun).where(EvalRun.suite_id == es.id)).all()
        for er in e_runs:
            results = session.exec(select(EvalResult).where(EvalResult.eval_run_id == er.id)).all()
            for res in results:
                session.delete(res)
            session.delete(er)
        session.delete(es)

    session.delete(project)
    session.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Runs execution & background tasks
# ---------------------------------------------------------------------------

class RunCreate(BaseModel):
    project_id: int
    model: str
    task: str
    max_steps: Optional[int] = 20
    auto_approve: Optional[bool] = False
    mode: Optional[str] = "swarm"  # "swarm" or "direct"


async def launch_run_async(
    project_id: int,
    model: str,
    task: str,
    max_steps: int = 20,
    auto_approve: bool = False,
    mode: str = "swarm",
) -> int:
    """Core launcher used by both API POST /api/runs and TaskScheduler."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise ValueError("Project not found")

        run = Run(
            project_id=project_id,
            model=model,
            task=task,
            max_steps=max_steps,
            status=RunStatus.running,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    _active_runs[run_id] = {
        "events": asyncio.Queue(),
        "history": [],
        "subscribers": set(),
        "task": None,
        "executor": None,
    }

    t = asyncio.create_task(
        background_run_task(
            run_id=run_id,
            project_id=project.id,
            workspace_path=project.workspace_path,
            model=model,
            task=task,
            max_steps=max_steps,
            auto_approve=auto_approve,
            mode=mode,
        )
    )
    _active_runs[run_id]["task"] = t
    return run_id


async def background_run_task(
    run_id: int,
    project_id: int,
    workspace_path: str,
    model: str,
    task: str,
    max_steps: int,
    auto_approve: bool,
    mode: str = "swarm",
):
    with Session(engine) as session:
        run = session.get(Run, run_id)
        project = session.get(Project, project_id)
        if not run or not project:
            return

        tool_registry = ToolRegistry(workspace=Path(workspace_path))
        run_memory = RunMemory(run_id)
        tool_registry.run_memory = run_memory
        project_knowledge = ProjectKnowledge(project_id)

        use_graph = (
            mode != "direct"
            and project.graph
            and isinstance(project.graph, dict)
            and bool(project.graph.get("nodes"))
        )

        if use_graph:
            executor = GraphExecutor(
                project=project,
                run=run,
                session=session,
                graph=project.graph,
                model=model,
                tool_registry=tool_registry,
                workspace=Path(workspace_path),
                max_steps=max_steps,
                auto_approve=auto_approve,
                run_memory=run_memory,
                project_knowledge=project_knowledge,
                db_session=session,
            )
        else:
            executor = AgentLoop(
                model=model,
                tool_registry=tool_registry,
                workspace=Path(workspace_path),
                max_steps=max_steps,
                auto_approve=auto_approve,
                run_memory=run_memory,
                project_knowledge=project_knowledge,
                node_id="single",
                db_session=session,
            )

        if run_id in _active_runs:
            _active_runs[run_id]["executor"] = executor
            queue = _active_runs[run_id]["events"]
            history = _active_runs[run_id]["history"]
        else:
            queue = asyncio.Queue()
            history = []

        all_events = []
        step_counter = 0

        try:
            async for event in executor.run(task):
                all_events.append(event)
                history.append(event)
                await queue.put(event)

                step = Step(
                    run_id=run_id,
                    node_id=getattr(event, "node_id", "single"),
                    event_type=event.type,
                    step_index=getattr(event, "step_index", step_counter),
                )

                if event.type == "thought":
                    step.thought = getattr(event, "content", None)
                    step.tokens_used = getattr(event, "tokens_used", 0)
                elif event.type == "action":
                    step.action = getattr(event, "tool_name", None)
                    step.action_input = getattr(event, "action_input", None)
                elif event.type == "observation":
                    step.observation = getattr(event, "content", None)
                    step.tokens_used = getattr(event, "tokens_used", 0)
                elif event.type == "approval_required":
                    step.action = getattr(event, "tool_name", None)
                    step.action_input = getattr(event, "action_input", None)
                    step.diff_preview = getattr(event, "preview", None)
                elif event.type == "final_answer":
                    step.observation = getattr(event, "content", None)
                    run.final_answer = getattr(event, "content", None)

                session.add(step)
                session.commit()
                step_counter += 1

                if run_id in _active_runs:
                    for sub in list(_active_runs[run_id].get("subscribers", set())):
                        try:
                            await sub.put(event)
                        except Exception:
                            pass

            run.status = RunStatus.done
            run.finished_at = datetime.utcnow()
            run.step_count = step_counter

            # Run post-run reflection
            reflector = PostRunReflector()
            db_steps = session.exec(select(Step).where(Step.run_id == run_id)).all()
            success = any(isinstance(e, FinalAnswerEvent) for e in all_events)
            await reflector.reflect(
                model=model,
                task=task,
                steps=db_steps,
                success=success,
                session=session,
                project_id=project_id,
                run_id=run_id,
            )
            run_memory.persist(session)

        except asyncio.CancelledError:
            run.status = RunStatus.cancelled
            run.finished_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error executing run #{run_id}: {e}")
            run.status = RunStatus.error
            run.error_message = str(e)
            run.finished_at = datetime.utcnow()
        finally:
            session.add(run)
            session.commit()
            try:
                await queue.put(None)
            except Exception:
                pass
            # Notify subscribers termination
            if run_id in _active_runs:
                for sub in list(_active_runs[run_id].get("subscribers", set())):
                    try:
                        await sub.put(None)
                    except Exception:
                        pass
            # Cleanup active runs to avoid leaks
            _active_runs.pop(run_id, None)


@app.post("/api/runs")
async def create_run(payload: RunCreate, session: Session = Depends(get_session)):
    project = session.get(Project, payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = await launch_run_async(
        project_id=project.id,
        model=payload.model,
        task=payload.task,
        max_steps=payload.max_steps or 20,
        auto_approve=payload.auto_approve or False,
        mode=payload.mode or "swarm",
    )
    return {"id": run_id, "status": "running"}


@app.get("/api/runs")
def list_runs(project_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(Run).where(Run.project_id == project_id).order_by(Run.started_at.desc())
    ).all()


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    steps = session.exec(select(Step).where(Step.run_id == run_id).order_by(Step.id)).all()
    return {"run": run, "steps": steps}


@app.delete("/api/runs/{run_id}")
def cancel_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_id in _active_runs:
        task = _active_runs[run_id].get("task")
        if task and not task.done():
            task.cancel()

    run.status = RunStatus.cancelled
    session.add(run)
    session.commit()
    # ensure cleanup
    _active_runs.pop(run_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/run/{run_id}")
async def run_websocket(websocket: WebSocket, run_id: int):
    await websocket.accept()

    # If run is currently active in memory
    if run_id in _active_runs:
        run_data = _active_runs[run_id]
        executor = run_data.get("executor")
        history = list(run_data.get("history", []))

        # 1. Send all events occurred so far
        for ev in history:
            try:
                await websocket.send_json(ev.to_dict() if hasattr(ev, "to_dict") else ev)
            except Exception:
                pass

        # 2. Subscribe to incoming events
        sub_queue = asyncio.Queue()
        run_data.setdefault("subscribers", set()).add(sub_queue)

        async def reader():
            try:
                while True:
                    event = await sub_queue.get()
                    if event is None:
                        break
                    try:
                        await websocket.send_json(
                            event.to_dict() if hasattr(event, "to_dict") else event
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        async def writer():
            try:
                while True:
                    data = await websocket.receive_json()
                    # Validate message
                    if not isinstance(data, dict) or "type" not in data:
                        try:
                            await websocket.send_json({"type": "error", "message": "malformed message"})
                        except Exception:
                            pass
                        continue

                    msg_type = data.get("type")
                    if msg_type == "approve":
                        if executor and hasattr(executor, "resolve_approval"):
                            try:
                                res = executor.resolve_approval("approved")
                                if inspect.isawaitable(res):
                                    await res
                            except Exception as e:
                                logger.exception("Failed to resolve approval: %s", e)
                    elif msg_type == "edit":
                        if executor and hasattr(executor, "resolve_approval"):
                            try:
                                res = executor.resolve_approval("edited", data.get("action_input"))
                                if inspect.isawaitable(res):
                                    await res
                            except Exception as e:
                                logger.exception("Failed to resolve edit: %s", e)
                    elif msg_type == "reject":
                        if executor and hasattr(executor, "resolve_approval"):
                            try:
                                res = executor.resolve_approval("rejected")
                                if inspect.isawaitable(res):
                                    await res
                            except Exception as e:
                                logger.exception("Failed to resolve reject: %s", e)
            except (WebSocketDisconnect, Exception):
                pass

        r_task = asyncio.create_task(reader())
        w_task = asyncio.create_task(writer())

        try:
            done, pending = await asyncio.wait([r_task, w_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
        finally:
            run_data.get("subscribers", set()).discard(sub_queue)
            # Send termination event
            try:
                with Session(engine) as session:
                    run_obj = session.get(Run, run_id)
                    status = run_obj.status.value if run_obj and hasattr(run_obj.status, 'value') else str(run_obj.status) if run_obj else "unknown"
                    await websocket.send_json({"type": "end", "status": status})
            except Exception:
                pass

    else:
        # Run already completed: replay DB steps
        with Session(engine) as session:
            steps = session.exec(select(Step).where(Step.run_id == run_id).order_by(Step.id)).all()
            for step in steps:
                event_obj = {
                    "type": step.event_type or "thought",
                    "node_id": step.node_id,
                    "step_index": step.step_index,
                }
                if step.event_type == "thought":
                    event_obj["content"] = step.thought or ""
                    event_obj["tokens_used"] = step.tokens_used
                elif step.event_type == "action":
                    event_obj["tool_name"] = step.action or ""
                    event_obj["action_input"] = step.action_input or {}
                elif step.event_type == "observation":
                    event_obj["content"] = step.observation or ""
                    event_obj["tokens_used"] = step.tokens_used
                elif step.event_type == "approval_required":
                    event_obj["tool_name"] = step.action or ""
                    event_obj["action_input"] = step.action_input or {}
                    event_obj["preview"] = step.diff_preview or ""
                elif step.event_type == "final_answer":
                    event_obj["content"] = step.observation or ""
                    event_obj["total_tokens"] = step.tokens_used

                try:
                    await websocket.send_json(event_obj)
                except Exception:
                    pass
            # send termination
            try:
                run_obj = session.get(Run, run_id)
                status = run_obj.status.value if run_obj and hasattr(run_obj.status, 'value') else str(run_obj.status) if run_obj else "unknown"
                await websocket.send_json({"type": "end", "status": status})
            except Exception:
                pass

    if websocket.client_state.name != "DISCONNECTED":
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/knowledge")
def list_knowledge(project_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(KnowledgeEntry).where(KnowledgeEntry.project_id == project_id)
    ).all()


@app.delete("/api/projects/{project_id}/knowledge/{entry_id}")
def delete_knowledge(project_id: int, entry_id: int, session: Session = Depends(get_session)):
    entry = session.get(KnowledgeEntry, entry_id)
    if not entry or entry.project_id != project_id:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    session.delete(entry)
    session.commit()
    return {"ok": True}


@app.post("/api/projects/{project_id}/knowledge/reset")
def reset_knowledge(project_id: int, session: Session = Depends(get_session)):
    entries = session.exec(
        select(KnowledgeEntry).where(KnowledgeEntry.project_id == project_id)
    ).all()
    for entry in entries:
        session.delete(entry)
    session.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Scheduled & Recurring Tasks endpoints
# ---------------------------------------------------------------------------

class ScheduledTaskCreate(BaseModel):
    project_id: int
    name: str
    task_prompt: str
    cron_or_interval: Optional[str] = "every_1h"
    interval_minutes: Optional[int] = 60
    is_active: Optional[bool] = True


class ScheduledTaskUpdate(BaseModel):
    name: Optional[str] = None
    task_prompt: Optional[str] = None
    cron_or_interval: Optional[str] = None
    interval_minutes: Optional[int] = None
    is_active: Optional[bool] = None


@app.get("/api/schedules")
def list_schedules(project_id: Optional[int] = None, session: Session = Depends(get_session)):
    query = select(ScheduledTask)
    if project_id is not None:
        query = query.where(ScheduledTask.project_id == project_id)
    return session.exec(query.order_by(ScheduledTask.created_at.desc())).all()


@app.post("/api/schedules")
def create_schedule(payload: ScheduledTaskCreate, session: Session = Depends(get_session)):
    task = ScheduledTask(**payload.model_dump())
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@app.patch("/api/schedules/{schedule_id}")
def update_schedule(
    schedule_id: int, payload: ScheduledTaskUpdate, session: Session = Depends(get_session)
):
    task = session.get(ScheduledTask, schedule_id)
    if not task:
        raise HTTPException(status_code=404, detail="Schedule not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@app.delete("/api/schedules/{schedule_id}")
def delete_schedule(schedule_id: int, session: Session = Depends(get_session)):
    task = session.get(ScheduledTask, schedule_id)
    if not task:
        raise HTTPException(status_code=404, detail="Schedule not found")
    session.delete(task)
    session.commit()
    return {"ok": True}


@app.post("/api/schedules/{schedule_id}/run-now")
async def run_schedule_now(schedule_id: int, session: Session = Depends(get_session)):
    st = session.get(ScheduledTask, schedule_id)
    if not st:
        raise HTTPException(status_code=404, detail="Schedule not found")
    project = session.get(Project, st.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = await launch_run_async(
        project_id=project.id,
        model=project.auto_approve and session.exec(select(AppSettings)).first().default_model or "qwen2.5:1.5b",
        task=st.task_prompt,
        max_steps=25,
        auto_approve=project.auto_approve,
    )
    st.last_run_at = datetime.utcnow()
    session.add(st)
    session.commit()
    return {"run_id": run_id, "status": "running"}


# ---------------------------------------------------------------------------
# Evals endpoints
# ---------------------------------------------------------------------------

class EvalSuiteCreate(BaseModel):
    project_id: int
    name: str
    description: Optional[str] = ""


class EvalTestCaseCreate(BaseModel):
    suite_id: int
    name: str
    input_prompt: str
    expected_output: Optional[str] = ""
    eval_type: Optional[str] = "rubric"
    rubric: Optional[str] = "Rate the accuracy and completeness from 1 to 10."
    pass_threshold: Optional[float] = 7.0


@app.get("/api/evals/suites")
def list_eval_suites(project_id: Optional[int] = None, session: Session = Depends(get_session)):
    query = select(EvalSuite)
    if project_id is not None:
        query = query.where(EvalSuite.project_id == project_id)
    return session.exec(query.order_by(EvalSuite.created_at.desc())).all()


@app.post("/api/evals/suites")
def create_eval_suite(payload: EvalSuiteCreate, session: Session = Depends(get_session)):
    suite = EvalSuite(**payload.model_dump())
    session.add(suite)
    session.commit()
    session.refresh(suite)
    return suite


@app.get("/api/evals/suites/{suite_id}")
def get_eval_suite(suite_id: int, session: Session = Depends(get_session)):
    suite = session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Eval suite not found")
    tests = session.exec(select(EvalTestCase).where(EvalTestCase.suite_id == suite_id)).all()
    runs = session.exec(
        select(EvalRun).where(EvalRun.suite_id == suite_id).order_by(EvalRun.started_at.desc())
    ).all()
    return {"suite": suite, "tests": tests, "runs": runs}


@app.delete("/api/evals/suites/{suite_id}")
def delete_eval_suite(suite_id: int, session: Session = Depends(get_session)):
    suite = session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")
    session.delete(suite)
    session.commit()
    return {"ok": True}


@app.post("/api/evals/suites/{suite_id}/tests")
def add_eval_test(
    suite_id: int, payload: EvalTestCaseCreate, session: Session = Depends(get_session)
):
    test = EvalTestCase(**payload.model_dump())
    test.suite_id = suite_id
    session.add(test)
    session.commit()
    session.refresh(test)
    return test


@app.delete("/api/evals/tests/{test_id}")
def delete_eval_test(test_id: int, session: Session = Depends(get_session)):
    test = session.get(EvalTestCase, test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    session.delete(test)
    session.commit()
    return {"ok": True}


@app.post("/api/evals/suites/{suite_id}/run")
async def run_eval_suite_endpoint(suite_id: int):
    from evals import execute_eval_suite
    eval_run_id = await execute_eval_suite(suite_id)
    return {"eval_run_id": eval_run_id, "status": "running"}


@app.get("/api/evals/runs/{run_id}")
def get_eval_run_results(run_id: int, session: Session = Depends(get_session)):
    eval_run = session.get(EvalRun, run_id)
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    results = session.exec(select(EvalResult).where(EvalResult.eval_run_id == run_id)).all()
    return {"eval_run": eval_run, "results": results}


# ---------------------------------------------------------------------------
# App Settings endpoints
# ---------------------------------------------------------------------------

class SettingsUpdate(BaseModel):
    ollama_host: Optional[str] = None
    default_model: Optional[str] = None
    default_timeout_seconds: Optional[int] = None
    output_size_cap_kb: Optional[int] = None
    auto_approve_default: Optional[bool] = None


@app.get("/api/settings")
def get_settings(session: Session = Depends(get_session)):
    settings = session.exec(select(AppSettings)).first()
    if not settings:
        settings = AppSettings()
        session.add(settings)
        session.commit()
        session.refresh(settings)
    return settings


@app.post("/api/settings")
def update_settings(payload: SettingsUpdate, session: Session = Depends(get_session)):
    settings = session.exec(select(AppSettings)).first()
    if not settings:
        settings = AppSettings()
        session.add(settings)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)
    settings.updated_at = datetime.utcnow()
    session.add(settings)
    session.commit()
    session.refresh(settings)
    return settings


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
