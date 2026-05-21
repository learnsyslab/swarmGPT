"""FastAPI server for the SwarmGPT browser interface."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import drone_models
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from swarm_gpt.core import AppBackend
from swarm_gpt.utils import generate_default_colors
from swarm_gpt.utils.llm_providers import (
    DEFAULT_OPENAI_MODEL_CHOICES,
    PROVIDER_LABEL_OLLAMA,
    PROVIDER_LABEL_OPENAI,
    LLMProvider,
    default_openai_model,
    ollama_installed_model_names,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MUSIC_DIR = ROOT / "music"
WEB_DIST_DIR = ROOT / "web" / "dist"
SWARM_BACKGROUND = ROOT / "swarm_gpt" / "ui" / "swarm.png"
DRONE_ASSET_DIR = Path(drone_models.__file__).resolve().parent / "data" / "assets"

JobStatus = Literal["queued", "thinking", "filtering", "ready", "deploying", "failed"]


@dataclass(frozen=True)
class ApiConfig:
    """Configuration used to create per-job backends."""

    music_dir: Path = MUSIC_DIR
    preset_dir: Path | None = None
    strict_processing: bool = True
    strict_drone_match: bool = True
    model_id: str = "gpt-4o"
    llm_provider: LLMProvider = "openai"
    use_motion_primitives: bool = True


class JobRequest(BaseModel):
    """Payload for starting a choreography job."""

    selection: str = Field(min_length=1)
    provider: LLMProvider = "openai"
    model_id: str = Field(default_factory=default_openai_model, min_length=1, alias="modelId")


class RefineRequest(BaseModel):
    """Payload for refining an existing choreography."""

    message: str = Field(min_length=1)
    provider: LLMProvider | None = None
    model_id: str | None = Field(default=None, alias="modelId")


class Job:
    """Mutable state for one browser choreography job."""

    def __init__(self, job_id: str, backend: AppBackend):
        """Create a job container for one backend instance."""
        self.id = job_id
        self.backend = backend
        self.status: JobStatus = "queued"
        self.events: list[dict[str, Any]] = []
        self.playback: dict[str, Any] | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self._event_id = 0

    @property
    def is_running(self) -> bool:
        """Return whether this job currently owns a live worker thread."""
        return self.thread is not None and self.thread.is_alive()

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append a structured event to the job log."""
        self._event_id += 1
        self.events.append(
            {
                "id": self._event_id,
                "type": event_type,
                "createdAt": datetime.now(UTC).isoformat(),
                "payload": payload or {},
            }
        )


class JobStore:
    """Thread-safe in-memory job registry for the local single-user app."""

    def __init__(self):
        """Initialize an empty job registry."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, backend: AppBackend) -> Job:
        """Register a new job for a backend instance."""
        with self._lock:
            job = Job(uuid.uuid4().hex, backend)
            self._jobs[job.id] = job
            job.emit("queued", {"jobId": job.id})
            return job

    def get(self, job_id: str) -> Job:
        """Return a registered job or raise ``KeyError``."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def snapshot(self, job_id: str) -> dict[str, Any]:
        """Return a compact status snapshot for a job."""
        with self._lock:
            job = self.get(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "error": job.error,
                "ready": job.playback is not None,
                "events": len(job.events),
            }

    def emit(
        self,
        job: Job,
        event_type: str,
        payload: dict[str, Any] | None = None,
        status: JobStatus | None = None,
    ) -> None:
        """Append an event and optionally update the job status."""
        with self._lock:
            if status is not None:
                job.status = status
            job.emit(event_type, payload)

    def fail(self, job: Job, exc: BaseException) -> None:
        """Mark a job as failed and append a failure event."""
        logger.exception("Job %s failed", job.id)
        with self._lock:
            job.status = "failed"
            job.error = str(exc)
            job.emit("failed", {"message": str(exc)})

    def events_after(self, job_id: str, after_id: int) -> list[dict[str, Any]]:
        """Return all events with ids greater than ``after_id``."""
        with self._lock:
            job = self.get(job_id)
            return [event for event in job.events if event["id"] > after_id]


def _backend_from_config(config: ApiConfig, provider: LLMProvider, model_id: str) -> AppBackend:
    return AppBackend(
        music_dir=config.music_dir,
        preset_dir=config.preset_dir,
        strict_processing=config.strict_processing,
        strict_drone_match=config.strict_drone_match,
        model_id=model_id,
        llm_provider=provider,
        use_motion_primitives=config.use_motion_primitives,
    )


def _run_simulation_with_events(backend: AppBackend, store: JobStore, job: Job) -> dict[str, Any]:
    gen = backend.simulate()
    while True:
        try:
            key, data, total = next(gen)
        except StopIteration as exc:
            if exc.value is None:
                raise RuntimeError("Simulation finished without returning sim data")
            return exc.value
        if key == "progress":
            denominator = max(float(total), 1.0)
            store.emit(
                job,
                "safety_progress",
                {
                    "current": int(data),
                    "total": int(total),
                    "percent": min(1.0, max(0.0, float(data) / denominator)),
                },
            )


def _audio_url(song: str) -> str:
    return f"/api/media/music/{quote(song, safe='')}"


def _preset_item(backend: AppBackend, preset: str) -> dict[str, Any]:
    metadata = backend.preset_metadata(preset)
    song = str(metadata["song"])
    return {
        "id": preset,
        "label": song,
        "kind": "preset",
        "previewUrl": _audio_url(song),
        **metadata,
    }


def _safe_music_path(config: ApiConfig, song: str) -> Path:
    music_dir = config.music_dir.resolve()
    candidate = (music_dir / f"{song}.mp3").resolve()
    if candidate.parent != music_dir or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Song not found")
    return candidate


def normalize_playback(sim_data: dict[str, Any], backend: AppBackend) -> dict[str, Any]:
    """Convert an AxSwarm/Crazyflow sim log into the browser replay contract."""
    timestamps = np.asarray(sim_data["timestamps"], dtype=float)
    states = np.asarray(sim_data["states"], dtype=float)
    if states.ndim != 3:
        raise ValueError(f"Expected states with shape (T, drones, 13), got {states.shape}")
    if states.shape[0] != timestamps.shape[0]:
        raise ValueError(f"State/timestamp mismatch: {states.shape[0]} != {timestamps.shape[0]}")
    if states.shape[2] != 13:
        raise ValueError(f"Expected state vector length 13, got {states.shape[2]}")
    num_drones = int(sim_data["num_drones"])
    if states.shape[1] != num_drones:
        raise ValueError(f"State drone count mismatch: {states.shape[1]} != {num_drones}")

    colors = generate_default_colors(num_drones, limit=1.0)
    bounds = backend.settings["axswarm"]
    sample_rate = 0.0
    if len(timestamps) > 1:
        sample_rate = float(1.0 / np.median(np.diff(timestamps)))
    return {
        "schemaVersion": 1,
        "audioUrl": _audio_url(backend.music_manager.song),
        "song": backend.music_manager.song,
        "numDrones": num_drones,
        "timestamps": timestamps.tolist(),
        "states": states.tolist(),
        "fields": {"pos": [0, 3], "quat": [3, 7], "vel": [7, 10], "angVel": [10, 13]},
        "bounds": {"min": bounds["pos_min"], "max": bounds["pos_max"]},
        "colors": colors.tolist(),
        "sampleRate": sample_rate,
    }


def _start_thread(job: Job, target: Any) -> None:
    thread = threading.Thread(target=target, name=f"swarmgpt-job-{job.id}", daemon=True)
    job.thread = thread
    thread.start()


def _run_initial_job(store: JobStore, job: Job, selection: str) -> None:
    try:
        store.emit(job, "thinking_started", {"selection": selection}, status="thinking")
        messages = job.backend.initial_prompt(selection)
        store.emit(job, "conversation", {"messages": messages})
        store.emit(job, "safety_started", {}, status="filtering")
        sim_data = _run_simulation_with_events(job.backend, store, job)
        playback = normalize_playback(sim_data, job.backend)
        with store._lock:
            job.playback = playback
        store.emit(
            job,
            "ready",
            {
                "playbackUrl": f"/api/jobs/{job.id}/playback",
                "duration": playback["timestamps"][-1] if playback["timestamps"] else 0.0,
            },
            status="ready",
        )
    except BaseException as exc:
        store.fail(job, exc)


def _run_refine_job(
    store: JobStore, job: Job, message: str, provider: LLMProvider | None, model_id: str | None
) -> None:
    try:
        if provider is not None and model_id:
            job.backend.choreographer.configure_llm(provider, model_id)
            store.emit(job, "llm_configured", {"provider": provider, "modelId": model_id})
        store.emit(job, "thinking_started", {"refine": True}, status="thinking")
        messages = job.backend.reprompt(message)
        store.emit(job, "conversation", {"messages": messages})
        store.emit(job, "safety_started", {"refine": True}, status="filtering")
        sim_data = _run_simulation_with_events(job.backend, store, job)
        playback = normalize_playback(sim_data, job.backend)
        with store._lock:
            job.playback = playback
        store.emit(
            job,
            "ready",
            {
                "playbackUrl": f"/api/jobs/{job.id}/playback",
                "duration": playback["timestamps"][-1] if playback["timestamps"] else 0.0,
                "refined": True,
            },
            status="ready",
        )
    except BaseException as exc:
        store.fail(job, exc)


def _run_deploy_job(store: JobStore, job: Job) -> None:
    try:
        store.emit(job, "deploy_started", {}, status="deploying")
        deployed = job.backend.deploy()
        if deployed is False:
            raise RuntimeError("ROS2 is not installed. Switch to the deploy environment.")
        store.emit(job, "deploy_complete", {}, status="ready")
    except BaseException as exc:
        store.fail(job, exc)


def create_app(config: ApiConfig | None = None) -> FastAPI:
    """Create the SwarmGPT browser API app."""
    config = config or ApiConfig()
    store = JobStore()
    app = FastAPI(title="SwarmGPT Browser API")

    @app.get("/api/library")
    def library() -> dict[str, Any]:
        backend = _backend_from_config(config, config.llm_provider, config.model_id)
        songs = [
            {"id": song, "label": song, "kind": "song", "previewUrl": _audio_url(song)}
            for song in backend.songs
        ]
        presets = [_preset_item(backend, preset) for preset in backend.presets]
        return {"songs": songs, "presets": presets}

    @app.get("/api/llm")
    def llm() -> dict[str, Any]:
        return {
            "providers": [
                {
                    "id": "openai",
                    "label": PROVIDER_LABEL_OPENAI,
                    "models": list(DEFAULT_OPENAI_MODEL_CHOICES),
                    "defaultModel": default_openai_model(),
                },
                {
                    "id": "ollama",
                    "label": PROVIDER_LABEL_OLLAMA,
                    "models": ollama_installed_model_names(),
                    "defaultModel": None,
                },
            ],
            "defaultProvider": config.llm_provider,
            "defaultModel": config.model_id,
        }

    @app.get("/api/assets/swarm.png")
    def swarm_background() -> FileResponse:
        if not SWARM_BACKGROUND.is_file():
            raise HTTPException(status_code=404, detail="Background image not found")
        return FileResponse(SWARM_BACKGROUND)

    @app.get("/api/assets/drone/{asset_path:path}")
    def drone_asset(asset_path: str) -> FileResponse:
        asset_root = DRONE_ASSET_DIR.resolve()
        candidate = (asset_root / asset_path).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(asset_root):
            raise HTTPException(status_code=404, detail="Drone asset not found")
        return FileResponse(candidate)

    @app.get("/api/media/music/{song}")
    def music(song: str) -> FileResponse:
        return FileResponse(_safe_music_path(config, song), media_type="audio/mpeg")

    @app.post("/api/jobs", status_code=202)
    def create_job(request: JobRequest) -> dict[str, Any]:
        backend = _backend_from_config(config, request.provider, request.model_id)
        if request.selection not in backend.songs and request.selection not in backend.presets:
            raise HTTPException(status_code=404, detail="Song or preset not found")
        job = store.create(backend)
        _start_thread(job, lambda: _run_initial_job(store, job, request.selection))
        return {"jobId": job.id, "eventsUrl": f"/api/jobs/{job.id}/events"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        try:
            return store.snapshot(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None

    @app.websocket("/api/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        try:
            store.get(job_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        cursor = 0
        try:
            while True:
                events = store.events_after(job_id, cursor)
                for event in events:
                    cursor = max(cursor, int(event["id"]))
                    await websocket.send_json(event)
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    @app.get("/api/jobs/{job_id}/playback")
    def playback(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        if job.playback is None:
            raise HTTPException(status_code=409, detail="Playback is not ready")
        return job.playback

    @app.post("/api/jobs/{job_id}/refine", status_code=202)
    def refine(job_id: str, request: RefineRequest) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        if job.is_running:
            raise HTTPException(status_code=409, detail="Job is already running")
        _start_thread(
            job,
            lambda: _run_refine_job(
                store, job, request.message, request.provider, request.model_id
            ),
        )
        return {"jobId": job.id}

    @app.post("/api/jobs/{job_id}/deploy", status_code=202)
    def deploy(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        if job.is_running:
            raise HTTPException(status_code=409, detail="Job is already running")
        if not job.backend.splines:
            raise HTTPException(status_code=409, detail="Run the safety filter first")
        _start_thread(job, lambda: _run_deploy_job(store, job))
        return {"jobId": job.id}

    @app.post("/api/jobs/{job_id}/preset")
    def save_preset(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        if job.is_running:
            raise HTTPException(status_code=409, detail="Job is still running")
        if not job.backend.splines:
            raise HTTPException(status_code=409, detail="Run the safety filter first")
        try:
            preset_id = job.backend.save_preset()
            preset = _preset_item(job.backend, preset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        store.emit(job, "preset_saved", {"preset": preset})
        return {"preset": preset}

    @app.delete("/api/presets/{preset_id:path}")
    def delete_preset(preset_id: str) -> dict[str, Any]:
        backend = _backend_from_config(config, config.llm_provider, config.model_id)
        try:
            backend.delete_preset(preset_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Preset not found") from None
        return {"deleted": preset_id}

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if not WEB_DIST_DIR.is_dir():
            raise HTTPException(status_code=404, detail="Frontend build not found")
        requested = (WEB_DIST_DIR / full_path).resolve()
        web_root = WEB_DIST_DIR.resolve()
        if requested.is_file() and requested.is_relative_to(web_root):
            return FileResponse(requested)
        return FileResponse(web_root / "index.html")

    return app


app = create_app()
