# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import msgspec.json
from aiohttp import web
from pydantic import ValidationError

from . import API_VERSION, CAPABILITIES
from .artifacts import redact_secrets, resolve_artifact
from .config import ServiceConfig
from .runner import SwebenchRunner
from .schemas import ArtifactInfo, RunRequest, RunStatus


class RunManager:
    def __init__(self, *, config: ServiceConfig, runner: Any):
        self.config = config
        self.runner = runner
        self.runs: dict[str, RunStatus] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_runs)

    def active_count(self) -> int:
        return sum(
            1 for run in self.runs.values() if run.status in {"queued", "running"}
        )

    async def submit(self, request: RunRequest) -> RunStatus:
        if self.active_count() >= self.config.max_concurrent_runs:
            raise web.HTTPTooManyRequests(text="too many active SWE-bench runs")
        run_id = uuid.uuid4().hex
        now = time.time()
        status = RunStatus(
            run_id=run_id, status="queued", created_at=now, updated_at=now
        )
        self.runs[run_id] = status
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(redact_secrets(request.model_dump(mode="json")))
        )
        self._write_status(status)
        self._tasks[run_id] = asyncio.create_task(
            self._execute(run_id, request, run_dir)
        )
        return status

    async def _execute(self, run_id: str, request: RunRequest, run_dir: Path) -> None:
        async with self._semaphore:
            self._update(run_id, status="running")
            try:
                result = await asyncio.to_thread(self.runner.run, request, run_dir)
            except Exception as exc:
                self._update(run_id, status="failed", error=str(exc))
                return
            artifacts = [
                ArtifactInfo(name=name, url=f"/v1/runs/{run_id}/artifacts/{name}")
                for name in (
                    "preds.json",
                    "swe_bench_agent.log",
                    "swe_bench_eval.log",
                    "swe_bench_results.json",
                    "status.json",
                )
                if (run_dir / name).exists()
            ]
            self._update(run_id, status="succeeded", result=result, artifacts=artifacts)

    def _update(self, run_id: str, **updates: Any) -> None:
        current = self.runs[run_id]
        data = current.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        updated = RunStatus.model_validate(data)
        self.runs[run_id] = updated
        self._write_status(updated)

    def _write_status(self, status: RunStatus) -> None:
        run_dir = self.run_dir(status.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_bytes(
            msgspec.json.encode(redact_secrets(status.model_dump(mode="json")))
        )

    def get(self, run_id: str) -> RunStatus:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise web.HTTPNotFound(text="unknown run_id") from exc

    def run_dir(self, run_id: str) -> Path:
        return self.config.artifact_root / run_id


def create_app(config: ServiceConfig, runner: Any | None = None) -> web.Application:
    config = ServiceConfig(
        host=config.host,
        port=config.port,
        artifact_root=config.artifact_root.expanduser().resolve(),
        max_concurrent_runs=config.max_concurrent_runs,
        subprocess_timeout_s=config.subprocess_timeout_s,
    )
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    runner = runner or SwebenchRunner(
        project_root=Path(__file__).resolve().parents[1],
        subprocess_timeout_s=config.subprocess_timeout_s,
    )
    manager = RunManager(config=config, runner=runner)
    app = web.Application()
    app["manager"] = manager

    async def health(request: web.Request) -> web.Response:
        return web.json_response(
            {"api_version": API_VERSION, "capabilities": CAPABILITIES, "status": "ok"}
        )

    async def post_run(request: web.Request) -> web.Response:
        try:
            data = await request.json()
            run_request = RunRequest.model_validate(data)
        except (ValidationError, ValueError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        status = await manager.submit(run_request)
        return web.json_response(
            redact_secrets(status.model_dump(mode="json")), status=202
        )

    async def get_run(request: web.Request) -> web.Response:
        status = manager.get(request.match_info["run_id"])
        return web.json_response(redact_secrets(status.model_dump(mode="json")))

    async def get_artifact(request: web.Request) -> web.FileResponse:
        run_id = request.match_info["run_id"]
        name = request.match_info["name"]
        manager.get(run_id)
        try:
            path = resolve_artifact(manager.run_dir(run_id), name)
        except FileNotFoundError as exc:
            raise web.HTTPNotFound(text="artifact not found") from exc
        return web.FileResponse(path)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/runs", post_run)
    app.router.add_get("/v1/runs/{run_id}", get_run)
    app.router.add_get("/v1/runs/{run_id}/artifacts/{name}", get_artifact)
    return app
