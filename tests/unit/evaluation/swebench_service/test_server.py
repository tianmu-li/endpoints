# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
)
sys.path.insert(0, str(_SERVICE_ROOT))

from swebench_service.config import ServiceConfig  # noqa: E402
from swebench_service.server import create_app  # noqa: E402

pytestmark = pytest.mark.unit


class FakeRunner:
    def __init__(self, *, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.requests: list[object] = []

    def run(self, request, run_dir: Path):
        self.requests.append(request)
        if self.delay:
            import time

            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("runner failed")
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":1,"submitted_instances":1}'
        )
        return {"resolved_instances": 1, "submitted_instances": 1}


def _payload() -> dict:
    return {
        "model_name": "test-model",
        "endpoint_urls": ["http://endpoint"],
        "endpoint_api_key": "secret",
        "generation_params": {"temperature": 0.0},
        "subset": "lite",
        "split": "test",
        "num_instances": 1,
        "workers": 1,
        "max_eval_workers": 1,
        "evaluated_instance_ids": ["repo__repo-1"],
    }


async def _client(tmp_path: Path, runner) -> TestClient:
    app = create_app(
        ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1),
        runner=runner,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _auth_client(tmp_path: Path, runner) -> TestClient:
    app = create_app(
        ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1, auth_token="tok"),
        runner=runner,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_health_response_schema(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        resp = await client.get("/health")
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 200
    assert body["api_version"] == "v1"
    assert "swebench.run" in body["capabilities"]
    assert "artifacts.download" in body["capabilities"]


@pytest.mark.asyncio
async def test_post_run_validates_requests(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        resp = await client.post("/v1/runs", json={"model_name": ""})
    finally:
        await client.close()

    assert resp.status == 400


@pytest.mark.asyncio
async def test_optional_auth_token_is_enforced(tmp_path):
    client = await _auth_client(tmp_path, FakeRunner())
    try:
        unauthorized = await client.get("/health")
        authorized = await client.get(
            "/health", headers={"Authorization": "Bearer tok"}
        )
    finally:
        await client.close()

    assert unauthorized.status == 401
    assert authorized.status == 200


@pytest.mark.asyncio
async def test_runner_transitions_to_succeeded(tmp_path):
    runner = FakeRunner()
    client = await _client(tmp_path, runner)
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert submit.status == 202
    assert status["status"] == "succeeded"
    assert status["result"] == {"resolved_instances": 1, "submitted_instances": 1}
    assert runner.requests[0].evaluated_instance_ids == ["repo__repo-1"]


@pytest.mark.asyncio
async def test_runner_transitions_to_failed(tmp_path):
    client = await _client(tmp_path, FakeRunner(fail=True))
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert status["status"] == "failed"
    assert "runner failed" in status["error"]


@pytest.mark.asyncio
async def test_bounded_concurrency_returns_429(tmp_path):
    client = await _client(tmp_path, FakeRunner(delay=0.2))
    try:
        first = await client.post("/v1/runs", json=_payload())
        second = await client.post("/v1/runs", json=_payload())
    finally:
        await client.close()

    assert first.status == 202
    assert second.status == 429


@pytest.mark.asyncio
async def test_artifact_endpoint_blocks_path_traversal(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        run_id = submitted["run_id"]
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            if status["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        ok = await client.get(f"/v1/runs/{run_id}/artifacts/preds.json")
        blocked = await client.get(f"/v1/runs/{run_id}/artifacts/..%2Fstatus.json")
    finally:
        await client.close()

    assert ok.status == 200
    assert blocked.status == 404


@pytest.mark.asyncio
async def test_status_redacts_api_keys(tmp_path):
    client = await _client(tmp_path, FakeRunner(delay=0.05))
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        status_path = tmp_path / submitted["run_id"] / "status.json"
        request_path = tmp_path / submitted["run_id"] / "request.json"
        for _ in range(100):
            if request_path.exists():
                break
            await asyncio.sleep(0.01)
        request_text = request_path.read_text()
        status_text = status_path.read_text()
    finally:
        await client.close()

    assert "secret" not in request_text
    assert "<redacted>" in request_text
    assert "secret" not in status_text


@pytest.mark.asyncio
async def test_failed_status_redacts_secret_values(tmp_path):
    class SecretFailRunner:
        def run(self, request, run_dir: Path):
            raise RuntimeError(f"failed with api_key={request.endpoint_api_key}")

    client = await _client(tmp_path, SecretFailRunner())
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert "secret" not in status["error"]
    assert "<redacted>" in status["error"]
