# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import yaml

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
)
sys.path.insert(0, str(_SERVICE_ROOT))

from swebench_service.runner import SwebenchRunner  # noqa: E402
from swebench_service.schemas import RunRequest  # noqa: E402

pytestmark = pytest.mark.unit


def _request(endpoints: list[str]) -> RunRequest:
    return RunRequest(
        benchmark_config={
            "model_params": {"name": "test-model"},
            "endpoint_config": {"endpoints": endpoints},
        },
        model_name="test-model",
        subset="lite",
        split="test",
        num_instances=1,
        workers=1,
        max_eval_workers=1,
        evaluated_instance_ids=["repo__repo-1"],
    )


def test_base_env_keeps_proxies_and_sets_no_proxy_for_loopback(monkeypatch, tmp_path):
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "intel.com")

    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://localhost:30000"]))

    assert env["http_proxy"] == "http://proxy.example:8080"
    assert env["https_proxy"] == "http://proxy.example:8080"
    assert {"127.0.0.1", "localhost", "intel.com"} <= set(env["NO_PROXY"].split(","))
    assert env["NO_PROXY"] == env["no_proxy"]


def test_base_env_keeps_proxies_for_non_loopback_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")

    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://swebench-host:30000"]))

    assert env["https_proxy"] == "http://proxy.example:8080"
    assert "swebench-host" in env["NO_PROXY"].split(",")


def test_patch_config_rewrites_localhost_api_base_to_127_0_0_1(tmp_path):
    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    patched = runner._patch_config(
        tmp_path,
        _request(["http://localhost:30000"]).benchmark_config,
        _request(["http://localhost:30000"]),
    )

    cfg = yaml.safe_load(patched.read_text())
    assert cfg["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:30000/v1"
    assert cfg["model"]["model_kwargs"]["api_key"] == "EMPTY"
