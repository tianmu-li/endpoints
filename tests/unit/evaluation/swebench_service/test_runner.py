# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import threading
from pathlib import Path

import msgspec.json
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

from swebench_service import runner as runner_mod  # noqa: E402
from swebench_service.runner import (  # noqa: E402
    CancellationToken,
    RunCancelled,
    RunnerError,
    SwebenchRunner,
)
from swebench_service.schemas import RunRequest  # noqa: E402

pytestmark = pytest.mark.unit


def _request(endpoints: list[str]) -> RunRequest:
    return RunRequest(
        model_name="test-model",
        endpoint_urls=endpoints,
        endpoint_api_key=None,
        generation_params={"name": "test-model"},
        subset="lite",
        split="test",
        num_instances=1,
        workers=1,
        max_eval_workers=1,
        evaluated_instance_ids=["repo__repo-1"],
    )


def test_run_subprocess_streams_output_to_log(tmp_path):
    log_path = tmp_path / "subprocess.log"

    runner_mod._run_subprocess(
        [sys.executable, "-c", "print('first'); print('second')"],
        log_path,
        cwd=tmp_path,
        timeout_s=5,
    )

    assert log_path.read_text() == "first\nsecond\n"


def test_run_subprocess_reports_bounded_failure_tail(tmp_path):
    log_path = tmp_path / "subprocess.log"
    script = (
        "import sys\n"
        "print('early-marker')\n"
        "for i in range(700): print(f'{i:04d}-' + 'x' * 100)\n"
        "print('final-marker')\n"
        "sys.exit(7)\n"
    )

    with pytest.raises(RunnerError, match="exited with code 7") as exc_info:
        runner_mod._run_subprocess(
            [sys.executable, "-c", script],
            log_path,
            cwd=tmp_path,
            timeout_s=5,
        )

    assert "early-marker" in log_path.read_text()
    failure_tail = str(exc_info.value).partition("\n")[2]
    assert "final-marker" in failure_tail
    assert "early-marker" not in failure_tail


def test_run_subprocess_timeout_preserves_partial_log(tmp_path):
    log_path = tmp_path / "subprocess.log"

    with pytest.raises(RunnerError, match="timed out after 1s"):
        runner_mod._run_subprocess(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path,
            cwd=tmp_path,
            timeout_s=1,
        )

    assert log_path.read_text() == "started\n"


def test_run_subprocess_cancellation_preserves_partial_log(tmp_path):
    log_path = tmp_path / "subprocess.log"
    cancel_token = CancellationToken()
    cancel_timer = threading.Timer(0.2, cancel_token.cancel)
    cancel_timer.start()
    try:
        with pytest.raises(RunCancelled, match="subprocess cancelled"):
            runner_mod._run_subprocess(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                log_path,
                cwd=tmp_path,
                timeout_s=5,
                cancel_token=cancel_token,
            )
    finally:
        cancel_timer.cancel()
        cancel_timer.join()

    assert log_path.read_text() == "started\n"


def test_base_env_keeps_proxies_and_sets_no_proxy_for_loopback(monkeypatch, tmp_path):
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("all_proxy", "socks5://proxy.example:1080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example:1080")
    monkeypatch.setenv("NO_PROXY", "intel.com")

    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://localhost:30000"]))

    assert env["http_proxy"] == "http://proxy.example:8080"
    assert env["https_proxy"] == "http://proxy.example:8080"
    assert env["HTTP_PROXY"] == "http://proxy.example:8080"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert env["all_proxy"] == "socks5://proxy.example:1080"
    assert env["ALL_PROXY"] == "socks5://proxy.example:1080"
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
        _request(["http://localhost:30000"]),
    )

    cfg = yaml.safe_load(patched.read_text())
    assert cfg["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:30000/v1"
    assert cfg["model"]["model_kwargs"]["api_key"] == "EMPTY"


def test_run_agent_filters_exact_instance_ids(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run_subprocess(cmd, *args, **kwargs):
        commands.append(cmd)

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)
    request = _request(["http://endpoint:30000"])
    request.evaluated_instance_ids = ["repo__repo-1", "repo.with.regex+chars"]
    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._run_agent(request, tmp_path / "config.yaml", tmp_path, tmp_path)

    cmd = commands[0]
    assert "--slice" not in cmd
    assert cmd[cmd.index("--filter") + 1] == (
        "^(?:repo__repo\\-1|repo\\.with\\.regex\\+chars)$"
    )


def test_run_agent_toolcall_patch_prepends_overlay_pythonpath(monkeypatch, tmp_path):
    envs: list[dict[str, str]] = []
    overlay = tmp_path / "overlay"
    request = _request(["http://endpoint:30000"])
    request.enable_swebench_toolcall_patch = True
    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_create_overlay(self, overlay_root, replacement_root):
        assert replacement_root == self._template_dir
        overlay.mkdir()
        return overlay

    def fake_run_subprocess(cmd, log_path, *, env, **kwargs):
        envs.append(env)

    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    monkeypatch.setattr(
        SwebenchRunner, "_create_toolcall_patch_overlay", fake_create_overlay
    )
    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    runner._run_agent(request, tmp_path / "config.yaml", tmp_path, tmp_path)

    pythonpath = envs[0]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[:2] == [str(overlay), "/existing/path"]


def test_validate_prediction_ids_rejects_unexpected_instances(tmp_path):
    request = _request(["http://endpoint:30000"])
    request.evaluated_instance_ids = ["repo__repo-1"]
    preds = tmp_path / "preds.json"
    preds.write_bytes(
        msgspec.json.encode({"repo__repo-1": "patch", "repo__repo-2": "patch"})
    )
    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    with pytest.raises(RunnerError, match="unexpected SWE-bench"):
        runner._validate_prediction_ids(request, preds)


def test_run_eval_persists_harness_run_id(monkeypatch, tmp_path):
    runner = SwebenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    request = _request(["http://endpoint:30000"])
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "run"
    output_dir.mkdir()
    run_dir.mkdir()
    preds_path = output_dir / "preds.json"
    preds_path.write_text('{"repo__repo-1":"patch"}')

    def fake_run_subprocess(cmd, log_path, *, cwd, **kwargs):
        assert cmd[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
        run_id = cmd[cmd.index("--run_id") + 1]
        assert (run_dir / "swe_bench_eval_run_id.txt").read_text() == run_id
        (cwd / f"test-model.{run_id}.json").write_text(
            '{"resolved_instances":1,"submitted_instances":1}'
        )

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    result_path = runner._run_eval(request, preds_path, output_dir, run_dir)

    assert result_path.exists()
    assert (run_dir / "swe_bench_eval_run_id.txt").read_text().startswith("endpoints_")
