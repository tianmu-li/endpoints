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

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import msgspec.json
import yaml

from .artifacts import redact_secrets
from .schemas import RunRequest, TemplateName


class RunnerError(RuntimeError):
    pass


TEMPLATE_FILES: dict[TemplateName, str] = {
    "default": "swebench_template.yaml",
    "qwen_tools": "swebench_qwen_tools_template.yaml",
}


def _normalize_endpoint_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    parsed = urlparse(base)
    if parsed.hostname == "localhost":
        netloc = "127.0.0.1"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        base = urlunparse(parsed._replace(netloc=netloc))
    return base


def _exact_instance_filter(instance_ids: list[str]) -> str:
    return (
        "^(?:" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"
    )


def _run_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> None:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            cwd=str(cwd),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", errors="replace")
        )
        log_path.write_text(partial)
        raise RunnerError(f"subprocess timed out after {timeout_s}s: {cmd}") from exc
    log_path.write_text(completed.stdout or "")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-50:])
        raise RunnerError(
            f"subprocess exited with code {completed.returncode}: {cmd}\n{tail}"
        )


class SwebenchRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
    ):
        self.project_root = project_root.resolve()
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(self, request: RunRequest, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = (
            {request.endpoint_api_key} if request.endpoint_api_key else set()
        )
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(request.model_dump(), secret_values=secret_values)
            )
        )

        output_dir = run_dir / "swe_bench_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tempfile.TemporaryDirectory(prefix="swebench_config_") as config_tmp:
            patched_config = self._patch_config(
                Path(config_tmp),
                request,
            )
            self._run_agent(request, patched_config, output_dir, run_dir)

        preds_path = output_dir / "preds.json"
        if not preds_path.exists():
            raise RunnerError("mini-extra did not produce preds.json")
        self._validate_prediction_ids(request, preds_path)
        shutil.copy2(preds_path, run_dir / "preds.json")

        result_path = self._run_eval(request, preds_path, output_dir, run_dir)
        shutil.copy2(result_path, run_dir / "swe_bench_results.json")
        return msgspec.json.decode(result_path.read_bytes(), type=dict)

    def _load_template(self, request: RunRequest) -> dict[str, Any]:
        template_path = self._template_dir / TEMPLATE_FILES[request.template]
        with template_path.open() as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise RunnerError("swebench template must be a YAML mapping")
        model_cfg = loaded.get("model")
        if not isinstance(model_cfg, dict):
            raise RunnerError("swebench template must define model")
        if not isinstance(model_cfg.get("model_kwargs"), dict):
            raise RunnerError("swebench template must define model.model_kwargs")
        return loaded

    @property
    def _template_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"

    def _patch_config(self, config_dir: Path, request: RunRequest) -> Path:
        cfg = self._load_template(request)
        model_cfg = cfg["model"]
        model_kwargs = model_cfg["model_kwargs"]

        model_cfg["model_name"] = request.model_name
        if request.endpoint_urls:
            base = _normalize_endpoint_base(str(request.endpoint_urls[0]))
            model_kwargs["api_base"] = base + "/v1"
        else:
            base = ""
            model_kwargs["api_base"] = ""

        if request.endpoint_api_key:
            model_kwargs["api_key"] = request.endpoint_api_key
        elif urlparse(base).hostname in {"localhost", "127.0.0.1", "::1"}:
            model_kwargs["api_key"] = "EMPTY"
        else:
            model_kwargs.pop("api_key", None)

        for field in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = request.generation_params.get(field)
            if val is not None:
                model_kwargs[field] = val
            else:
                model_kwargs.pop(field, None)

        if (
            max_new_tokens := request.generation_params.get("max_new_tokens")
        ) is not None:
            model_kwargs["max_tokens"] = max_new_tokens
        else:
            model_kwargs.pop("max_tokens", None)

        if (
            chat_tmpl := request.generation_params.get("chat_template_kwargs")
        ) is not None:
            model_kwargs["chat_template_kwargs"] = chat_tmpl
        else:
            model_kwargs.pop("chat_template_kwargs", None)

        config_dir.mkdir(parents=True, exist_ok=True)
        patched_path = config_dir / "swebench_patched.yaml"
        with patched_path.open("w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return patched_path

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
    ) -> None:
        instance_filter = _exact_instance_filter(request.evaluated_instance_ids)
        cmd = [
            "mini-extra",
            "swebench",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            instance_filter,
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
        ]
        if request.enable_swebench_toolcall_patch:
            with tempfile.TemporaryDirectory(prefix="minisweagent_overlay_") as tmp:
                env = self._agent_env(request, Path(tmp))
                _run_subprocess(
                    cmd,
                    run_dir / "swe_bench_agent.log",
                    cwd=output_dir,
                    timeout_s=self.subprocess_timeout_s,
                    env=env,
                )
                return
        _run_subprocess(
            cmd,
            run_dir / "swe_bench_agent.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=self._base_env(request),
        )

    def _base_env(self, request: RunRequest) -> dict[str, str]:
        env = dict(os.environ)
        no_proxy = {"127.0.0.1", "localhost"}
        for endpoint in request.endpoint_urls:
            host = urlparse(str(endpoint)).hostname
            if host:
                no_proxy.add(host)
        existing = env.get("NO_PROXY") or env.get("no_proxy")
        if existing:
            no_proxy.update(
                part.strip() for part in existing.split(",") if part.strip()
            )
        no_proxy_value = ",".join(sorted(no_proxy))
        env["NO_PROXY"] = no_proxy_value
        env["no_proxy"] = no_proxy_value
        return env

    def _agent_env(self, request: RunRequest, overlay_root: Path) -> dict[str, str]:
        env = self._base_env(request)
        overlay = self._create_toolcall_patch_overlay(overlay_root, self._template_dir)
        pythonpath = [str(overlay)]
        if existing := env.get("PYTHONPATH"):
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env

    def _create_toolcall_patch_overlay(
        self, overlay_root: Path, replacement_root: Path
    ) -> Path:
        site_packages = self._resolve_minisweagent_site_packages()
        package_src = site_packages / "minisweagent"
        if not package_src.is_dir():
            raise RunnerError(
                f"minisweagent package directory not found: {package_src}"
            )
        package_dest = overlay_root / "minisweagent"
        shutil.copytree(package_src, package_dest)
        replacements = {
            "actions_toolcall.py": "minisweagent/models/utils/actions_toolcall.py",
            "litellm_model.py": "minisweagent/models/litellm_model.py",
        }
        for src_name, rel_dest in replacements.items():
            src = replacement_root / src_name
            if not src.exists():
                raise RunnerError(
                    "enable_swebench_toolcall_patch requested, but replacement "
                    f"file is missing on the service host: {src}"
                )
            shutil.copy2(src, overlay_root / rel_dest)
        return overlay_root

    def _validate_prediction_ids(self, request: RunRequest, preds_path: Path) -> None:
        try:
            preds = msgspec.json.decode(preds_path.read_bytes(), type=dict)
        except msgspec.DecodeError as exc:
            raise RunnerError("mini-extra produced invalid preds.json") from exc
        expected = set(request.evaluated_instance_ids)
        actual = {str(instance_id) for instance_id in preds}
        unexpected = sorted(actual - expected)
        if unexpected:
            raise RunnerError(
                "mini-extra produced predictions for unexpected SWE-bench "
                f"instances: {', '.join(unexpected[:10])}"
            )

    def _resolve_minisweagent_site_packages(self) -> Path:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import minisweagent.models.utils.actions_toolcall as m; print(m.__file__)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RunnerError("could not locate minisweagent: " + result.stderr.strip())
        last_line = next(
            (line for line in reversed(result.stdout.splitlines()) if line.strip()),
            "",
        )
        if not last_line:
            raise RunnerError("could not locate minisweagent: empty output")
        actions_toolcall = Path(last_line.strip())
        try:
            site_packages = actions_toolcall.parents[3]
        except IndexError as exc:
            raise RunnerError(
                f"could not resolve site-packages from {actions_toolcall}"
            ) from exc
        if not site_packages.is_dir():
            raise RunnerError(f"resolved site-packages does not exist: {site_packages}")
        return site_packages

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
    ) -> Path:
        run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
        dataset_name = {
            "verified": "princeton-nlp/SWE-bench_Verified",
            "lite": "princeton-nlp/SWE-bench_Lite",
        }.get(request.subset)
        if dataset_name is None:
            raise RunnerError(f"unknown SWE-bench subset: {request.subset}")
        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            request.split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(request.max_eval_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            *request.evaluated_instance_ids,
        ]
        _run_subprocess(
            cmd,
            run_dir / "swe_bench_eval.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
        )
        safe_model = request.model_name.replace("/", "__")
        result_path = output_dir / f"{safe_model}.{run_id}.json"
        if result_path.exists():
            return result_path
        candidates = sorted(output_dir.rglob(f"*{run_id}*.json"))
        if not candidates:
            raise RunnerError(f"SWE-bench result file not found for run_id={run_id}")
        return candidates[0]
