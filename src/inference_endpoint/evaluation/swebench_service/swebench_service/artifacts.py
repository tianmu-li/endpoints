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

from pathlib import Path
from typing import Any

SECRET_KEY_NAMES = {"api_key", "authorization", "access_token", "secret_key"}
SAFE_ARTIFACT_NAMES = {
    "preds.json",
    "swe_bench_agent.log",
    "swe_bench_eval.log",
    "swe_bench_results.json",
    "status.json",
}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if str(key).lower() in SECRET_KEY_NAMES
                else redact_secrets(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def resolve_artifact(run_dir: Path, name: str) -> Path:
    if name not in SAFE_ARTIFACT_NAMES or "/" in name or "\\" in name:
        raise FileNotFoundError(name)
    path = (run_dir / name).resolve()
    root = run_dir.resolve()
    if path.parent != root or not path.is_file():
        raise FileNotFoundError(name)
    return path
