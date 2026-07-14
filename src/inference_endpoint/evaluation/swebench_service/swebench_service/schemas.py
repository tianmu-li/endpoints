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

from typing import Any, Literal

from pydantic import BaseModel, Field

RunState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class RunRequest(BaseModel):
    benchmark_config: dict[str, Any]
    model_name: str = Field(min_length=1)
    subset: str = "verified"
    split: str = "test"
    num_instances: int = Field(ge=1)
    workers: int = Field(ge=1)
    max_eval_workers: int = Field(ge=1)
    evaluated_instance_ids: list[str] = Field(min_length=1)
    enable_swebench_toolcall_patch: bool = False
    swebench_config_template: str | None = None


class ArtifactInfo(BaseModel):
    name: str
    url: str


class RunStatus(BaseModel):
    run_id: str
    status: RunState
    created_at: float
    updated_at: float
    error: str | None = None
    result: dict[str, Any] | None = None
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
