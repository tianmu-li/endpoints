# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Qwen mini-swe-agent toolcall replacements."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_TEMPLATES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
    / "swebench_service"
    / "templates"
)
_ACTIONS_TOOLCALL = _TEMPLATES_DIR / "actions_toolcall.py"
_LITELLM_MODEL = _TEMPLATES_DIR / "litellm_model.py"


def _load_actions_module(monkeypatch):
    class FormatError(Exception):
        pass

    exceptions_mod = types.ModuleType("minisweagent.exceptions")
    exceptions_mod.FormatError = FormatError
    multimodal_mod = types.ModuleType("minisweagent.models.utils.openai_multimodal")
    multimodal_mod.expand_multimodal_content = lambda msg, pattern: msg

    for name in [
        "minisweagent",
        "minisweagent.models",
        "minisweagent.models.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions_mod)
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models.utils.openai_multimodal",
        multimodal_mod,
    )

    spec = importlib.util.spec_from_file_location(
        "_test_actions_toolcall", _ACTIONS_TOOLCALL
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_litellm_model_module(monkeypatch):
    class LitellmError(Exception):
        pass

    litellm_mod = types.ModuleType("litellm")
    litellm_mod.exceptions = SimpleNamespace(
        UnsupportedParamsError=LitellmError,
        NotFoundError=LitellmError,
        PermissionDeniedError=LitellmError,
        ContextWindowExceededError=LitellmError,
        AuthenticationError=LitellmError,
    )
    litellm_mod.utils = SimpleNamespace(register_model=lambda model: None)

    models_mod = types.ModuleType("minisweagent.models")
    models_mod.GLOBAL_MODEL_STATS = SimpleNamespace(add=lambda cost: None)
    actions_mod = types.ModuleType("minisweagent.models.utils.actions_toolcall")
    actions_mod.TOOL_SCHEMAS = []
    actions_mod.format_toolcall_observation_messages = lambda **kwargs: []
    actions_mod.parse_toolcall_actions = lambda *args, **kwargs: []
    anthropic_mod = types.ModuleType("minisweagent.models.utils.anthropic_utils")
    anthropic_mod._reorder_anthropic_thinking_blocks = lambda messages: messages
    cache_mod = types.ModuleType("minisweagent.models.utils.cache_control")
    cache_mod.set_cache_control = lambda messages, mode: messages
    multimodal_mod = types.ModuleType("minisweagent.models.utils.openai_multimodal")
    multimodal_mod.expand_multimodal_content = lambda message, pattern: message
    retry_mod = types.ModuleType("minisweagent.models.utils.retry")
    retry_mod.retry = lambda **kwargs: []

    for name, module in {
        "litellm": litellm_mod,
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.models": models_mod,
        "minisweagent.models.utils": types.ModuleType("minisweagent.models.utils"),
        "minisweagent.models.utils.actions_toolcall": actions_mod,
        "minisweagent.models.utils.anthropic_utils": anthropic_mod,
        "minisweagent.models.utils.cache_control": cache_mod,
        "minisweagent.models.utils.openai_multimodal": multimodal_mod,
        "minisweagent.models.utils.retry": retry_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_test_litellm_model", _LITELLM_MODEL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_litellm_model_loopback_preserves_proxy_environment(monkeypatch):
    proxy_vars = {
        "http_proxy": "http://proxy.example:8080",
        "https_proxy": "http://proxy.example:8080",
        "HTTP_PROXY": "http://proxy.example:8080",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "all_proxy": "socks5://proxy.example:1080",
        "ALL_PROXY": "socks5://proxy.example:1080",
    }
    for name, value in proxy_vars.items():
        monkeypatch.setenv(name, value)

    litellm_model = _load_litellm_model_module(monkeypatch)
    config = SimpleNamespace(
        model_kwargs={"api_base": "http://127.0.0.1:30000/v1"},
        litellm_model_registry=None,
    )

    litellm_model.LitellmModel(config_class=lambda **kwargs: config)

    assert {name: os.environ[name] for name in proxy_vars} == proxy_vars


def test_finish_emits_relative_pathspecs_and_git_add_intent(monkeypatch):
    actions_mod = _load_actions_module(monkeypatch)

    actions = actions_mod.parse_toolcall_actions(
        [
            _tool_call(
                "finish",
                {
                    "files_modified": [
                        "/testbed/pkg/new file.py",
                        "tests/test_widget.py",
                    ]
                },
            )
        ],
        format_error_template="{{ error }}",
    )

    command = actions[0]["command"]
    assert (
        command == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
        "git -C /testbed add -N -- 'pkg/new file.py' tests/test_widget.py && "
        "git -C /testbed diff HEAD -- 'pkg/new file.py' tests/test_widget.py"
    )
    assert "/testbed/pkg/new file.py" not in command


def test_str_replace_editor_view_range_emits_clean_awk(monkeypatch):
    actions_mod = _load_actions_module(monkeypatch)

    actions = actions_mod.parse_toolcall_actions(
        [
            _tool_call(
                "str_replace_editor",
                {
                    "command": "view",
                    "path": "/testbed/pkg/file.py",
                    "view_range": [2, 4],
                },
            )
        ],
        format_error_template="{{ error }}",
    )

    command = actions[0]["command"]
    assert (
        command == "awk 'NR>=2 && NR<=4 "
        '{printf "%6d\\t%s\\n", NR, $0}\' /testbed/pkg/file.py'
    )
    assert r"\'" not in command


def test_str_replace_editor_view_range_rejects_non_integers(monkeypatch):
    actions_mod = _load_actions_module(monkeypatch)

    with pytest.raises(
        actions_mod.FormatError, match="view_range values must be integers"
    ):
        actions_mod.parse_toolcall_actions(
            [
                _tool_call(
                    "str_replace_editor",
                    {
                        "command": "view",
                        "path": "/testbed/pkg/file.py",
                        "view_range": ["one", 4],
                    },
                )
            ],
            format_error_template="{{ error }}",
        )
