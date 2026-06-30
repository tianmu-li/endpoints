# SWE-bench Accuracy Smoke-Test Runbook

End-to-end validation for the SWE-bench accuracy pipeline. Unit tests mock all
subprocesses, so running the real pipeline is the only way to catch Docker,
HuggingFace access, or mini-swe-agent wiring issues.

## 0. Preconditions

- Docker daemon running (swebench harness spawns one container per instance).
- Docker Hub auth or a pre-seeded image cache for uncached SWE-bench images.
- Network egress to PyPI and HuggingFace Hub.
- `uv` binary on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- `patch` binary on PATH (`sudo apt-get install patch`).
- Parent endpoints env already synced (`uv sync --extra dev` from repo root).

## 1. Sync the accuracy subproject

From the repo root:

```bash
cd examples/10_Agentic_Inference/accuracy
uv sync
```

Sanity check:

```bash
uv run mini-extra --help
uv run python -m swebench.harness.run_evaluation --help
```

Override the default subproject path via env var if needed:

```bash
export SWE_BENCH_PROJECT_PATH=/path/to/examples/10_Agentic_Inference/accuracy
```

## 2. End-to-end test (requires live endpoint)

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_Agentic_Inference/swe_bench_accuracy.yaml
```

Scorer preflight resolves the requested SWE-bench instances, pre-pulls the
required Docker images, and applies `finish_tool.patch` to the minisweagent
install. The patch adds `finish` and `str_replace_editor` tools to the agent:
`finish` auto-extracts the git diff on submit (eliminating the manual
patch.txt ceremony), and `str_replace_editor` provides exact-match atomic file
edits. The patch is idempotent — re-running the benchmark does not re-apply it.

## Common failure modes

| Symptom                                              | Likely cause                          | Fix                                                       |
| ---------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| `FileNotFoundError: SWE-bench subproject not found`  | subproject not synced                 | Run `uv sync` in `examples/10_Agentic_Inference/accuracy` |
| `patch is not on PATH`                               | patch binary missing                  | `sudo apt-get install patch`                              |
| `Failed to apply finish_tool.patch`                  | patch version mismatch                | Re-sync with `uv sync` and retry                          |
| Docker error during `run_evaluation`                 | Docker daemon not running             | Start Docker and retry                                    |
| `Failed to pre-pull required SWE-bench Docker image` | Docker Hub rate limit or missing auth | Run `docker login` or use a local image cache/mirror      |
