# Collecting Outputs for Performance Runs

This example demonstrates how to collect and log model response outputs during performance benchmarks.

## Overview

By default, performance runs (`--mode perf`) do **not** collect response outputs to minimize memory overhead and I/O latency. However, for debugging, analysis, or archival purposes, you can enable output collection using the `--collect-outputs` CLI flag or the `collect_outputs: true` config field.

## When to Use

- **Debugging**: Save outputs for analyzing model behavior under load
- **Validation**: Verify response quality without slowing down the benchmark
- **Archival**: Store responses for compliance or future analysis
- **Combined metrics**: Analyze performance alongside response content

## Usage

### CLI Flag

```bash
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8000 \
  --model meta-llama/Llama-2-7b-hf \
  --dataset perf:data.jsonl \
  --collect-outputs
```

### YAML Config

```yaml
type: offline
collect_outputs: true
# ... rest of config
```

### From-config with override

```bash
uv run inference-endpoint benchmark from-config \
  --config benchmark_with_output_collection.yaml
```

## Output Locations

When `collect_outputs` is enabled, responses are stored in:

- **JSONL format**: `{report_dir}/events/` — one JSON record per line
- **SQLite format**: `{report_dir}/events.db` — queryable database

Each response is keyed by its query ID for correlation with performance metrics.

## How It Works

The `collect_outputs` flag **enables output collection for performance runs**:

| Mode          | Flag                | Outputs Collected? |
| ------------- | ------------------- | ------------------ |
| `--mode perf` | none                | ❌                 |
| `--mode perf` | `--collect-outputs` | ✅                 |
| `--mode acc`  | n/a                 | ✅                 |
| `--mode both` | n/a                 | ✅                 |

This allows performance benchmarks to optionally capture outputs without the overhead of full accuracy evaluation.

## Memory Considerations

Enabling output collection increases memory usage proportional to:

- Number of queries issued
- Average response length (tokens → bytes)

For large-scale benchmarks (e.g., 100k+ queries), consider:

- Using `--dataset perf:data.jsonl,samples=N` to limit dataset size
- Piping outputs to disk via the EventLogger (default behavior)
- Running in `--mode perf` (without collection) if storage is constrained

## Integration with Accuracy Evaluation

If you later want to run accuracy evaluation on collected outputs:

```bash
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8000 \
  --model meta-llama/Llama-2-7b-hf \
  --dataset acc:data.jsonl \
  --mode acc
```

The responses collected in the first run are independent; the accuracy run uses responses it collects during its own execution.

## Example Workflow

```bash
# 1. Performance benchmark with output collection
uv run inference-endpoint benchmark from-config \
  --config benchmark_with_output_collection.yaml \
  --report-dir results/perf_with_outputs

# 2. Inspect collected outputs (JSONL)
head results/perf_with_outputs/events.jsonl | jq .data

# 3. Query responses via SQLite
sqlite3 results/perf_with_outputs/events.db \
  "SELECT sample_uuid, data FROM event_records WHERE event_type LIKE '%complete%';"
```
