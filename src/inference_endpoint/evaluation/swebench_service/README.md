# SWE-bench Service

Runs mini-swe-agent and the SWE-bench harness on a host with Docker. The
benchmark client only needs this service URL.

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080
```

The endpoint URLs in the benchmark config must be reachable from the service
host. Docker is required only on the service host.
