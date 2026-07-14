# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for service-backed SWEBenchScorer."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
import yaml
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import Scorer, SWEBenchScorer
from inference_endpoint.exceptions import SetupError

pytestmark = pytest.mark.unit

_DATASET_NAME = "swe_bench"
_MODEL_NAME = "TestOrg/test-model-7b"


def _write_benchmark_config(report_dir: Path, model_params: dict | None = None) -> None:
    mp: dict = {"name": _MODEL_NAME}
    if model_params is not None:
        mp.update(model_params)
    (report_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "model_params": mp,
                "endpoint_config": {
                    "endpoints": ["http://endpoint-host:30000"],
                    "api_key": "secret-key",
                },
            }
        )
    )


def _write_sample_idx_map(report_dir: Path, n: int = 3) -> None:
    idx_map = {_DATASET_NAME: {f"uuid-{i}": i for i in range(n)}}
    (report_dir / "sample_idx_map.json").write_bytes(msgspec.json.encode(idx_map))


def _make_dataset(n: int = 3) -> MagicMock:
    ds = MagicMock()
    ds.dataframe = pd.DataFrame(
        {
            "instance_id": [f"repo__repo-{i}" for i in range(n)],
            "prompt": ["placeholder"] * n,
        }
    )
    ds.num_samples.return_value = n
    return ds


def _make_scorer(report_dir: Path, *, dataset=None, **kwargs) -> SWEBenchScorer:
    return SWEBenchScorer(
        dataset_name=_DATASET_NAME,
        dataset=dataset or _make_dataset(),
        report_dir=report_dir,
        swebench_service_url="http://service-host:18080",
        poll_interval_s=0.01,
        **kwargs,
    )


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "report"
    d.mkdir()
    _write_benchmark_config(d)
    _write_sample_idx_map(d)
    return d


class TestSWEBenchScorerRegistration:
    def test_registered(self):
        assert "swe_bench_scorer" in Scorer.PREDEFINED
        assert Scorer.get("swe_bench_scorer") is SWEBenchScorer

    def test_skip_endpoint_phase(self):
        assert SWEBenchScorer.SKIP_ENDPOINT_PHASE is True

    def test_external_sample_count(self):
        assert SWEBenchScorer.external_sample_count({"num_instances": 50}) == 50
        assert (
            SWEBenchScorer.external_sample_count({})
            == SWEBenchScorer.DEFAULT_NUM_INSTANCES
        )
        assert SWEBenchScorer.external_sample_count({"num_instances": "bad"}) is None
        assert SWEBenchScorer.external_sample_count({"num_instances": 0}) is None


class TestSWEBenchScorerPreflight:
    def test_preflight_calls_health(self, monkeypatch):
        calls: list[tuple[str, str]] = []

        def fake_http_json(url, *, method="GET", **kwargs):
            calls.append((url, method))
            return {
                "api_version": "v1",
                "capabilities": ["swebench.run", "artifacts.download"],
            }

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)

        SWEBenchScorer.preflight(
            {
                "swebench_service_url": "http://service-host:18080",
                "subset": "lite",
                "num_instances": 1,
            }
        )

        assert calls == [("http://service-host:18080/health", "GET")]

    def test_preflight_never_calls_docker_or_subprocess(self, monkeypatch):
        def fail_run(*args, **kwargs):
            pytest.fail("client preflight must not run local subprocesses")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fail_run)
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "api_version": "v1",
                    "capabilities": ["swebench.run", "artifacts.download"],
                }
            ),
        )

        SWEBenchScorer.preflight({"swebench_service_url": "http://service-host:18080"})

    def test_missing_service_url_raises(self):
        with pytest.raises(SetupError, match="swebench_service_url is required"):
            SWEBenchScorer.preflight({})

    def test_capability_mismatch_raises(self, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {"api_version": "v1", "capabilities": []}
            ),
        )

        with pytest.raises(SetupError, match="missing required capabilities"):
            SWEBenchScorer.preflight({"swebench_service_url": "http://service-host"})

    def test_api_version_mismatch_raises(self, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "api_version": "v2",
                    "capabilities": ["swebench.run", "artifacts.download"],
                }
            ),
        )

        with pytest.raises(SetupError, match="API version mismatch"):
            SWEBenchScorer.preflight({"swebench_service_url": "http://service-host"})


class TestSWEBenchScorerScore:
    def test_score_submits_exact_instance_ids_and_computes_rate(
        self, report_dir, monkeypatch
    ):
        payloads: list[dict] = []

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            if method == "POST":
                payloads.append(payload)
                return {
                    "run_id": "run-1",
                    "status": "succeeded",
                    "result": {"resolved_instances": 2, "submitted_instances": 3},
                    "artifacts": [],
                }
            raise AssertionError(f"unexpected GET {url}")

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        scorer = _make_scorer(report_dir)

        score, n_repeats = scorer.score()

        assert score == pytest.approx(2 / 3)
        assert n_repeats == 1
        assert scorer.complete is True
        assert payloads[0]["evaluated_instance_ids"] == [
            "repo__repo-0",
            "repo__repo-1",
            "repo__repo-2",
        ]
        assert (
            payloads[0]["benchmark_config"]["endpoint_config"]["api_key"]
            == "secret-key"
        )
        assert (report_dir / "swe_bench_results.json").exists()

    def test_score_polls_until_terminal(self, report_dir, monkeypatch):
        calls: list[str] = []

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            calls.append(url)
            if method == "POST":
                return {"run_id": "run-1", "status": "queued"}
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {"resolved_instances": 1, "submitted_instances": 3},
                "artifacts": [],
            }

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(1 / 3)
        assert calls[-1] == "http://service-host:18080/v1/runs/run-1"

    def test_service_failure_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "run_id": "run-1",
                    "status": "failed",
                    "error": "boom",
                }
            ),
        )
        scorer = _make_scorer(report_dir)

        score, n_repeats = scorer.score()

        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_missing_result_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {"run_id": "run-1", "status": "succeeded"}
            ),
        )
        scorer = _make_scorer(report_dir)

        score, _ = scorer.score()

        assert score is None
        assert scorer.complete is False

    def test_zero_denominator_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(SWEBenchScorer, "_http_json", MagicMock())
        scorer = _make_scorer(report_dir, dataset=_make_dataset(n=0))

        score, _ = scorer.score()

        assert score is None
        assert scorer.complete is False
        SWEBenchScorer._http_json.assert_not_called()

    def test_artifact_result_fallback(self, report_dir, monkeypatch):
        result_path = report_dir / "swe_bench_results.json"

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "artifacts": [{"name": "swe_bench_results.json", "url": "/artifact"}],
            }

        def fake_download(service_url, status, output_dir):
            result_path.write_text(
                json.dumps({"resolved_instances": 1, "submitted_instances": 3})
            )

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(SWEBenchScorer, "_download_artifacts", fake_download)

        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(1 / 3)
