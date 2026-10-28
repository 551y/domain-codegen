"""SemTrans provider, protected-settings and API-contract regression tests."""

import json
import io
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext, redirect_stdout
from fastapi import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.semtrans_llm import ModelServiceError, OpenAICompatibleClient
from server import provider_settings
from server.semtrans_worker import _validate, run as run_semtrans_demo
import server.app as platform_app
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "semtrans"))
from SemTrans.testing_and_repair.differential_testing import _check_logic_consistency
from SemTrans.llm import create_model_completion, process_model_response


class _ChatHandler(BaseHTTPRequestHandler):
    seen = {}
    response_status = 200
    response_body = None
    response_delay_s = 0

    def do_POST(self):
        type(self).seen = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")))),
        }
        if type(self).response_delay_s:
            time.sleep(type(self).response_delay_s)
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if type(self).response_body is not None:
            body = type(self).response_body
        elif self.response_status == 200:
            body = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        else:
            body = {"error": {"message": "private upstream diagnostic"}}
        try:
            self.wfile.write(json.dumps(body).encode("utf-8"))
        except OSError:
            pass

    def log_message(self, *_args):
        return


class SemTransProviderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_openai_compatible_zcode_request_and_response(self):
        base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        client = OpenAICompatibleClient(base_url=base_url, api_key="unit-test-secret", timeout=2)
        response = client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "hello"}], max_tokens=8,
        )
        self.assertEqual(response.choices[0].message.content, "OK")
        self.assertEqual(_ChatHandler.seen["path"], "/v1/chat/completions")
        self.assertEqual(_ChatHandler.seen["authorization"], "Bearer unit-test-secret")
        self.assertEqual(_ChatHandler.seen["body"]["model"], "test-model")

    def test_codex_uses_openai_responses_api(self):
        base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        _ChatHandler.response_body = {"output_text": "OK"}
        try:
            client = OpenAICompatibleClient(base_url=base_url, api_key="unit-test-secret", provider="codex", timeout=2)
            response = client.chat.completions.create(
                model="gpt-5-codex", messages=[{"role": "user", "content": "hello"}], max_tokens=8,
            )
            self.assertEqual(response.choices[0].message.content, "OK")
            self.assertEqual(_ChatHandler.seen["path"], "/v1/responses")
            self.assertEqual(_ChatHandler.seen["body"]["input"][0]["content"], "hello")
        finally:
            _ChatHandler.response_body = None

    def test_compatible_transport_does_not_guess_streaming_from_model_name(self):
        base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        client = OpenAICompatibleClient(base_url=base_url, api_key="unit-test-secret", timeout=2)
        for model in ("qwen-compatible", "kimi-compatible", "claude-compatible"):
            with self.subTest(model=model):
                response = create_model_completion(
                    client=client, model_type=model,
                    system_prompt="system", user_prompt="user",
                )
                content = process_model_response(model_type=model, response=response, logger=None)
                self.assertEqual(content, "OK")
                self.assertFalse(_ChatHandler.seen["body"].get("stream", False))
                self.assertEqual(_ChatHandler.seen["body"]["messages"][0]["role"], "system")

    def test_provider_error_does_not_echo_response_body_or_secret(self):
        _ChatHandler.response_status = 401
        try:
            client = OpenAICompatibleClient(
                base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                api_key="unit-test-secret", timeout=2,
            )
            with self.assertRaises(ModelServiceError) as error:
                client.test_connection("test-model")
            self.assertNotIn("unit-test-secret", str(error.exception))
            self.assertNotIn("private upstream diagnostic", str(error.exception))
            self.assertIn("鉴权失败", str(error.exception))
        finally:
            _ChatHandler.response_status = 200

    def test_provider_url_rejects_plain_http_external_hosts(self):
        client = OpenAICompatibleClient(base_url="http://example.invalid/v1", api_key="unit-test-secret")
        with self.assertRaises(ModelServiceError):
            client.test_connection("test-model")

    def test_rate_limit_empty_reply_and_unexpected_shape_are_errors(self):
        base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        cases = [
            (429, None, "限流"),
            (200, {"choices": [{"message": {"content": ""}}]}, "空的"),
            (200, {"unexpected": "shape"}, "格式不符合"),
        ]
        try:
            for status, body, message in cases:
                with self.subTest(status=status, body=body):
                    _ChatHandler.response_status = status
                    _ChatHandler.response_body = body
                    client = OpenAICompatibleClient(base_url=base_url, api_key="unit-test-secret", timeout=2)
                    with self.assertRaises(ModelServiceError) as error:
                        client.test_connection("test-model")
                    self.assertIn(message, str(error.exception))
                    self.assertNotIn("unit-test-secret", str(error.exception))
        finally:
            _ChatHandler.response_status = 200
            _ChatHandler.response_body = None

    def test_provider_timeout_is_reported_without_request_details(self):
        _ChatHandler.response_delay_s = 0.2
        try:
            client = OpenAICompatibleClient(
                base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                api_key="unit-test-secret", timeout=0.02,
            )
            with self.assertRaises(ModelServiceError) as error:
                client.test_connection("test-model")
            self.assertIn("超时", str(error.exception))
            self.assertNotIn("unit-test-secret", str(error.exception))
        finally:
            _ChatHandler.response_delay_s = 0

    @unittest.skipUnless(os.name == "nt", "DPAPI encryption is Windows-specific")
    def test_top_settings_endpoint_is_the_only_shared_configuration(self):
        with tempfile.TemporaryDirectory(prefix="shared-provider-api-test-") as temp_dir:
            path = Path(temp_dir) / "model-provider-settings.json"
            names = ("HERTA_CHAT_API_KEY", "ZAI_API_KEY", "HERTA_CHAT_PROVIDER",
                     "HERTA_CHAT_BASE_URL", "HERTA_CHAT_MODEL", "DOMAINCODEGEN_API_KEY_TYPE")
            old = {name: os.environ.get(name) for name in names}
            try:
                with patch.object(provider_settings, "_settings_path", return_value=path):
                    result = platform_app.update_settings(platform_app.SettingsRequest(
                        api_key="shared-unit-secret",
                        chat_provider="openai",
                        chat_base_url="https://example.invalid/v1",
                        chat_model="shared-model",
                        api_key_type="standard",
                    ))
                    public = platform_app.get_settings_api()
                    self.assertTrue(result["api_key_configured"])
                    self.assertEqual(result["chat_model"], "shared-model")
                    self.assertEqual(result["chat_base_url"], "https://example.invalid/v1")
                    self.assertEqual(result["provider"], "codex")
                    self.assertEqual(set(result["providers"]), {"zcode", "codex", "deepseek"})
                    self.assertFalse(result["providers"]["zcode"]["api_key_configured"])
                    self.assertNotIn("api_key", result)
                    self.assertNotIn("shared-unit-secret", json.dumps(result))
                    self.assertNotIn("shared-unit-secret", json.dumps(public))
                    self.assertNotIn("shared-unit-secret", path.read_text(encoding="utf-8"))
                    self.assertFalse(hasattr(platform_app, "get_semtrans_settings_api"))
                    self.assertFalse(hasattr(platform_app, "test_semtrans_settings_api"))
            finally:
                for name, value in old.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                try:
                    from herta.config import reload_settings
                    reload_settings()
                except Exception:
                    pass

    @unittest.skipUnless(os.name == "nt", "DPAPI encryption is Windows-specific")
    def test_shared_settings_secret_is_dpapi_encrypted_and_round_trips(self):
        with tempfile.TemporaryDirectory(prefix="semtrans-settings-test-") as temp_dir:
            path = Path(temp_dir) / "model-provider-settings.json"
            with patch.object(provider_settings, "_settings_path", return_value=path):
                saved = provider_settings.save_provider_settings(
                    provider="openai",
                    base_url="https://example.invalid/v1", model="demo-model", api_key="unit-test-secret",
                    api_key_type="standard",
                )
                self.assertTrue(provider_settings.public_provider_settings(saved)["api_key_configured"])
                self.assertEqual(provider_settings.public_provider_settings(saved)["api_key_type"], "standard")
                self.assertNotIn("api_key", provider_settings.public_provider_settings(saved))
                self.assertNotIn("unit-test-secret", path.read_text(encoding="utf-8"))
                loaded = provider_settings.get_provider_settings()
                self.assertEqual(loaded["api_key"], "unit-test-secret")
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["providers"]["codex"].pop("api_key_type")
                path.write_text(json.dumps(raw), encoding="utf-8")
                self.assertEqual(provider_settings.get_provider_settings()["api_key_type"], "unclassified")
                with self.assertRaises(ValueError):
                    provider_settings.save_provider_settings(api_key_type="not-a-real-type")
                provider_settings.save_provider_settings(api_key_type="standard")
                provider_settings.save_provider_settings(clear_api_key=True)
                self.assertFalse(provider_settings.get_provider_settings()["api_key"])

    @unittest.skipUnless(os.name == "nt", "DPAPI encryption is Windows-specific")
    def test_provider_profiles_are_independent(self):
        with tempfile.TemporaryDirectory(prefix="provider-profile-test-") as temp_dir:
            path = Path(temp_dir) / "model-provider-settings.json"
            with patch.object(provider_settings, "_settings_path", return_value=path):
                provider_settings.save_provider_settings(provider="zcode", api_key="zcode-unit-secret")
                provider_settings.save_provider_settings(provider="codex", api_key="codex-unit-secret")
                provider_settings.save_provider_settings(provider="deepseek", api_key="deepseek-unit-secret")
                current = provider_settings.get_provider_settings()
                self.assertEqual(current["provider"], "deepseek")
                self.assertEqual(current["api_key"], "deepseek-unit-secret")
                self.assertEqual(current["profiles"]["zcode"]["api_key"], "zcode-unit-secret")
                self.assertEqual(current["profiles"]["codex"]["api_key"], "codex-unit-secret")
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn("zcode-unit-secret", raw)
                self.assertNotIn("codex-unit-secret", raw)
                self.assertNotIn("deepseek-unit-secret", raw)

    @unittest.skipUnless(os.name == "nt", "DPAPI encryption is Windows-specific")
    def test_coding_plan_key_is_blocked_before_any_network_request(self):
        with tempfile.TemporaryDirectory(prefix="semtrans-plan-policy-test-") as temp_dir:
            path = Path(temp_dir) / "model-provider-settings.json"
            with patch.object(provider_settings, "_settings_path", return_value=path):
                saved = provider_settings.save_provider_settings(
                    api_key="unit-test-plan-secret", api_key_type="coding_plan",
                )
                environment = {
                    name: os.environ.get(name)
                    for name in ("HERTA_CHAT_API_KEY", "ZAI_API_KEY", "HERTA_CHAT_PROVIDER",
                                 "HERTA_CHAT_BASE_URL", "HERTA_CHAT_MODEL", "DOMAINCODEGEN_API_KEY_TYPE")
                }
                try:
                    os.environ["HERTA_CHAT_API_KEY"] = saved["api_key"]
                    os.environ["ZAI_API_KEY"] = saved["api_key"]
                    os.environ["HERTA_CHAT_PROVIDER"] = saved["provider"]
                    os.environ["HERTA_CHAT_BASE_URL"] = saved["base_url"]
                    os.environ["HERTA_CHAT_MODEL"] = saved["model"]
                    os.environ["DOMAINCODEGEN_API_KEY_TYPE"] = saved["api_key_type"]
                    from herta.config import reload_settings
                    reload_settings()
                    with patch.object(platform_app, "OpenAICompatibleClient") as client:
                        with self.assertRaises(HTTPException) as error:
                            platform_app.test_model_settings_api()
                        self.assertEqual(error.exception.status_code, 400)
                        self.assertIn("官方支持工具", error.exception.detail)
                        client.assert_not_called()

                    task_id = platform_app._task_create("paddle")
                    try:
                        request = platform_app.GenerateRequest(
                            kb="paddle", source="semtrans/alexnet", mode="sample",
                        )
                        with patch.object(platform_app, "OpenAICompatibleClient") as client:
                            platform_app._run_semtrans(task_id, request)
                            client.assert_not_called()
                        self.assertEqual(platform_app.tasks[task_id]["status"], "failed")
                        self.assertIn("不能在此直连套餐 Key", platform_app.tasks[task_id]["error"])
                        self.assertNotIn("unit-test-plan-secret", platform_app.tasks[task_id]["error"])
                    finally:
                        with platform_app._tasks_lock:
                            platform_app.tasks.pop(task_id, None)
                finally:
                    for name, value in environment.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                    try:
                        reload_settings()
                    except Exception:
                        pass


class _FakeNetwork:
    def __init__(self, outputs):
        self.outputs = outputs
        self.model = self

    def __call__(self, *_inputs):
        return self.outputs


class _FakeSymbolNet:
    def __init__(self, outputs):
        self.model = _FakeNetwork(outputs)

    def grad_search(self, *_args):
        return self

    def apply_weights(self, *_args):
        return None

    def to_tensor(self, value):
        return value

    def to_numpy(self, value):
        return value

    def stop_grad(self):
        return nullcontext()

    def clear_cache(self):
        return None


class SemTransNumericalEvidenceChecks(unittest.TestCase):
    def test_every_output_is_checked_and_reported(self):
        source = _FakeSymbolNet({
            "main": np.array([1.0, 2.0], dtype=np.float32),
            "aux": np.array([3.0, 4.0], dtype=np.float32),
        })
        target = _FakeSymbolNet({
            "main": np.array([1.0, 2.0], dtype=np.float32),
            "aux": np.array([3.0, 4.2], dtype=np.float32),
        })
        metrics = []
        issues = _check_logic_consistency(
            source, target, [np.array([0.5], dtype=np.float32)], {},
            rtol=1e-4, atol=1e-4, logger=logging.getLogger(__name__),
            case_idx=0, numerical_checks=metrics,
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(metrics[0]["outputs"]), 2)
        self.assertEqual(metrics[0]["outputs"][0]["failed_elements"], 0)
        self.assertGreater(metrics[0]["outputs"][1]["failed_elements"], 0)
        self.assertEqual(issues[0]["error_type"], "output_value_mismatch")
        self.assertIn("failed elements", issues[0]["error_message"])

    def test_multiple_matching_outputs_pass(self):
        outputs = {
            "main": np.array([1.0, 2.0], dtype=np.float32),
            "aux": np.array([3.0, 4.0], dtype=np.float32),
        }
        metrics = []
        issues = _check_logic_consistency(
            _FakeSymbolNet(outputs), _FakeSymbolNet(outputs),
            [np.array([0.5], dtype=np.float32)], {}, rtol=1e-4, atol=1e-4,
            logger=logging.getLogger(__name__), case_idx=1,
            numerical_checks=metrics,
        )
        self.assertEqual(issues, [])
        self.assertTrue(metrics[0]["passed"])
        self.assertEqual(metrics[0]["outputs"][1]["shape"], [2])


class SemTransFailClosedChecks(unittest.TestCase):
    def test_validator_never_marks_zero_sample_run_as_passed(self):
        api = [None] * 9
        api[5] = lambda **_kwargs: (True, None)
        api[8] = logging.getLogger(__name__)
        with patch("server.semtrans_worker._instantiate", return_value=(object(), object())):
            report, issues = _validate(
                Path("unused-source.py"), Path("unused-target.py"),
                {"class_name": "EmptyCase", "input_shape": [1, 1]},
                {"max_sample": 0, "search_budget_ms": 1, "target_count": 1},
                api, seed=7, phase="zero-sample-test", rtol=1e-4, atol=1e-4,
            )
        self.assertEqual(report["status"], "unverified")
        self.assertFalse(report["verified"])
        self.assertEqual(report["valid_paired_cases"], 0)
        self.assertIn("valid paired test input", report["reason"])
        self.assertEqual(issues, [])


class SemTransTupleOutputChecks(unittest.TestCase):
    def test_single_graph_name_preserves_tuple_outputs_for_getitem(self):
        """torch.chunk must keep both branches for the next getitem node."""
        first = np.zeros((1, 2), dtype=np.float32)
        second = np.ones((1, 2), dtype=np.float32)
        cache = {}
        instruction = SimpleNamespace(retvals=lambda: ["chunk"])
        from SemTrans.math_symbol_base.symbolic_system.symbolic_evaluator import SemanticTracer

        SemanticTracer._store_instruction_outputs(cache, instruction, (first, second))
        self.assertIsInstance(cache["chunk"], tuple)
        self.assertTrue(np.array_equal(cache["chunk"][0], first))
        self.assertTrue(np.array_equal(cache["chunk"][1], second))


class SemTransWorkerOrchestrationChecks(unittest.TestCase):
    def test_real_bundled_model_profiles_load_with_semtrans_frontend(self):
        root = Path(__file__).resolve().parents[2]
        cases = json.loads((root / "server" / "semtrans_demo_cases.json").read_text(encoding="utf-8"))
        from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
        from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
        import torch

        observed = {}
        for name, profile in cases.items():
            source = (root / profile["source_path"]).resolve()
            model = load_model_instance(
                str(source), profile["class_name"], "cpu", init_kwargs=profile.get("init_kwargs", {})
            )
            model.eval()
            output = model(torch.zeros(profile["input_shape"], dtype=torch.float32))
            symbolnet = create_symbolnet(
                model,
                use_gradient=False,
                input_map={"x": AbsTensor(list(profile["input_shape"]), profile.get("input_dtype", "float32"))},
                framework_type="pytorch",
            )
            symbolnet.to("cpu")
            self.assertTrue(source.is_file())
            self.assertTrue(hasattr(output, "shape"))
            observed[name] = list(output.shape)
        self.assertEqual(observed, {
            "alexnet": [1, 1000],
            "densenet": [1, 1000],
            "shufflenetv2": [1, 1000],
            "squeezenet": [1, 1000],
            "mnasnet": [1, 1000],
        })

    def test_worker_failure_marks_inflight_stage_failed_and_redacts_key(self):
        root = Path(__file__).resolve().parents[2]
        secret = "worker-orchestration-unit-secret"

        def fail_after_generation_started(*, on_event, **_kwargs):
            on_event({"name": "LLM 生成 Paddle 候选", "status": "running"})
            raise platform_app.SemTransWorkerError(f"upstream timeout {secret}")

        with tempfile.TemporaryDirectory(prefix="semtrans-failed-stage-") as temp_dir:
            task_id = platform_app._task_create("paddle")
            request = platform_app.GenerateRequest(
                kb="paddle", source="semtrans/alexnet", mode="sample",
            )
            settings = {
                "provider": "zcode", "api_key": secret, "api_key_type": "standard",
                "base_url": "https://example.invalid/v1", "model": "unit-test-model",
                "profiles": {},
            }
            try:
                with patch.object(platform_app, "get_provider_settings", return_value=settings), \
                     patch.object(platform_app, "GEN_DIR", Path(temp_dir)), \
                     patch.object(platform_app, "run_semtrans_worker", side_effect=fail_after_generation_started):
                    platform_app._run_semtrans(task_id, request)

                task = platform_app.tasks[task_id]
                self.assertEqual(task["status"], "failed")
                self.assertNotIn(secret, json.dumps(task, ensure_ascii=False))
                steps = task["intermediate"]["evaluation_steps"]
                self.assertEqual(steps[-1]["name"], "LLM 生成 Paddle 候选")
                self.assertEqual(steps[-1]["status"], "failed")
                self.assertNotIn(secret, steps[-1]["detail"])
            finally:
                with platform_app._tasks_lock:
                    platform_app.tasks.pop(task_id, None)

    def test_real_semtrans_finds_and_repairs_controlled_avgpool_fault(self):
        root = Path(__file__).resolve().parents[2]
        fixture_root = root / "server" / "tests" / "fixtures"
        cases = json.loads((fixture_root / "semtrans_cases.json").read_text(encoding="utf-8"))
        profile = cases["tiny_avgpool"]
        profile["fault_injection"] = {
            "label": "AvgPool padding divisor mismatch (controlled test fault)",
            "mutation": "Paddle AvgPool2D exclusive=True → exclusive=False",
            "pattern": r"(?i)\bexclusive\s*=\s*True\b",
            "replacement": "exclusive=False",
        }
        source_path = fixture_root / profile["source_file"]
        target_code = '''import paddle
import paddle.nn as nn

class TinyAvgPool(nn.Layer):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=True)

    def forward(self, x):
        return self.pool(x)
'''

        class _MockModelService:
            calls = 0

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **_kwargs):
                type(self).calls += 1
                message = SimpleNamespace(content=target_code)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        request = {
            "source_code": source_path.read_text(encoding="utf-8"),
            "source_label": profile["label"],
            "profile": profile,
            "config": {
                "max_repairs": 1,
                "max_sample": 1,
                "target_count": 2,
                "search_budget_ms": 600,
                "atol": 1e-4,
                "rtol": 1e-4,
                "seed": 20261028,
            },
            "model_service": {
                "base_url": "http://127.0.0.1/v1",
                "model": "unit-test-model",
                "api_key": "unit-test-secret",
            },
        }
        with tempfile.TemporaryDirectory(prefix="semtrans-worker-test-") as task_dir:
            request["output_dir"] = task_dir
            with patch.dict(sys.modules, {"semtrans_llm": SimpleNamespace(OpenAICompatibleClient=_MockModelService)}):
                with redirect_stdout(io.StringIO()):
                    result = run_semtrans_demo(request)

        self.assertGreaterEqual(_MockModelService.calls, 2)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["verified"])
        self.assertEqual(result["repair_rounds"], 1)
        rounds = result["validation_rounds"]
        self.assertEqual([row["status"] for row in rounds], ["failed", "passed", "passed"])
        self.assertEqual(rounds[0]["issues"][0]["cluster_key"], "semantic_mismatch")
        self.assertIsNotNone(rounds[0]["semantic_evidence"]["first_mismatch"])
        self.assertIn("region_rewrite_requires_formula_review", rounds[0]["issues"][0]["prompt_content"])
        self.assertEqual(rounds[-1]["phase"], "held_out_1")
        self.assertGreater(rounds[-1]["valid_paired_cases"], 0)
        self.assertNotIn("unit-test-secret", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
