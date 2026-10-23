"""Regression checks for ModelX multi-file artifact delivery and failure reporting.

Run with a Paddle/CMC environment:
    python -m unittest discover -s server/tests -p test_paddle_artifacts.py -v
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server import app
from server.paddle_converter import (
    ConversionResult, convert_pytorch_source, paddle_demo_cases, run_paddle_smoke_test,
)


class PaddleArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="modelx-artifact-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        gen_patch = patch.object(app, "GEN_DIR", self.root / "generated")
        gen_patch.start()
        self.addCleanup(gen_patch.stop)
        self.main = self.root / "densenet_paddle.py"
        self.main.write_text((app.PADDLE_PRESET_DIR / "densenet_to_paddle.py").read_text(encoding="utf-8"), encoding="utf-8")
        self.profile = paddle_demo_cases()["densenet"]

    def copy_helper(self):
        text = (app.PADDLE_PRESET_DIR / "densenet" / "modified_KaimingNormal.py").read_text(encoding="utf-8")
        (self.root / "modified_KaimingNormal.py").write_text(text, encoding="utf-8")
        return text

    def test_displayed_bundle_runs_with_relative_imports(self):
        self.copy_helper()
        result = run_paddle_smoke_test(self.main, profile=self.profile)
        self.assertTrue(result.get("import_ok"), result)
        self.assertTrue(result.get("forward_ok"), result)
        self.assertTrue(result.get("finite"), result)
        self.assertEqual(result.get("class_name"), "DenseNet")
        self.assertEqual(result.get("output_shape"), [1, 1000])

    def test_missing_helper_cannot_be_reported_as_pass(self):
        result = run_paddle_smoke_test(self.main, profile=self.profile)
        self.assertFalse(result.get("import_ok"), result)
        self.assertFalse(result.get("forward_ok"), result)
        self.assertIn("ModuleNotFoundError", result.get("error", ""))

    def test_invalid_helper_fails_static_check(self):
        code = self.main.read_text(encoding="utf-8")
        evaluation = app._eval_paddle(code, {"total_calls": 1}, [], companion_files={
            "modified_KaimingNormal.py": "def broken(:\n",
        })
        self.assertFalse(evaluation["checks"]["伴随 Python 模块语法合法"])

    def test_converter_bug_does_not_fall_back_to_preset(self):
        task_id = app._task_create("paddle")
        with patch.object(app, "convert_pytorch_source", side_effect=RuntimeError("conversion regression")):
            app._run_paddle(task_id, app.GenerateRequest(kb="paddle", source="pytorch/vision/densenet"))
        self.assertEqual(app.tasks[task_id]["status"], "failed")
        self.assertIn("conversion regression", app.tasks[task_id]["error"])

    def fake_task(self, runtime, failed_api_count=0):
        code = "import paddle\nclass Tiny(paddle.nn.Layer):\n    def forward(self, x): return x\n"
        conversion = ConversionResult(code, "tiny.py", success_api_count=1,
                                      failed_api_count=failed_api_count)
        task_id = app._task_create("paddle")
        self.addCleanup(app.tasks.pop, task_id, None)
        with patch.object(app, "convert_pytorch_source", return_value=conversion), \
                patch.object(app, "run_paddle_smoke_test", return_value=runtime):
            app._run_paddle(task_id, app.GenerateRequest(kb="paddle", source="pytorch/vision/alexnet"))
        self.assertEqual(app.tasks[task_id]["status"], "completed", app.tasks[task_id])
        return app.tasks[task_id]

    def test_constructor_error_is_failed_not_skipped(self):
        code = (
            "import paddle\nclass Tiny(paddle.nn.Layer):\n"
            "    def __init__(self):\n        super().__init__()\n"
            "        raise RuntimeError('constructor regression')\n"
            "    def forward(self, x): return x\n"
        )
        self.main.write_text(code, encoding="utf-8")
        runtime = run_paddle_smoke_test(self.main, profile={"class_name": "Tiny"})
        self.assertTrue(runtime.get("import_ok"), runtime)
        self.assertTrue(runtime.get("construction_attempted"), runtime)
        self.assertFalse(runtime.get("construction_ok"), runtime)
        self.assertFalse(runtime.get("forward_attempted"), runtime)
        self.assertIn("constructor regression", runtime.get("error", ""))
        task = self.fake_task(runtime)
        self.assertEqual(task["intermediate"]["evaluation_steps"][1]["status"], "failed")
        self.assertFalse(task["result"]["evaluation"]["checks"]["Paddle 模型构造成功"])

    def test_runtime_error_always_has_a_failed_check(self):
        for attempted in (False, True):
            with self.subTest(forward_attempted=attempted):
                runtime = {"attempted": True, "import_ok": True,
                           "forward_attempted": attempted, "forward_ok": False,
                           "error": "runtime regression"}
                task = self.fake_task(runtime)
                self.assertEqual(task["intermediate"]["evaluation_steps"][1]["status"], "failed")
                self.assertFalse(all(task["result"]["evaluation"]["checks"].values()))

    def test_partial_conversion_not_shown_as_fully_passed(self):
        task = self.fake_task({"attempted": False, "skip_reason": "Paddle not installed"},
                              failed_api_count=1)
        self.assertEqual(task["intermediate"]["evaluation_steps"][0]["status"], "failed")
        self.assertFalse(task["result"]["evaluation"]["checks"]["ModelX API 全部转换"])

    def test_unavailable_runtime_is_explicitly_skipped(self):
        task = self.fake_task({"attempted": False, "available": False,
                              "skip_reason": "Paddle not installed"})
        self.assertEqual(task["intermediate"]["evaluation_steps"][1]["status"], "skipped")
        self.assertIn("Paddle not installed", task["result"]["evaluation"]["note"])
        self.assertNotIn("Paddle dummy 前向成功", task["result"]["evaluation"]["checks"])

    def test_conversion_does_not_leak_task_import_paths_or_disable_logging(self):
        # Paddle adds its own native-library path on first import; that is not
        # a ModelX task leak and must remain available to its runtime.
        import paddle  # noqa: F401
        before_paths = list(sys.path)
        before_logging = logging.root.manager.disable
        source = app.OPALIGN_MODELS / "pytorch" / "vision" / "sourceModels" / "alexnet.py"
        conversion = convert_pytorch_source(source, self.root / "conversion")
        self.assertTrue(conversion.code)
        self.assertEqual(sys.path, before_paths)
        self.assertEqual(logging.root.manager.disable, before_logging)

    def test_dependency_fallback_preserves_companions_and_marks_replay(self):
        task_id = app._task_create("paddle")
        self.addCleanup(app.tasks.pop, task_id, None)
        with patch.object(app, "convert_pytorch_source", side_effect=ImportError("dependency unavailable")):
            app._run_paddle(task_id, app.GenerateRequest(kb="paddle", source="pytorch/vision/densenet"))
        task = app.tasks[task_id]
        self.assertEqual(task["status"], "completed", task)
        result = task["result"]
        self.assertTrue(result["pre_generated"])
        self.assertIn("modified_KaimingNormal.py", {f["name"] for f in result["files"]})
        self.assertTrue(result["evaluation"]["paddle_runtime"].get("forward_ok"), result)
        self.assertEqual(task["intermediate"]["evaluation_steps"][0]["status"], "skipped")

    def test_nonfinite_and_wrong_shape_outputs_cannot_pass(self):
        for value, expected, error in (("float('nan')", [1, 1000], "NaN"),
                                       ("1.0", [1, 10], "expected")):
            with self.subTest(value=value, expected=expected):
                self.main.write_text(
                    "import paddle\nclass Tiny(paddle.nn.Layer):\n"
                    f"    def forward(self, x): return paddle.full([1, 1000], {value})\n",
                    encoding="utf-8",
                )
                runtime = run_paddle_smoke_test(self.main, profile={
                    "class_name": "Tiny", "expected_output_shape": expected,
                })
                self.assertTrue(runtime.get("forward_attempted"), runtime)
                self.assertFalse(runtime.get("forward_ok"), runtime)
                self.assertIn(error, runtime.get("error", ""))

    def test_missing_model_profile_is_an_explicit_skip_not_a_pass(self):
        self.main.write_text("import paddle\n__all__ = []\n", encoding="utf-8")
        runtime = run_paddle_smoke_test(self.main)
        self.assertTrue(runtime.get("import_ok"), runtime)
        self.assertFalse(runtime.get("forward_ok"), runtime)
        self.assertTrue(runtime.get("skip_reason"), runtime)
        self.assertIn("elapsed_s", runtime)
        task = self.fake_task(runtime)
        self.assertEqual(task["intermediate"]["evaluation_steps"][1]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
