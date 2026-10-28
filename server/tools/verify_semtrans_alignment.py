"""Exercise the real SemTrans PyTorch/Paddle frontends and region validator.

Run with a Python environment containing PyTorch, Paddle and SemTrans runtime
dependencies, for example:
    python server/tools/verify_semtrans_alignment.py

The test does not call an LLM. It supplies deterministic target modules to
test several real semantic pass paths and a runnable AvgPool divisor mismatch.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "semtrans"))
from server.semtrans_worker import _validate


GOOD_TARGET = '''import paddle
import paddle.nn as nn

class TinyCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2D((1, 1))

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))
'''

BAD_POOL_TARGET = '''import paddle
import paddle.nn as nn

class TinyAvgPool(nn.Layer):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)

    def forward(self, x):
        return self.pool(x)
'''

GOOD_MLP_TARGET = '''import paddle
import paddle.nn as nn

class TinyMLP(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
'''

GOOD_CONV_CLASSIFIER_TARGET = '''import paddle
import paddle.nn as nn

class TinyConvClassifier(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))
'''

GOOD_BN_TARGET = '''import paddle
import paddle.nn as nn

class TinyBN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(1, 2, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2D(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
'''

GOOD_CONV_STACK_TARGET = '''import paddle
import paddle.nn as nn

class TinyConvStack(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(2, 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2D((1, 1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        return self.pool(self.relu2(self.conv2(x)))
'''

GOOD_MAXPOOL_TARGET = '''import paddle
import paddle.nn as nn

class TinyMaxPool(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.adaptive = nn.AdaptiveAvgPool2D((1, 1))

    def forward(self, x):
        return self.adaptive(self.pool(self.relu(self.conv(x))))
'''

GOOD_AVGPOOL_PASS_TARGET = '''import paddle
import paddle.nn as nn

class TinyAvgPoolPass(nn.Layer):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)

    def forward(self, x):
        return self.pool(x)
'''


def main() -> None:
    fixture_root = ROOT / "server" / "tests" / "fixtures"
    cases = json.loads((fixture_root / "semtrans_cases.json").read_text(encoding="utf-8"))
    config = {"bounded_demo": True, "max_sample": 1, "target_count": 2,
              "max_time_ms": 600, "seed": 20261027}
    results = []
    with tempfile.TemporaryDirectory(prefix="semtrans-align-") as directory:
        task_root = Path(directory)
        target_modules = {
            "tiny_cnn": GOOD_TARGET,
            "tiny_avgpool": BAD_POOL_TARGET,
        }
        for name, target_code in target_modules.items():
            profile = cases[name]
            source = fixture_root / profile["source_file"]
            target = task_root / f"{name}_target.py"
            target.write_text(target_code, encoding="utf-8")
            report, issues = _validate(
                source, target, profile, config, _semtrans_api(),
                seed=20261027, phase=name, rtol=1e-4, atol=1e-4,
            )
            expected = "failed" if name in {"tiny_avgpool", "tiny_mlp", "tiny_conv_classifier", "tiny_bn"} else "passed"
            semantic_evidence = report.get("semantic_evidence", {})
            numerical_checks = report.get("numerical_checks", [])
            passed = report["status"] == expected and report["valid_paired_cases"] > 0
            results.append({"case": name, "passed": passed, "expected": expected,
                            "status": report["status"], "valid_paired_cases": report["valid_paired_cases"],
                            "elapsed_s": report["elapsed_s"],
                            "alignment_records": len(semantic_evidence.get("alignment_records", [])),
                            "parameter_alignment_records": len(semantic_evidence.get("parameter_alignment_records", [])),
                            "numerical_checks": numerical_checks,
                            "issues": issues})
    print(json.dumps({"passed": all(row["passed"] for row in results), "cases": results}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)


def _semtrans_api():
    from SemTrans.enhanced_conversion.framework_knowledge_retrivals import enhanced_conversion_agent
    from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
    from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
    from SemTrans.testing_and_repair.constraint_driven_repair import (
        assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix, get_repair_system_prompt,
    )
    from SemTrans.testing_and_repair.logging import CONVERSION_LOG
    import logging

    CONVERSION_LOG.handlers.clear()
    CONVERSION_LOG.addHandler(logging.NullHandler())
    CONVERSION_LOG.setLevel(logging.CRITICAL)
    return (enhanced_conversion_agent, AbsTensor, create_symbolnet, load_model_instance,
            assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix,
            get_repair_system_prompt, CONVERSION_LOG)


if __name__ == "__main__":
    main()
