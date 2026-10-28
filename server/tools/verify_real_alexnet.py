"""Run SemTrans differential validation on a real bundled AlexNet pair.

This is intentionally deterministic and does not call an LLM.  It exercises
the framework frontends plus cross-framework parameter injection, including
PyTorch OI versus Paddle IO Linear weights.
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


TARGET = '''import paddle
import paddle.nn as nn

class AlexNet(nn.Layer):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(), nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(4096, 4096),
            nn.ReLU(), nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        return self.classifier(x)
'''


def _api():
    from SemTrans.enhanced_conversion.framework_knowledge_retrivals import enhanced_conversion_agent
    from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
    from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
    from SemTrans.testing_and_repair.constraint_driven_repair import (
        assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix,
        get_repair_system_prompt,
    )
    from SemTrans.testing_and_repair.logging import CONVERSION_LOG
    import logging
    CONVERSION_LOG.handlers.clear()
    CONVERSION_LOG.addHandler(logging.NullHandler())
    CONVERSION_LOG.setLevel(logging.CRITICAL)
    return (enhanced_conversion_agent, AbsTensor, create_symbolnet, load_model_instance,
            assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix,
            get_repair_system_prompt, CONVERSION_LOG)


def main():
    cases = json.loads((ROOT / "server" / "semtrans_demo_cases.json").read_text(encoding="utf-8"))
    profile = cases["alexnet"]
    source = (ROOT / profile["source_path"]).resolve()
    config = {"bounded_demo": True, "max_sample": 1, "target_count": 1,
              "max_time_ms": 1800, "seed": 20261028}
    with tempfile.TemporaryDirectory(prefix="semtrans-real-alexnet-") as directory:
        target = Path(directory) / "alexnet_target.py"
        target.write_text(TARGET, encoding="utf-8")
        # Capture the parameter-balancing decision for the regression case.
        # This keeps the check useful when a frontend changes its native
        # parameter names/layouts again.
        from SemTrans.math_symbol_base.symbolic_system.symbolic_evaluator import SemanticTracer
        original_balance = SemanticTracer._balance_and_inject
        def traced_balance(self, src_inv, tgt_reqs):
            result = original_balance(self, src_inv, tgt_reqs)
            if "B" in tgt_reqs:
                print("[AlexNet regression] B source entries:",
                      [(row.get("op_key"), list(row["value"].shape)) for row in src_inv.get("B", [])])
                print("[AlexNet regression] B target slots:",
                      [(type(obj).__name__, attr) for obj, attr in tgt_reqs["B"]])
            return result
        SemanticTracer._balance_and_inject = traced_balance
        report, issues = _validate(source, target, profile, config, _api(),
                                   seed=20261028, phase="alexnet-linear-regression",
                                   rtol=1e-4, atol=1e-4)
    print(json.dumps({"report": report, "issues": issues}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()
