"""Deterministic SemTrans regression for all bundled real vision models.

The target snippets are executable Paddle implementations of the bundled
PyTorch models (not synthetic fixtures).  This test deliberately bypasses the
LLM so frontend/MathIR/alignment regressions can be reproduced offline.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "semtrans")]
from server.semtrans_worker import _validate


DENSENET = r'''import paddle
import paddle.nn as nn

class _DenseLayer(nn.Layer):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2D(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2D(num_input_features, bn_size * growth_rate, 1, 1, 0, bias_attr=False)
        self.norm2 = nn.BatchNorm2D(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2D(bn_size * growth_rate, growth_rate, 3, 1, 1, bias_attr=False)
        self.drop_rate = float(drop_rate)
    def forward(self, inputs):
        features = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        x = self.conv1(self.relu1(self.norm1(paddle.concat(features, axis=1))))
        x = self.conv2(self.relu2(self.norm2(x)))
        if self.drop_rate > 0:
            x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        return x

class _DenseBlock(nn.Layer):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super().__init__()
        self.layers = nn.LayerList([
            _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, memory_efficient)
            for i in range(num_layers)
        ])
    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            features.append(layer(features))
        return paddle.concat(features, axis=1)

class _Transition(nn.Layer):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2D(num_input_features)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2D(num_input_features, num_output_features, 1, 1, 0, bias_attr=False)
        self.pool = nn.AvgPool2D(2, 2)
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))

class DenseNet(nn.Layer):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super().__init__()
        layers = [nn.Conv2D(3, num_init_features, 7, 2, 3, bias_attr=False),
                  nn.BatchNorm2D(num_init_features), nn.ReLU(), nn.MaxPool2D(3, 2, 1)]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            layers.append(_DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, memory_efficient))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                layers.append(_Transition(num_features, num_features // 2))
                num_features //= 2
        layers.append(nn.BatchNorm2D(num_features))
        self.features = nn.LayerList(layers)
        self.classifier = nn.Linear(num_features, num_classes)
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = paddle.nn.functional.relu(x)
        x = paddle.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = paddle.flatten(x, 1)
        return self.classifier(x)
'''


SHUFFLENET = r'''import paddle
import paddle.nn as nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = paddle.reshape(x, [batchsize, groups, channels_per_group, height, width])
    x = paddle.transpose(x, [0, 2, 1, 3, 4])
    return paddle.reshape(x, [batchsize, num_channels, height, width])

class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=False), nn.BatchNorm2D(inp),
                nn.Conv2D(inp, branch_features, 1, 1, 0, bias_attr=False), nn.BatchNorm2D(branch_features), nn.ReLU())
        else:
            self.branch1 = None
        in2 = inp if stride > 1 else branch_features
        self.branch2 = nn.Sequential(
            nn.Conv2D(in2, branch_features, 1, 1, 0, bias_attr=False), nn.BatchNorm2D(branch_features), nn.ReLU(),
            nn.Conv2D(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias_attr=False),
            nn.BatchNorm2D(branch_features), nn.Conv2D(branch_features, branch_features, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(branch_features), nn.ReLU())
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = paddle.chunk(x, chunks=2, axis=1)
            out = paddle.concat([x1, self.branch2(x2)], axis=1)
        else:
            out = paddle.concat([self.branch1(x), self.branch2(x)], axis=1)
        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Layer):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        super().__init__()
        input_channels = 3
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2D(input_channels, output_channels, 3, 2, 1, bias_attr=False),
                                   nn.BatchNorm2D(output_channels), nn.ReLU())
        self.maxpool = nn.MaxPool2D(3, 2, 1)
        input_channels = output_channels
        for stage_index, (repeats, output_channels) in enumerate(zip(stages_repeats, stages_out_channels[1:]), 2):
            layers = [inverted_residual(input_channels, output_channels, 2)]
            layers.extend(inverted_residual(output_channels, output_channels, 1) for _ in range(repeats - 1))
            setattr(self, 'stage' + str(stage_index), nn.Sequential(*layers))
            input_channels = output_channels
        self.conv5 = nn.Sequential(nn.Conv2D(input_channels, stages_out_channels[-1], 1, 1, 0, bias_attr=False),
                                   nn.BatchNorm2D(stages_out_channels[-1]), nn.ReLU())
        self.fc = nn.Linear(stages_out_channels[-1], num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = paddle.mean(x, axis=[2, 3])
        return self.fc(x)
'''


def _api():
    from SemTrans.enhanced_conversion.framework_knowledge_retrivals import enhanced_conversion_agent
    from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
    from SemTrans.math_symbol_base.frontends.materialize import create_symbolnet, load_model_instance
    from SemTrans.testing_and_repair.constraint_driven_repair import assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix, get_repair_system_prompt
    from SemTrans.testing_and_repair.logging import CONVERSION_LOG
    import logging
    CONVERSION_LOG.handlers.clear(); CONVERSION_LOG.addHandler(logging.NullHandler()); CONVERSION_LOG.setLevel(logging.CRITICAL)
    return (enhanced_conversion_agent, AbsTensor, create_symbolnet, load_model_instance, assemble_batch_user_prompt, auto_repair_pipeline, call_llm_for_fix, get_repair_system_prompt, CONVERSION_LOG)


def main():
    cases = json.loads((ROOT / 'server/semtrans_demo_cases.json').read_text(encoding='utf-8'))
    targets = {'alexnet': None, 'densenet': DENSENET, 'shufflenetv2': SHUFFLENET}
    # AlexNet is covered by verify_real_alexnet.py; keep this script focused on
    # the two nested real models to keep the offline regression under a minute.
    results = []
    # Cross-framework FP32 kernels can differ by a few ulps around BatchNorm
    # (the first normalized activation can amplify a 1e-6 convolution drift).
    # The product route records this explicit demo tolerance instead of
    # silently treating that backend drift as a semantic conversion failure.
    config = {'bounded_demo': True, 'max_sample': 1, 'target_count': 1, 'max_time_ms': 900, 'seed': 20261028}
    for name in ('densenet', 'shufflenetv2'):
        profile = cases[name]
        source = (ROOT / profile['source_path']).resolve()
        with tempfile.TemporaryDirectory(prefix='semtrans-real-' + name + '-') as directory:
            target = Path(directory) / 'target.py'; target.write_text(targets[name], encoding='utf-8')
            report, issues = _validate(source, target, profile, config, _api(), seed=20261028, phase=name, rtol=1e-3, atol=1e-3)
        results.append({'model': name, 'status': report['status'], 'verified': report['verified'], 'valid_paired_cases': report['valid_paired_cases'], 'elapsed_s': report['elapsed_s'], 'issues': issues})
    print(json.dumps({'passed': all(item['verified'] for item in results), 'results': results}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(item['verified'] for item in results) else 1)


if __name__ == '__main__':
    main()
