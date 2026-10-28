import os

try:
    from openai import OpenAI
except ImportError:  # Optional dependency for LLM-backed conversion.
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:  # Optional dependency for Claude-backed conversion.
    Anthropic = None


def _openai_client(api_key_env, base_url_env, default_base_url):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None
    if OpenAI is None:
        raise ImportError("Install SemTrans with the 'llm' extra to use this backend.")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get(base_url_env, default_base_url),
    )


def _anthropic_client(api_key_env, base_url_env, default_base_url):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None
    if Anthropic is None:
        raise ImportError("Install SemTrans with the 'llm' extra to use this backend.")
    return Anthropic(
        api_key=api_key,
        base_url=os.environ.get(base_url_env, default_base_url),
    )


class InputConfig:
    # --- Processing Mode ---
    MODE_SINGLE = 'SINGLE_FILE'
    MODE_BATCH  = 'BATCH_DIR'

    # --- Content Type ---
    TYPE_CODE   = 'SOURCE_CODE'
    TYPE_STATIC = 'STATIC_MODEL'


def detect_input_config(source_path):
    """
    Detects the processing mode and content type of the input.
    Returns:
        (processing_mode, content_type)
    """
    if not os.path.exists(source_path):
        raise ValueError(f"Input path not found: {source_path}")

    # 1. Handle File Input
    if os.path.isfile(source_path):
        ext = os.path.splitext(source_path)[1].lower()

        # [Case A] Python Source Code
        if ext == '.py':
            return InputConfig.MODE_SINGLE, InputConfig.TYPE_CODE

        # [Case B] Static Model Files
        # Add all supported static formats here
        static_exts = {
            '.pt', '.pth', '.bin', '.ckpt',  # PyTorch
            '.onnx',  # ONNX
            '.pb', '.tflite', '.h5', '.keras',  # TensorFlow
            '.pdmodel'  # Paddle
        }
        if ext in static_exts:
            return InputConfig.MODE_SINGLE, InputConfig.TYPE_STATIC

        # Unknown file type
        raise ValueError(f"Unsupported file extension: {ext}")

    # 2. Handle Directory Input
    if os.path.isdir(source_path):
        files = os.listdir(source_path)

        # [Case C] TensorFlow SavedModel Dir (Treated as ONE static model)
        if os.path.exists(os.path.join(source_path, "saved_model.pb")):
            return InputConfig.MODE_SINGLE, InputConfig.TYPE_STATIC

        # [Case D] Paddle Inference Model Dir (Treated as ONE static model)
        # Checks for .pdmodel and corresponding params
        if any(f.endswith('.pdmodel') for f in files) or \
                ('model' in files and 'params' in files):
            return InputConfig.MODE_SINGLE, InputConfig.TYPE_STATIC

        # [Case E] Batch Source Code Processing (Standard Directory)
        return InputConfig.MODE_BATCH, InputConfig.TYPE_CODE

    raise ValueError(f"Invalid source path type: {source_path}")


LLM_clients = {
    "deepSeek": {
        "API": _openai_client(
            "SEMTRANS_DEEPSEEK_API_KEY",
            "SEMTRANS_DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1",
        ),
        "type": os.environ.get("SEMTRANS_DEEPSEEK_MODEL", "deepseek-v4-pro")
    },
    "qwen": {
        "API": _openai_client(
            "SEMTRANS_QWEN_API_KEY",
            "SEMTRANS_QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "type": os.environ.get("SEMTRANS_QWEN_MODEL", "qwen3-235b-a22b")
    },
    "claude": {
        "API": _anthropic_client(
            "SEMTRANS_CLAUDE_API_KEY",
            "SEMTRANS_CLAUDE_BASE_URL",
            "https://api.anthropic.com",
        ),
        "type": os.environ.get("SEMTRANS_CLAUDE_MODEL", "claude-sonnet-4-20250514")
    },
    "kimi": {
        "API": _openai_client(
            "SEMTRANS_KIMI_API_KEY",
            "SEMTRANS_KIMI_BASE_URL",
            "https://api.moonshot.cn/v1",
        ),
        "type": os.environ.get("SEMTRANS_KIMI_MODEL", "kimi-k2-0711-preview")
    },
    "gemini":{
        "API": _openai_client(
            "SEMTRANS_GEMINI_API_KEY",
            "SEMTRANS_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "type": os.environ.get("SEMTRANS_GEMINI_MODEL", "gemini-3-pro-preview")
    }
}

conversion_patterns_template = {
    "Syntax Inconsistency Example": [
        "\n//  Example: Function signature differences"
        "\nSource: `framework_A.function(input, dim=axis)`"
        "\nTarget: `framework_B.function(input, axis=dim)`",
        "\n//  Example: Function Usage differences"
        "\nSource: `framework_A.optim.optimizer(...)`\n 'framework_A.optim.lr_scheduler(...)`'"
        "\nTarget: `framework_B.optim.lr_scheduler(...)`\n 'framework_B.optim.optimizer(...)'",
        "\n//  Example: Container Construction (Fixes Argument Error)"
        "\n// Problem: Source accepts List/Dict. Target requires Unpacked Args (*args)."
        "\nSource: `self.seq = framework_A.Sequential(layer_list)`"
        "\nTarget: `self.seq = framework_B.Sequential(*layer_list)`",
        "\n//  Example: Container Inheritance -> Composition (Fixes __init__ crash)"
        "\nSource: `class Block(framework_A.Sequential): ...`"
        "\nTarget: `class Block(framework_B.Layer): self.seq = framework_B.Sequential(...)`"

    ],
    "Semantic Inconsistency Example": [
        "\n// Example: Missing Feature -> Manual Implementation"
        "\nSource: `result = framework_A.divide(a, b, rounding_mode='floor')`"
        "\nTarget: `# Target lacks rounding_mode parameter; add explicit flooring \nresult = framework_B.floor(framework_B.divide(a, b))`",
        "\n// Example: Implicit Mode -> Explicit Calculation (CRITICAL)"
        "\n// Source uses a string flag (e.g., mode='fan_out', 'auto') to imply a calculation."
        "\n// Target requires the calculated numeric value directly. You must DERIVE it from tensor attributes."
        "\nSource: `framework_A.init.initializer(w, mode='fan_out')`"
        "\nTarget: calc_val = w.shape[0] * w.shape[2] * w.shape[3]  # Manually computing fan_out`"
        "\n`framework_B.init.initializer(explicit_param=calc_val)(w)`"
    ],
    "Missing Operator Example": [
        "\n// Example: Source framework has a complex operation that is missing in target framework"
        "\nSource: `framework_A.complex_operation(x, y)`"
        "\nTarget: `framework_B.basic_op1(x) + framework_B.basic_op2(y)`",
        "\n// Example: Optimization/Utility Cleanup (Zombie Code Removal)"
        "\n// If an optimization module (e.g., checkpoint, recompute) is missing, REMOVE the optimization branch entirely."
        "\nSource: `if self.use_memory_opt:`\n` out = framework_A.utils.checkpoint(self.func, x)`\n`else:`\n` out = self.func(x)`"
        "\nTarget: `# Optimization module missing; fallback to standard execution path`\n`out = self.func(x)`",
        "\n// Example: Optimization Fallback (Preserve Semantics, Ignore Memory/Speed)"
        "\n// Diagnosis: Missing module is for optimization (e.g. checkpoint, recompute)."
        "\n// Fix: REMOVE the wrapper, execute the core function directly."
        "\nSource: `out = checkpoint(self.layer_block, x)`"
        "\nTarget: `out = self.layer_block(x)  # Fallback to standard forward`"
    ]
}

conversion_examples = {   ('pytorch', 'paddle'): [   '//  Example: Function signature differences\n'
                               'Source: `torch.sum(input, dim=1, keepdim=True)`\n'
                               'Target: `paddle.sum(input, axis=1, keepdim=True)`\n'
                               '\n'
                               '//  Example: Function Usage differences\n'
                               'Source: `torch.optim.Adam(model.parameters(), lr=0.001)`\n'
                               " 'torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)`'\n"
                               'Target: `paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())`\n'
                               " 'paddle.optimizer.lr.StepLR(learning_rate=0.001, step_size=30, gamma=0.1)`'\n"
                               '\n'
                               '//  Example: Container Construction (Fixes Argument Error)\n'
                               '// Problem: Source accepts List/Dict. Target requires Unpacked Args (*args).\n'
                               'Source: `self.seq = torch.nn.Sequential(layer_list)`\n'
                               'Target: `self.seq = paddle.nn.Sequential(*layer_list)`\n'
                               '\n'
                               '//  Example: Container Inheritance -> Composition (Fixes __init__ crash)\n'
                               'Source: `class Block(torch.nn.Sequential): ...`\n'
                               'Target: `class Block(paddle.nn.Layer): self.seq = paddle.nn.Sequential(...)`\n'
                               '\n',
                               '// Example: Missing Feature -> Manual Implementation\n'
                               "Source: `result = torch.nn.functional.gelu(x, approximate='tanh')`\n"
                               'Target: `# Target lacks approximate parameter; use exact GELU and manual tanh '
                               'approximation\n'
                               'result = paddle.nn.functional.gelu(x)`\n'
                               '\n'
                               '// Example: Implicit Mode -> Explicit Calculation (CRITICAL)\n'
                               "// Source uses a string flag (e.g., mode='fan_out', 'auto') to imply a calculation.\n"
                               '// Target requires the calculated numeric value directly. You must DERIVE it from '
                               'tensor attributes.\n'
                               "Source: `torch.nn.init.kaiming_uniform_(w, a=0, mode='fan_out')`\n"
                               'Target: calc_val = paddle.multiply(paddle.multiply(paddle.to_tensor(w.shape[0]), '
                               'paddle.to_tensor(w.shape[2])), paddle.to_tensor(w.shape[3]))\n'
                               '`paddle.nn.initializer.KaimingUniform(fan_in=calc_val.item())(w)`\n'
                               '\n',
                               '// Example: Source framework has a complex operation that is missing in target '
                               'framework\n'
                               "Source: `torch.nn.functional.gelu(x, approximate='tanh')`\n"
                               'Target: `paddle.multiply(paddle.scale(x, scale=0.5), '
                               'paddle.add(paddle.tanh(paddle.scale(x, scale=0.7978845608)), paddle.full(x.shape, '
                               '1.0)))`\n'
                               '\n'
                               '// Example: Optimization/Utility Cleanup (Zombie Code Removal)\n'
                               '// If an optimization module (e.g., checkpoint, recompute) is missing, REMOVE the '
                               'optimization branch entirely.\n'
                               'Source: `if self.use_checkpoint:`\n'
                               '` out = torch.utils.checkpoint.checkpoint(self.layer, x)`\n'
                               '`else:`\n'
                               '` out = self.layer(x)`\n'
                               'Target: `# Optimization module missing; fallback to standard execution path`\n'
                               '`out = self.layer(x)`\n'
                               '\n'
                               '// Example: Optimization Fallback (Preserve Semantics, Ignore Memory/Speed)\n'
                               '// Diagnosis: Missing module is for optimization (e.g. checkpoint, recompute).\n'
                               '// Fix: REMOVE the wrapper, execute the core function directly.\n'
                               'Source: `out = torch.utils.checkpoint.checkpoint(self.layer_block, x)`\n'
                               'Target: `out = self.layer_block(x)  # Fallback to standard forward`\n'
                               '\n']}
