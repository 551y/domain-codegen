import json
from .config import LLM_clients, conversion_patterns_template, conversion_examples
from SemTrans.llm import create_model_completion, process_model_response


def get_conversion_example(source_framework, target_framework, logger, is_JIT, LLM_api, LLM_type):
    """
    Retrieve conversion examples: Check the in-memory cache (utils.conversion_examples) first;
    if not found, invoke the Agent to generate them.
    """

    cache_key = (source_framework, target_framework)

    if not is_JIT and cache_key in conversion_examples:
        cached_data = conversion_examples[cache_key]
        return cached_data[0], cached_data[1], cached_data[2]

    conversion_example = []

    ordered_keys = [
        "Syntax Inconsistency Example",
        "Semantic Inconsistency Example",
        "Missing Operator Example"
    ]

    for operator_type in ordered_keys:
        if operator_type not in conversion_patterns_template:
            conversion_example.append("")
            continue

        templates = conversion_patterns_template[operator_type]
        example_content = ""

        for template in templates:
            content = conversion_example_generation_agent(
                operator_type,
                template,
                source_framework,
                target_framework,
                logger,
                LLM_api,
                LLM_type
            )
            example_content = example_content + content + "\n\n"

        conversion_example.append(example_content)

    conversion_examples[cache_key] = conversion_example
    # Keep generated examples in memory only. A conversion run must not modify
    # the installed source tree or tracked configuration files.

    return conversion_example[0], conversion_example[1], conversion_example[2]


def conversion_example_generation_agent(conversion_type, type_template, source_framework, target_framework, logger,
                                        LLM_api, LLM_type):
    client = LLM_api
    if client is None:
        raise EnvironmentError(
            "The selected LLM client is not configured; set its SEMTRANS_*_API_KEY environment variable."
        )
    if source_framework is None:
        raise EnvironmentError("Please provide sourceFrameworkCode, sourceFramework.")
    system_prompt = f""" You are an expert in deep learning framework conversion. Your task is to generate exactly 
    ONE code example for cross-framework conversion based on the user's provided template. You must strictly follow 
    the template format without any modifications, additions, or omissions. Do not include any extra text, 
    explanations, or comments outside the template. Your output should be exactly the template filled with realistic 
    function names and parameters from the specified frameworks."""

    user_prompt = f"""
        Generate ONE {conversion_type} example for converting from {source_framework} to {target_framework}.

        **Template to Follow:**
        {type_template}

        **Requirements:**
        - **CRITICAL PERFORMANCE RULE: Explicitly use {target_framework} native APIs (e.g., `{target_framework}.add`) instead of Python operators.**

        - **CLARITY RULE: Use CONCRETE VALUES (e.g., `1`, `True`, `'mean'`) or GENERIC VARIABLE NAMES (e.g., `x`, `y`) for arguments.** - Avoid: `dim=dim`, `axis=axis`, `dim=axis` (Confusing).
        - ✅ Good: `dim=1`, `axis=1`, `keepdim=True`.

        - Use only actual function names and parameters from {source_framework} and {target_framework}.
        - Ensure the example is realistic and common in practice.
        - Output must exactly match the template format, including all comments, line breaks, and code styling.
        - Generate only ONE example.

        **Frameworks:**
        - Source: {source_framework}
        - Target: {target_framework} 
    """

    response = create_model_completion(
        client=client,
        model_type=LLM_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    content = process_model_response(
        model_type=LLM_type,
        response=response,
        logger=logger
    )
    return content


def build_blind_tensorrt_conversion_prompt(code_content, static_context):
    """Build the shared initial prompt without executing or inspecting a model.

    ``static_context`` must originate from the benchmark manifest. Runtime
    errors, exported graphs, artifacts, numerical values, and SemTrans-JIT
    evidence are deliberately not accepted by this interface.
    """
    if not isinstance(static_context, dict):
        raise TypeError("TensorRT blind conversion requires a static manifest context")
    allowed = {
        "framework",
        "domain",
        "model_dir",
        "family_group",
        "factory_provider",
        "factory_identifier",
        "factory_kwargs",
        "input_adapter",
        "input_specs",
        "factory_provenance",
        "stress_group",
        "replaces_model",
    }
    unexpected = set(static_context) - allowed
    if unexpected:
        raise ValueError(
            "TensorRT blind conversion received non-static fields: "
            + ", ".join(sorted(unexpected))
        )

    descriptor = json.dumps(
        {key: static_context.get(key, "") for key in sorted(allowed)},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    system_prompt = """You are a senior PyTorch and TensorRT compiler engineer.

Generate one complete Python conversion adapter from the static PyTorch source provided by the user. This is a blind initial conversion: no model has been instantiated or run, no graph has been exported, and no compiler or numerical feedback exists.

OUTPUT CONTRACT:
1. Output raw executable Python code only, without Markdown or explanation.
2. Define exactly this public entry point:
   convert(model, example_inputs, artifact_path)
3. The function must return a dictionary containing at least `artifact` and `exported_program`.
4. The function may functionalize the model or rewrite an FX/ATen ExportedProgram when statically justified by the supplied source.
5. It must compile through `torch_tensorrt.dynamo.compile` using FP32, `disable_tf32=True`, `require_full_compilation=True`, and `min_block_size=1`.
   Call the compiler with the example inputs exactly once as the keyword
   argument `inputs=tuple(example_inputs)`; do not pass `example_inputs` as a
   positional argument and also pass `inputs=...`, because that produces the
   invalid `multiple values for argument 'inputs'` contract error.
6. It must save the compiled artifact to `artifact_path` with `torch_tensorrt.save(..., output_format="exported_program")`; do not use ONNX, TorchScript conversion as a substitute, hybrid execution, or PyTorch fallback.
7. Do not synthesize weights, replace the model with a different network, change observable outputs, or use test-specific constants.
8. Do not catch an error and silently return the original PyTorch model.
9. Do not execute conversion at module import time. Only `convert(...)` may perform export, compilation, or saving when the harness later calls it.
10. Do not include tests, example invocations, random input generation, benchmarking, logging of tensor values, or environment-specific absolute paths.

The later harness, not this adapter, will load and independently validate the saved strict TensorRT artifact."""

    user_prompt = f"""Create the blind PyTorch-to-TensorRT conversion adapter for this benchmark model.

STATIC BENCHMARK DESCRIPTOR
{descriptor}

FIXED TARGET CONTRACT
- Target compiler API: torch_tensorrt.dynamo.compile
- Precision: FP32 only
- disable_tf32: True
- require_full_compilation: True
- min_block_size: 1
- fallback: forbidden
- ONNX: forbidden
- The provided `example_inputs` are the only example inputs available to convert(...).
- Use exactly one compiler input form: `torch_tensorrt.dynamo.compile(exported_program, inputs=tuple(example_inputs), ...)`.
- Never write `compile(exported_program, tuple(example_inputs), inputs=tuple(example_inputs), ...)`.
- Preserve the source model's output structure and semantics.

STATIC PYTORCH SOURCE
{code_content}

Return only the complete Python module implementing convert(model, example_inputs, artifact_path)."""

    return {
        "system_prompt": system_prompt.strip(),
        "user_prompt": user_prompt.strip(),
        "prompt_field_sources": {
            "model_source": "static Python files only",
            "model_descriptor": "RQ1 PyTorch workload manifest only",
            "compiler_contract": "fixed experiment protocol",
            "runtime_information": "none",
            "native_baseline_information": "none",
            "D_repair": "not accessed",
            "D_test": "not accessed",
            "SemTrans_JIT": "not accessed",
        },
    }


def enhanced_conversion_agent(code_content, sourceFramework, targetFramework, logger, is_JIT,
                              LLM_api, LLM_type, static_context=None, task_guidance=None,
                              compact_prompt=False):
    """
    The enhanced prompt consists of three parts:
        - General Instructions (task description, output format),
        - Model Semantic Alignment (including model structure, operators, and framework default behavior)，
        - Source Model Code.
    """
    client = LLM_api
    if client is None:
        raise EnvironmentError(
            "The selected LLM client is not configured; set its SEMTRANS_*_API_KEY environment variable."
        )
    if sourceFramework is None:
        raise EnvironmentError("Please provide sourceFrameworkCode, sourceFramework.")

    if str(targetFramework).lower() in {"tensorrt", "torch-tensorrt", "torch_tensorrt"}:
        prompts = build_blind_tensorrt_conversion_prompt(code_content, static_context)
        logger.info("Sub-phase: Generating blind TensorRT conversion adapter")
        response = create_model_completion(
            client=client,
            model_type=LLM_type,
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
        )
        return process_model_response(
            model_type=LLM_type,
            response=response,
            logger=logger,
        )

    if compact_prompt:
        logger.info("Sub-phase: Generating compact demo conversion candidate")
        system_prompt = f"""
You convert executable {sourceFramework} model code to {targetFramework}.
Return ONLY complete executable Python source code. Do not return analysis,
chain-of-thought, Markdown fences, or explanations.
Keep the public model class name, constructor arguments, module hierarchy,
parameter shapes, evaluation behavior, and forward output semantics.
Use native {targetFramework} layers. Resolve framework API differences
directly in code, including container construction and imports.
The result must instantiate and run with the supplied input profile.
""".strip()
        user_prompt = f"""
Convert this real {sourceFramework} model to {targetFramework}.
Return code only and keep the same class and constructor.

MODEL-SPECIFIC GUIDANCE:
{task_guidance or "Preserve the source model exactly; do not simplify it."}

SOURCE CODE:
{code_content}
""".strip()
        response = create_model_completion(
            client=client,
            model_type=LLM_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return process_model_response(model_type=LLM_type, response=response, logger=logger)

    logger.info("Sub-phase: Generating operator conversion example")
    # get operator conversion example
    syntax_inconsistency_example, semantic_inconsistency_example, missing_operator_example = get_conversion_example(
        sourceFramework, targetFramework, logger, is_JIT, LLM_api, LLM_type)

    system_prompt = f"""
        You are an expert in deep learning model conversion from {sourceFramework} to {targetFramework}.

        **YOUR PRIMARY GOAL**: 
        Generate a target model class that is **immediately instantiable** and **executable**. 
        Any conversion that causes `TypeError` during `__init__` or `forward` is considered a CRITICAL FAILURE.

        CRITICAL OUTPUT RULES:
        1. **GUARANTEE INSTANTIABILITY**:
            - The generated class MUST successfully initialize using the original parameters.
            - **Architectural Check**: Do NOT blind-copy inheritance patterns. 
              - *Example*: If Source uses `class Block(nn.Sequential)`, check if Target supports this. If not (e.g. Paddle/MindSpore), REFACTOR to `class Block(nn.Layer)` and use composition (`self.seq = nn.Sequential(...)`).

        2. **GUARANTEE CALLABILITY**:
            - The `forward` method signature MUST match the input shape requirements.
            - Ensure container classes (like `Sequential`) are constructed correctly.
              - *Example*: If Target's `Sequential` expects unpacked args (`*layers`) but Source provides a list, you MUST unpack it (`Sequential(*list_of_layers)`).

        3. **Output EXCLUSIVELY raw executable Python code**.
        4. NO markdown blocks or explanations.
        5. **ADAPT IMPORTS**: Do NOT blindly reuse source imports. 
        6. **HANDLE MISSING UTILITIES (Zombie Code Rule)**: 
            - If an optimization module (e.g. `checkpoint`) is missing, **REMOVE the branch** and fallback to standard logic.
            - Do not generate calls to non-existent modules.
        7. **CLEANUP CALL SITES**: Sync deletions of imports with their usage.
        8. MAINTAIN exact parameter names and defaults in `__init__`.
        9. STRICTLY PRESERVE type hints (KEEP shape string).
        10. Update Tensor class names in type hints.
        """

    conversion_cot = f"""
        **=== CONVERSION STRATEGY (CHAIN OF THOUGHT) ===**

        Before generating the code, verify the following architectural decisions:

        1.  **Analyze Class Inheritance (The "Is-A" Problem)**:
            -   *Source Check*: Does any class inherit from a Container (e.g., `Sequential`, `ModuleDict`)?
            -   *Target Strategy*: Most target frameworks (Paddle/MindSpore) require inheriting from the Base Layer (e.g., `nn.Layer`, `nn.Cell`).
            -   *Action*: If yes, **Refactor to Composition**. Create the container as a member variable (e.g., `self.seq = ...`) instead of inheriting from it.

        2.  **Analyze Container Construction (The "Args" Problem)**:
            -   *Source Check*: Are `Sequential` or `LayerList` initialized with a List or OrderedDict?
            -   *Target Strategy*: Check if the target framework requires unpacked arguments (`*args`).
            -   *Action*: If yes, use star-unpacking: `nn.Sequential(*layers_list)`.

        3.  **Analyze Dependency & Optimization (The "Zombie" Problem)**:
            -   *Source Check*: Are there imports for optimization tools (e.g., `checkpoint`, `memory_efficient`, `spectral_norm`)?
            -   *Target Strategy*: If the exact tool is missing in {targetFramework}, **Cut the Loss**.
            -   *Action*: Remove the import AND the conditional branch. Fallback to the standard (unoptimized) implementation path.

        4.  **Analyze Parameter Initialization (The "Implicit" Problem)**:
            -   *Source Check*: Are initialization parameters string-based (e.g., `mode='fan_out'`)?
            -   *Target Strategy*: Target API might require explicit numeric values.
            -   *Action*: Calculate the value manually (e.g., from `weight.shape`) and pass it explicitly.
        """

    user_prompt = f"""
    **=== TASK DESCRIPTION ===**

    You are a deep learning framework conversion specialist. 
    You are requested to convert the following source model from {sourceFramework} to {targetFramework}.
    **The converted code must run without errors during `model = Model()` and `output = model(input)`.**

    **=== END DESCRIPTION ===**

    **=== CONVERSION GUIDELINES ===**

    **1. Syntax Inconsistency Example:**
    \t{syntax_inconsistency_example}

    **2. Semantic Inconsistency Example:**
    \t{semantic_inconsistency_example}

    **3. Missing Operator / Module Cleanup Example (Zombie Code):**
    \t{missing_operator_example}

    **=== END GUIDELINES ===**
    
    {conversion_cot}
    
    **=== SOURCE MODEL CONTENT ===**

    {code_content}

    **=== END CONTENT ===**
    """
    if task_guidance:
        user_prompt += f"\n\n=== ROUTE-SPECIFIC SEMTRANS TASK GUIDANCE ===\n{task_guidance}\n"
    logger.info("Sub-phase: Converting source model")
    response = create_model_completion(
        client=client,
        model_type=LLM_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    content = process_model_response(
        model_type=LLM_type,
        response=response,
        logger=logger
    )
    return content
