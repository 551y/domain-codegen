import textwrap
from collections import defaultdict

from SemTrans.enhanced_conversion.framework_knowledge_retrivals import get_conversion_example
from SemTrans.testing_and_repair.differential_testing import dual_differential_testing
from SemTrans.llm import create_model_completion, process_model_response


def get_repair_system_prompt(source_fw, target_fw):
    """
    Generate a generic System Prompt based on the source and target framework names.
    """
    template = f"""You are an expert Artificial Intelligence engineer specializing in Deep Learning Framework Conversion.
Your specific expertise is converting **{source_fw}** models to **{target_fw}** models.

Your goal is to fix the target model code based on the provided diagnosis. The issues may be:
1. **Runtime Crashes / Syntax Errors**: The code fails to run or load (provided via Traceback).
2. **Semantic Mismatches**: The code runs but outputs incorrect values (provided via Semantic Constraints).

CRITICAL RULES:
1. **Analyze the Input Source**:
   - If provided with **Semantic Constraints**, STRICTLY modify the logic to satisfy the mathematical/mechanism rules.
   - If provided with an **Error Traceback**, fix the code syntax or tensor shape operations to resolve the crash.
2. Maintain the original coding style and variable naming where possible.
3. Output **ONLY** the fixed Python code block (inside ```python ... ```).
4. Do NOT include explanations, theoretical background, or conversational fillers. Just the code.
5. When fixing multiple issues, ensure the code remains consistent and valid.
"""
    return template.strip()


def assemble_batch_user_prompt(repair_tasks, full_target_code, source_code_ref, is_crash,
                               sourceFramework, targetFramework, logger, is_JIT, LLM_api, LLM_type):
    """
    Universal Repair Prompt Assembler.
    Fixes:
    1. Correctly extracts 'prompt_content' key from pipeline tasks.
    2. Dynamic strategy injection for Initialization errors vs Logic errors.
    """

    # 1. Assemble Issues List
    issues_content = []
    has_init_issue = False  

    for i, task in enumerate(repair_tasks):
        error_type = task.get('cluster_key', 'Unknown Error')
        root_cause = task.get('root_cause', 'Unknown Cause')
        fault_layer = task.get('fault_layer', 'Global/Unknown')

        
        
        raw_content = task.get('prompt_content') or task.get('prompt')

        
        if isinstance(raw_content, (list, tuple)):
            
            detailed_info = raw_content[-1]
        elif raw_content:
            detailed_info = str(raw_content)
        else:
            detailed_info = "No detailed diagnosis trace available."

        
        if "init_policy" in error_type.lower() or "initialization" in root_cause.lower():
            has_init_issue = True

        issue_block = f"""
        # ---------------------------------------------------
        # [ISSUE #{i + 1}]
        # ---------------------------------------------------
        # **Error Type**: {error_type}
        # **Location**: {fault_layer}
        # **Root Cause**: {root_cause}

        # --- Detailed Report (Diagnosis & Evidence) ---
        {detailed_info}
        """
        issues_content.append(textwrap.dedent(issue_block))

    all_issues_text = "\n".join(issues_content)

    # 2. Prepare Dynamic Sections (RAG Patterns)
    patterns_section = ""

    
    
    if is_crash or not has_init_issue:
        # A. Get RAG Patterns
        try:
            
            syntax_ex, semantic_ex, missing_ex = get_conversion_example(
                sourceFramework, targetFramework, logger, is_JIT, LLM_api, LLM_type
            )

            patterns_section = f"""
    ## REFERENCE: Critical Repair Patterns
    **Use these patterns to solve common inconsistencies.**

    **1. Syntax Pattern:**
    {syntax_ex}

    **2. Semantic/Architecture Pattern:**
    {semantic_ex}

    ---
    """
        except Exception:
            
            patterns_section = ""

    # 3. Dynamic Strategy Injection
    

    if has_init_issue:
        
        strategy_text = """
    1. **Analyze Init Policy**: Look at [ISSUE #1]. Compare the Source's `init` logic (e.g., Kaiming/Xavier) with the Target.
    2. **Rewrite Initialization**: 
       - If the Source uses explicit `nn.init` calls, replicate them in the Target's `__init__` or a dedicated `_init_weights` method.
       - Ensure `fan_in` vs `fan_out` modes match exactly.
    3. **Fix Global Strategy**: Do not just fix one layer; apply the correct policy to all applicable layers (Conv2d, Linear).
    """
    elif is_crash:
        
        strategy_text = """
    1. **Analyze Traceback**: Locate the exact line in [ISSUE] causing the crash.
    2. **Refactor Code**: 
       - Check for API mismatches (e.g., argument names).
       - Check for Shape mismatches (Padding, Stride).
       - Fix Inheritance or Import errors.
    """
    else:
        
        strategy_text = """
    1. **Analyze Constraints**: Read the Semantic Diagnosis in [ISSUE].
    2. **Trace Math**: Compare the `forward` calculation step-by-step.
    3. **Adjust Operators**: Check for subtle differences (e.g., `epsilon` values, `axis` direction, or pre-processing).
    """

    # 4. Assemble Final Prompt
    final_prompt_template = f"""
    # Batch Repair Task: Fix Issues via Source-Target Comparison

    The converted code has failed validation. You must identify the root cause by comparing the Source implementation with the Target implementation.

    ---
    {patterns_section}

    ## PART 1: Reference Source Code (The "Ground Truth")
    Use this to understand the original design pattern.
    ```python
    {source_code_ref if source_code_ref else "# Not available"}
    ```

    ---

    ## PART 2: Current Target Code (The "Buggy" One)
    ```python
    {full_target_code}
    ```

    ---

    ## PART 3: Issue Report (Diagnosis Results)
    {all_issues_text}

    ---

    ## PART 4: Execution Strategy
    {strategy_text}

    **Goal**: Return the FULL corrected Python code.
    """

    return textwrap.dedent(final_prompt_template).strip()


def auto_repair_pipeline(source_model, target_model, logger, rtol=1e-2, atol=1e-3,
                         use_cuda=False, is_static_source=False, input_search_config=None):
    """
    Fully automated repair pipeline (Responsible solely for diagnosis; does not handle System Prompt or final assembly).
    """

    # 1. Running differential testing
    # ------------------------------------------------------
    """
    Potential Error Types returned by dual_differential_testing:
    
     - 'initialization_mismatch': Global initialization strategy differs (e.g., Kaiming vs Xavier, Fan_in vs Fan_out).
     Action: Rewrite initialization policy.
     - 'structure_mismatch': Failed to load weights into the target model (e.g., Layer count mismatch, Weight shape mismatch).
     Action: Check layer definitions or transpose logic.
     - 'numerical_instability': Output contains NaN or Inf.
     Action: Check for overflow/underflow in math operators (Log, Exp, Div).
     - 'graph_connectivity_error': Output is dead (all zeros) despite active input.
     Action: Check for detached tensors or broken data flow in forward().
     - 'runtime_crash': Execution aborted due to Python exceptions (API mismatch, Device error).
     Action: Check API compatibility or input types.
     - 'output_shape_mismatch': Output tensor shapes differ between frameworks.
     Action: Check operator attributes (Padding, Stride, Dilation, Reshape).
     - 'output_value_mismatch': Numerical differences exceed tolerance (rtol/atol).
     Action: Check operator mathematical equivalence.
    """
    failed_cases = dual_differential_testing(
        source_model, target_model, logger, rtol, atol, use_cuda=use_cuda,
        is_static_source=is_static_source, input_search_config=input_search_config
    )

    
    
    real_errors = [case for case in failed_cases if case.get('risk_level') != 'WARNING']
    warnings = [case for case in failed_cases if case.get('risk_level') == 'WARNING']

    if not real_errors:
        if warnings:
            init_warnings = [w for w in warnings if w.get('error_type') == 'fine_tuning_risk']

            if init_warnings:
                logger.info("\n" + "=" * 80)
                logger.info(" [NOTICE] VALIDATION SUCCESSFUL WITH WARNINGS")
                logger.info(" The model logic is correct for Inference. However, check below for Fine-tuning risks.")
                logger.info("=" * 80)

                for w in init_warnings:
                    report_content = w.get('details', 'No details.')
                    for line in report_content.split('\n'):
                        logger.warning(line)

                logger.info("=" * 80 + "\n")

            logger.info(f"[Repair] DiffTest Passed! (Ignored {len(warnings)} Fine-tuning Risks)")

        else:
            logger.info("[Repair] DiffTest Passed. No repair needed.")

        return True, None
    logger.info(f"[Repair] Found {len(real_errors)} critical issues to fix.")

    # Group only critical failures. Warnings are reporting-only evidence and
    # must not consume a repair round.
    error_groups = defaultdict(list)
    for case in real_errors:
        error_groups[case.get('error_type') or 'unknown_error'].append(case)

    repair_tasks = []
    for error_type, cases in error_groups.items():
        evidence_blocks = []
        root_causes = []
        locations = []
        for index, case in enumerate(cases, start=1):
            root_cause = str(case.get('root_cause') or 'Unknown Cause')
            location = str(
                case.get('fault_layer')
                or case.get('case_id')
                or case.get('location')
                or 'Global/Unknown'
            )
            root_causes.append(root_cause)
            locations.append(location)

            # Raw inputs, weights, and full output tensors are not actionable
            # repair evidence and would make prompts unstable and oversized.
            details = []
            for key in (
                'error_message',
                'details',
                'semantic_evidence',
                'formula_evidence',
                'alignment_evidence',
                'first_mismatch',
                'mismatch_details',
                'traceback',
            ):
                value = case.get(key)
                if value not in (None, '', [], {}):
                    details.append(f"{key}: {value}")
            if not details:
                details.append('No additional diagnostic detail was produced.')
            evidence_blocks.append(
                f"Case {index}\n"
                f"location: {location}\n"
                f"root_cause: {root_cause}\n"
                + "\n".join(details)
            )

        repair_tasks.append(
            {
                'cluster_key': str(error_type),
                'root_cause': '; '.join(dict.fromkeys(root_causes)),
                'fault_layer': '; '.join(dict.fromkeys(locations)),
                'prompt_content': "\n\n".join(evidence_blocks),
                'case_count': len(cases),
                'evidence_kind': (
                    'semtrans_local_semantic_evidence'
                    if error_type in {'semantic_mismatch', 'output_value_mismatch'}
                    else 'runtime_or_structural_diagnostic'
                ),
            }
        )

    if not repair_tasks:
        raise RuntimeError('critical differential failures produced no repair tasks')
    logger.info(f"[Repair] Prepared {len(repair_tasks)} actionable repair task groups.")
    return False, repair_tasks

def call_llm_for_fix(system_prompt, user_prompt, logger, LLM_api, LLM_type) :
    client = LLM_api

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


