import psutil, random, math
import numpy as np
import time
from SemTrans.testing_and_repair.util import get_memory_capacity
from SemTrans.math_symbol_base.symbolic_system.symbolic_evaluator import SemanticTracer


def input_search(symbol_model, logger, use_cuda=False, exp_seed=None, search_config=None):
    search_config = search_config or {}
    search_started = time.perf_counter()
    if exp_seed is None:
        exp_seed = search_config.get("seed")
    if exp_seed is None:
        exp_seed = random.getrandbits(32)
    logger.info(f"[Search] Using seed {exp_seed}")

    np.random.seed(exp_seed)
    random.seed(exp_seed)
    symbol_model.set_seed(exp_seed)

    symbol_model.eval()
    symbol_model.use_gradient = False

    def seedme():
        model_seed = random.getrandbits(32)
        nseed = (model_seed + 1) % (2 ** 32)
        random.seed(nseed)
        np.random.seed(nseed)
        symbol_model.set_seed(nseed)

    # ------------------------------------------------------------
    # Test grad
    # If sampling can succeed, grad can succeed too as their initial input are the same.
    MAX_MEM = (
        psutil.virtual_memory().total
        if not use_cuda
        else get_memory_capacity(symbol_model)
    )
    MEM_FACTOR = 0.4

    seedme()

    # Initialize tensor samples
    init_tensor_samples = []
    max_sample, total_nbytes = symbol_model.calculate_max_sample(MAX_MEM, MEM_FACTOR)
    if search_config.get("max_sample") is not None:
        max_sample = min(max_sample, max(0, int(search_config["max_sample"])))
    logger.info(f"[Search] Get {max_sample} samples, requiring {total_nbytes}GB.")
    ESTIMATED_TIME_PER_CASE_MS = 2000
    calculated_time = max_sample * ESTIMATED_TIME_PER_CASE_MS

    MIN_TIME_MS = 1 if search_config.get("bounded_demo") else 10000
    MAX_TIME_CAP_MS = max(1, int(search_config.get("max_time_ms", 300000)))
    max_time_ms = max(MIN_TIME_MS, min(calculated_time, MAX_TIME_CAP_MS))
    for i in range(max_sample):
        data = symbol_model.get_random_inps(use_cuda=use_cuda)
        init_tensor_samples.append(data)

    # Initialize tensor samples
    init_weight_samples = []
    with symbol_model.stop_grad():
        for i in range(max_sample):
            weight_sample = {}
            for name, param in symbol_model.named_parameters():
                weight_sample[name] = symbol_model.get_weight_tensor(
                    param.shape, dtype=param.dtype, use_gpu=use_cuda
                )
            init_weight_samples.append(weight_sample)
            if i % 10 == 0:
                symbol_model.clear_cache()

    # Initialize searcher
    searcher = symbol_model.grad_search(init_tensor_samples, init_weight_samples, use_cuda=use_cuda)

    # Execute batch search
    n_try, sat_inputs = searcher.search(
        max_time_ms=max_time_ms, max_sample=max_sample,
        target_count=max(1, int(search_config.get("target_count", 200))), logger=logger
    )

    numpy_results = []
    for case in sat_inputs:

        inps_dict_np = symbol_model.to_numpy(case["inputs"])
        inps_np_list = list(inps_dict_np.values())

        w_np_dict = symbol_model.to_numpy(case["weights"])

        numpy_results.append({
            "inputs": inps_np_list,
            "weights": w_np_dict
        })

    logger.info(f"[Search] {symbol_model.framework_type} Generated {len(numpy_results)} valid paired cases.")
    telemetry = search_config.get("telemetry")
    if isinstance(telemetry, list):
        telemetry.append({
            "framework": str(symbol_model.framework_type),
            "attempts": int(n_try),
            "valid_cases": len(numpy_results),
            "elapsed_s": round(time.perf_counter() - search_started, 3),
            "budget_ms": int(max_time_ms),
        })
    return numpy_results


def dual_differential_testing(
        source_model,
        target_model,
        logger,
        rtol=1e-2,
        atol=1e-3,
        use_cuda=False,
        is_static_source=False,
        input_search_config=None
):
    """
    Main entry point for Two-Pronged Differential Testing.

    Strategy:
    1. [Phase 0] Static & Structure Checks (Fast Fail).
    2. [Phase 1] Input Generation (Fuzzing).
    3. [Phase 2 - Prong 1] Computational Semantic Verification (White-box).
       - Uses SemanticTracer to verify math symbol alignment.
    4. [Phase 3 - Prong 2] Numerical Logic Verification (Black-box).
       - Uses Dual Fuzzing to verify execution results.
    """

    # --- Internal Helper: Dynamic Structure Verification ---
    def _verify_weight_loading(target_model, weight_dict):
        try:
            target_model.grad_search(None, None, False).apply_weights(target_model, weight_dict)
            return None
        except Exception as e:
            return {
                "case_id": "Structure_Load",
                "error_type": "structure_mismatch",
                "root_cause": "Model Structure/Shape Mismatch",
                "error_message": (
                    f"Failed to load Source weights into Target model.\n"
                    f"This indicates a CRITICAL structural difference (e.g., Shape, Layer Name, or Layer Count).\n"
                    f"Init checks were skipped/discarded because structure is invalid.\n"
                    f"Details: {str(e)}"
                )
            }

    # =========================================================================
    # [Phase 0a] Static Measurement (Silent / Non-Blocking)
    # =========================================================================
    pending_init_risk = None
    if not is_static_source:
        logger.info("[DiffTest-Phase0] Measuring Static Init Stats (Silent)...")
        init_check_result = _check_static_consistency(source_model, target_model, logger)
        if init_check_result and init_check_result['error_type'] == 'initialization_mismatch':
            logger.warning("[DiffTest-Phase0] Initialization mismatch detected. Marking as Risk.")
            pending_init_risk = init_check_result

    # =========================================================================
    # [Phase 0b] Structure Check (The Blocker)
    # =========================================================================
    logger.info("[DiffTest-Phase0] Verifying Structure & Weight Loading...")
    current_source_weights = source_model.get_weights()
    load_error = _verify_weight_loading(target_model, current_source_weights)
    if load_error:
        logger.error(f"[DiffTest-Phase0] Critical Loading Failure: {load_error['error_message']}")
        return [load_error]
    logger.info("[DiffTest-Phase0] Structure Check Passed.")

    # =========================================================================
    # [Phase 1] Dual Fuzzing (Generate Inputs & Weights)
    # =========================================================================
    logger.info("[DiffTest-InputSearch] Generating test inputs...")
    input_search_config = input_search_config or {}
    cases_from_source = input_search(source_model, logger, use_cuda, search_config=input_search_config)
    cases_from_target = input_search(target_model, logger, use_cuda, search_config=input_search_config)
    all_test_cases = cases_from_source + cases_from_target

    if not all_test_cases:
        logger.warning("[DiffTest-Phase1] No inputs generated.")
        if pending_init_risk:
            return [{
                "case_id": "Init_Warning",
                "error_type": "fine_tuning_risk",
                "risk_level": "WARNING",
                "error_message": "Initialization mismatch detected (no inputs to verify logic)."
            }]
        return [{
            "case_id": "Input_Generation",
            "error_type": "input_generation_unavailable",
            "root_cause": "No valid paired inputs were generated",
            "error_message": "SemTrans found no valid paired input within the configured search budget; equivalence is unverified.",
        }]

    # Prepare for inference
    source_model.model.eval()
    target_model.model.eval()

    # =========================================================================
    # [Phase 2 - Prong 1] Computational Semantic Verification (White-box)
    # Carry out mathematical alignment verification based on the symbolic system using SemanticTracer
    # =========================================================================
    logger.info("=" * 60)
    logger.info("[DiffTest-Prong1] Starting Semantic Verification (SymPy/Numpy)...")
    logger.info("=" * 60)

    try:
        tracer = SemanticTracer(source_model, target_model, rtol, atol)
        trace_input = all_test_cases[0]["inputs"]
        semantic_issues = tracer.trace(trace_input)
        semantic_evidence = input_search_config.get("semantic_evidence")
        if isinstance(semantic_evidence, dict):
            semantic_evidence["alignment_records"] = list(tracer.alignment_records)
            semantic_evidence["parameter_alignment_records"] = list(tracer.parameter_alignment_records)
            semantic_evidence["semantic_issue_count"] = len(semantic_issues or [])
            semantic_evidence["first_mismatch"] = tracer.first_mismatch
        if semantic_issues:
            logger.error(f"[DiffTest-Prong1] Semantic Verification Failed with {len(semantic_issues)} issues.")
            semantic_errors = []
            for issue in semantic_issues:
                semantic_errors.append({
                    "case_id": "Semantic_Trace",
                    "error_type": "semantic_mismatch",
                    "root_cause": "Math Symbol / Operator Alignment Failure",
                    "error_message": issue
                })
            return semantic_errors
        logger.info("[DiffTest-Prong1] Semantic Verification Passed (Math formulas aligned).")
    except Exception as e:
        logger.error(f"[DiffTest-Prong1] Tracer Crash: {e}")
        # Numerical agreement alone cannot establish semantic localization.
        # Surface tracer failure explicitly instead of silently converting it
        # into a numerical-only pass.
        return [{
            "case_id": "Semantic_Trace",
            "error_type": "semantic_verification_unavailable",
            "root_cause": "Semantic Tracer Failure",
            "error_message": (
                "Semantic verification could not be completed; numerical "
                f"checking was not used as a substitute: {e}"
            ),
        }]

    # =========================================================================
    # [Phase 3 - Prong 2] Numerical Logic Verification (Black-box)
    
    # =========================================================================
    logger.info("=" * 60)
    logger.info("[DiffTest-Prong2] Starting Numerical Verification (Fuzzing)...")
    logger.info("=" * 60)

    for i, case in enumerate(all_test_cases):
        inputs_np = case["inputs"]
        weights_np = case["weights"]

        # A. Runtime Mechanism Check (NaN/Inf check on native weights)
        mech_failures = _check_mechanism_consistency(source_model, target_model, inputs_np, logger)
        if mech_failures:
            return mech_failures

        # B. Semantic Logic Check (Value check on shared weights)
        logic_failures = _check_logic_consistency(
            source_model, target_model,
            test_inputs=inputs_np, sampled_weights=weights_np,
            rtol=rtol, atol=atol, logger=logger, case_idx=i,
            numerical_checks=input_search_config.get("numerical_checks"),
        )
        if logic_failures:
            logger.info(f"[DiffTest-Prong2] Numerical Mismatch at Case #{i}.")
            return logic_failures

    logger.info("[DiffTest-Prong2] Numerical Logic Verified.")

    # =========================================================================
    # [Phase 4] Final Summary & Risk Reporting
    # =========================================================================
    final_report = []
    if pending_init_risk:
        risk_entry = {
            "case_id": "Init_Policy_Warning",
            "error_type": "fine_tuning_risk",
            "risk_level": "WARNING",
            "error_message": (
                "Default initialization strategy differs from Source.\n"
                "Impact: Benign for Inference, but Critical for Fine-tuning."
            ),
            "details": pending_init_risk.get('error_message')
        }
        final_report.append(risk_entry)
        logger.warning("\n[DiffTest-Summary] Model is logically correct, but has Fine-tuning Risks.")

    return final_report


def _check_static_consistency(source_model, target_model, logger, strict_init=False):
    """
    [StaticCheck] Verify static properties with Op-Type Awareness and Math-based Forensics.
    """
    logger.info("[DiffTest-PreCheck] Checking Static Consistency (Structure & Init Policy)...")

    # =========================================================================
    # 0. Helper: Map parameter names to Layer Types (e.g., 'features.0.weight' -> 'Conv2d')
    # =========================================================================
    def get_layer_type_map(symbol_net):
        raw_model = getattr(symbol_net, 'model', None)
        if not raw_model: return {}
        type_map = {}

        # PyTorch
        if hasattr(raw_model, "named_modules"):
            for layer_name, layer in raw_model.named_modules():
                op_type = layer.__class__.__name__
                if hasattr(layer, "named_parameters"):
                    for p_name, _ in layer.named_parameters(recurse=False):
                        full_name = f"{layer_name}.{p_name}" if layer_name else p_name
                        type_map[full_name] = op_type
        # Paddle
        elif hasattr(raw_model, "named_sublayers"):
            for layer_name, layer in raw_model.named_sublayers(include_self=True):
                op_type = layer.__class__.__name__
                for p_name, _ in layer.named_parameters(include_sublayers=False):
                    full_name = f"{layer_name}.{p_name}" if layer_name else p_name
                    type_map[full_name] = op_type
        # TF
        elif hasattr(raw_model, "layers"):
            for layer in raw_model.layers:
                op_type = layer.__class__.__name__
                for weight in layer.weights:
                    type_map[weight.name] = op_type
        return type_map

    
    src_type_map = get_layer_type_map(source_model)
    tgt_type_map = get_layer_type_map(target_model)

    
    src_weights_dict = source_model.get_weights()
    tgt_weights_dict = target_model.get_weights()

    # =========================================================================
    # 1. Init Strategy Forensics (Math-based naming)
    # =========================================================================
    def identify_strategy(arr):
        """
        Input: Numpy Array
        Output: (StrategyName, FanIn, FanOut, ActualStd)
        """
        if arr.ndim < 2: return "Scalar/Bias", 0, 0, 0

        # A. Calculate Geometric Features (Fan_in / Fan_out)
        shape = arr.shape
        if arr.ndim == 4:  # Conv2D: [Out, In, K, K]
            k_size = shape[2] * shape[3]
            fan_in = shape[1] * k_size
            fan_out = shape[0] * k_size
        elif arr.ndim == 2:  # Linear: [Out, In]
            fan_in = shape[1]
            fan_out = shape[0]
        else:
            return f"UnknownDim({arr.ndim})", 0, 0, 0

        if fan_in == 0 or fan_out == 0: return "Empty", 0, 0, 0

        act_std = float(np.std(arr))

        # B. Theoretical Stds
        # 1. Uniform_FanIn (Common in PyTorch Default) -> 1 / sqrt(3 * fan_in)
        t_uniform_fanin = 1.0 / math.sqrt(3 * fan_in)
        # 2. He_Init (Kaiming Normal) -> sqrt(2 / fan_in)
        t_he = math.sqrt(2.0 / fan_in)
        # 3. Glorot_Init (Xavier Normal) -> sqrt(2 / (fan_in + fan_out))
        t_glorot = math.sqrt(2.0 / (fan_in + fan_out))

        candidates = {
            "Uniform_FanIn": t_uniform_fanin,
            "He_Init": t_he,
            "Glorot_Init": t_glorot
        }

        best_name = "Custom"
        min_diff = float('inf')

        for name, theo_std in candidates.items():
            diff = abs(act_std - theo_std) / (theo_std + 1e-9)
            if diff < min_diff:
                min_diff = diff
                best_name = name

        # 15% tolerance
        if min_diff > 0.15:
            return f"Custom(std={act_std:.4f})", fan_in, fan_out, act_std

        return best_name, fan_in, fan_out, act_std

    # =========================================================================
    # 2. Process Metadata (Now includes Op Type)
    # =========================================================================
    def process_metadata(weights_dict, type_map):
        meta = []
        for name, arr in weights_dict.items():
            if arr.ndim > 1:
                # Retrieve Op Type from map, default to Unknown
                op_type = type_map.get(name, "Unknown")
                meta.append({
                    "id": name,
                    "op_type": op_type,  # New Field
                    "numel": arr.size,
                    "shape": arr.shape
                })
        return meta

    src_list = sorted(process_metadata(src_weights_dict, src_type_map), key=lambda x: x['numel'], reverse=True)
    tgt_list = sorted(process_metadata(tgt_weights_dict, tgt_type_map), key=lambda x: x['numel'], reverse=True)

    # Structure Check
    if len(src_list) != len(tgt_list):
        msg = f"Layer count mismatch: Src={len(src_list)}, Tgt={len(tgt_list)}"
        logger.error(msg)
        return {"case_id": "Structure", "error_type": "count_mismatch", "error_message": msg}

    init_details = []

    # =========================================================================
    # 3. Comparison Loop with Enhanced Logging
    # =========================================================================
    # Table Header: Includes Op Type
    header = f"{'Rank':<7} | {'Layer Name':<30} | {'Src Loc (Type)':<35} | {'Tgt Loc (Type)':<35} | {'Src Algo':<14} | {'Tgt Algo':<14} | {'Status'}"
    logger.info("-" * 170)
    logger.info(header)
    logger.info("-" * 170)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        s_arr = src_weights_dict[s['id']]
        t_arr = tgt_weights_dict[t['id']]

        s_strat, _, _, s_std = identify_strategy(s_arr)
        t_strat, _, _, t_std = identify_strategy(t_arr)

        
        src_loc = getattr(source_model, 'get_layer_location', lambda x: "N/A")(s['id'])
        tgt_loc = getattr(target_model, 'get_layer_location', lambda x: "N/A")(t['id'])

        
        src_type = s.get('op_type', 'Unknown')
        tgt_type = t.get('op_type', 'Unknown')

        src_info_str = f"{src_loc} ({src_type})"
        tgt_info_str = f"{tgt_loc} ({tgt_type})"

        status = "OK"
        is_mismatch = False

        # ---------------------------------------------------------------------
        
        # ---------------------------------------------------------------------

        
        if src_type.lower() != tgt_type.lower():
            status = "OPERATOR_TYPE_ERR"
            is_mismatch = True

        
        elif s_strat != t_strat and "Custom" not in s_strat and "Custom" not in t_strat:
            status = "INIT_STRATEGY_DIFF"  
            is_mismatch = True

        
        
        else:
            denom = max(s_std, t_std, 1e-9)
            if abs(s_std - t_std) / denom > 0.15:
                status = "INIT_VAL_DIFF"
                is_mismatch = True

        
        layer_name = s['id']
        if len(layer_name) > 28: layer_name = layer_name[:25] + "..."

        log_str = f"{i:<4} | {layer_name:<30} | {src_info_str:<35} | {tgt_info_str:<35} | {s_strat:<14} | {t_strat:<14} | {status}"

        if is_mismatch:
            logger.warning(log_str)
            init_details.append({
                "rank": i,
                "layer": layer_name,
                "src_info": src_info_str,
                "tgt_info": tgt_info_str,
                "src_algo": s_strat,
                "tgt_algo": t_strat,
                "issue": status
            })
        else:
            logger.info(log_str)

    logger.info("-" * 170)

    # =========================================================================
    # 4. Error Reporting
    # =========================================================================
    if init_details:
        count = len(init_details)
        logger.warning(f"[DiffTest-PreCheck] Found {count} initialization mismatches.")

        
        detail_msg = "\n".join([
            f"- Rank {item['rank']} [{item['layer']}]\n"
            f"  Src: {item['src_algo']} @ {item['src_info']}\n"
            f"  Tgt: {item['tgt_algo']} @ {item['tgt_info']}  <-- Check this!"
            for item in init_details[:5]
        ])

        if count > 5: detail_msg += f"\n... and {count - 5} more."

        error_payload = {
            "case_id": "Static_Init",
            "error_type": "initialization_mismatch",
            "root_cause": "Initialization Strategy Mismatch",
            "error_message": (
                f"Found {count} layers with mismatched initialization strategies.\n"
                f"Details:\n{detail_msg}\n"
                "Tip: 'Uniform_FanIn' is common in PyTorch; 'Glorot_Init' (Xavier) is common in Paddle/TF."
            ),
            "mismatch_details": init_details
        }

        if strict_init:
            return error_payload
        else:
            logger.warning(">> [Policy] 'strict_init=False'. Ignoring init mismatches.")
            return None

    return None


def execute_model_step(symbol_net, inputs):
    """
    Generic step: Numpy inputs (List or Dict) -> Tensor -> Inference -> Numpy -> Cleanup.
    Supports List (positional arguments) and Dict (keyword arguments).
    """
    import gc

    if isinstance(inputs, (list, tuple)):
        inputs_tensor = [symbol_net.to_tensor(v) for v in inputs]
        input_type = "list"
    elif isinstance(inputs, dict):
        inputs_tensor = {k: symbol_net.to_tensor(v) for k, v in inputs.items()}
        input_type = "dict"
    else:
        inputs_tensor = [symbol_net.to_tensor(inputs)]
        input_type = "list"

    with symbol_net.stop_grad():
        if input_type == "list":
            raw_out = symbol_net.model(*inputs_tensor)
        else:
            raw_out = symbol_net.model(**inputs_tensor)

    numpy_out = symbol_net.to_numpy(raw_out)

    del inputs_tensor
    del raw_out
    gc.collect()
    symbol_net.clear_cache()

    return numpy_out


def generate_safe_weight_sample(symbol_net):
    """
    Generate a set of numerically safe weight samples (Numpy format) based on the model structure.
    Applies Kaiming Initialization principles to prevent numerical explosion.
    """
    weight_sample = {}

    for name, param in symbol_net.named_parameters():
        shape = list(param.shape)

        if len(shape) >= 2:
            fan_in = np.prod(shape[1:])
        else:
            fan_in = 1

        if fan_in > 0:
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = 0.01

        np_noise = np.random.randn(*shape).astype(np.float32)

        weight_sample[name] = np_noise * scale

    return weight_sample


def _check_mechanism_consistency(source_model, target_model, inputs, logger=None):
    """
    Check mechanism issues under native initialization.

    Args:
        inputs: List[np.ndarray]
    """

    def _get_precision_config(data_array):
        dtype = data_array.dtype
        if np.issubdtype(dtype, np.floating):
            try:
                return np.finfo(dtype).eps, 1e-6
            except:
                return 1.19e-7, 1e-6
        elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            return 1e-9, 1e-9
        else:
            return 1.19e-7, 1e-6

    if logger is not None:
        logger.info("[DiffTest-MechanismCheck] Checking Initialization Mechanism (Native Weights)")

    try:
        src_numpy_dict = execute_model_step(source_model, inputs)
        tgt_numpy_dict = execute_model_step(target_model, inputs)

        src_values = [np.asarray(value) for value in src_numpy_dict.values()]
        tgt_values = [np.asarray(value) for value in tgt_numpy_dict.values()]
        if len(src_values) != len(tgt_values):
            return [{
                "case_id": "mech_check",
                "error_type": "output_structure_mismatch",
                "root_cause": "Output Structure Mismatch",
                "error_message": f"Output count differs: Source {len(src_values)} vs Target {len(tgt_values)}.",
            }]
        if not src_values:
            return [{
                "case_id": "mech_check",
                "error_type": "output_structure_mismatch",
                "root_cause": "Missing Model Output",
                "error_message": "Both models returned zero outputs.",
            }]
        for output_index, (src_output, tgt_output) in enumerate(zip(src_values, tgt_values)):
            if src_output.shape != tgt_output.shape:
                return [{
                    "case_id": "mech_check",
                    "error_type": "output_shape_mismatch",
                    "root_cause": "Output Shape Mismatch",
                    "error_message": (
                        f"Output #{output_index} shape differs under native weights: "
                        f"Source {src_output.shape} vs Target {tgt_output.shape}."
                    ),
                }]
            if src_output.dtype != tgt_output.dtype:
                return [{
                    "case_id": "mech_check",
                    "error_type": "output_dtype_mismatch",
                    "root_cause": "Output Dtype Mismatch",
                    "error_message": (
                        f"Output #{output_index} dtype differs under native weights: "
                        f"Source {src_output.dtype} vs Target {tgt_output.dtype}."
                    ),
                }]
            if not np.isfinite(src_output).all() or not np.isfinite(tgt_output).all():
                return [{
                    "case_id": "mech_check",
                    "error_type": "numerical_instability",
                    "root_cause": "Weight Initialization Mechanism",
                    "error_message": f"Output #{output_index} contains NaN or Inf during initialization check.",
                }]
        s_val = src_values[0]
        t_val = tgt_values[0]

        # 2. Check for Numerical Explosion
        # s_max = np.max(np.abs(s_val))
        # t_max = np.max(np.abs(t_val))

        eps, TAU_DEAD = _get_precision_config(s_val)
        TAU_ACTIVE = 1e-3

        s_cv = np.std(s_val) / (np.mean(np.abs(s_val)) + eps)
        t_cv = np.std(t_val) / (np.mean(np.abs(t_val)) + eps)

        if s_cv > TAU_ACTIVE and t_cv < TAU_DEAD:
            err_msg = (
                f"Initialization Failure Detected (Dead Output).\n"
                f" - Source CV: {s_cv:.2e} (Active)\n"
                f" - Target CV: {t_cv:.2e} (Dead/Constant)"
            )
            return [{
                "case_id": "mech_check",
                "input_data": inputs,
                "root_cause": "Weight Initialization Mechanism",
                "error_type": "inconsistency",
                "error_message": err_msg
            }]

    except Exception as e:
        return [{
            "case_id": "mech_check",
            "input_data": inputs,
            "error_type": "crash",
            "root_cause": "Runtime Crash",
            "error_message": f"Crash during mechanism check: {str(e)}"
        }]

    if logger is not None:
        logger.info("[DiffTest-MechanismCheck] PASS.")

    return []


def _check_logic_consistency(
    source_model, target_model, test_inputs, sampled_weights, rtol, atol, logger,
    case_idx, numerical_checks=None,
):
    """
    Logic Precision Check.
    """
    try:
        # 1. Load Weight
        
        
        source_model.grad_search(None, None, False).apply_weights(source_model, sampled_weights)
        target_model.grad_search(None, None, False).apply_weights(target_model, sampled_weights)

        # 2. Transpose Inputs
        src_inputs = [source_model.to_tensor(x) for x in test_inputs]
        tgt_inputs = [target_model.to_tensor(x) for x in test_inputs]

        # 3. Execute Forward
        src_out_dict = execute_model_step(source_model, src_inputs)
        tgt_out_dict = execute_model_step(target_model, tgt_inputs)

        src_values = [np.asarray(value) for value in src_out_dict.values()]
        tgt_values = [np.asarray(value) for value in tgt_out_dict.values()]
        if len(src_values) != len(tgt_values):
            return [{
                "case_id": case_idx,
                "error_type": "output_structure_mismatch",
                "root_cause": "Output Structure Mismatch",
                "error_message": f"Output count differs: Source {len(src_values)} vs Target {len(tgt_values)}.",
            }]
        if not src_values:
            return [{
                "case_id": case_idx,
                "error_type": "output_structure_mismatch",
                "root_cause": "Missing Model Output",
                "error_message": "Both models returned zero outputs.",
            }]
        output_metrics = []
        for output_index, (src_val, tgt_val) in enumerate(zip(src_values, tgt_values)):
            if not np.isfinite(src_val).all() or not np.isfinite(tgt_val).all():
                return [{
                    "case_id": case_idx,
                    "error_type": "numerical_instability",
                    "root_cause": "NaN/Inf in Output",
                    "error_message": f"Output #{output_index} contains NaN or Inf.",
                }]
            if src_val.shape != tgt_val.shape:
                msg = (
                    f"Output #{output_index} shape mismatch: "
                    f"Source {src_val.shape} vs Target {tgt_val.shape}"
                )
                logger.error(f"[DiffTest-LogicCheck] {msg}")
                return [{
                    "case_id": case_idx,
                    "error_type": "output_shape_mismatch",
                    "root_cause": "Operator Attribute Error (Stride/Padding/Reshape)",
                    "error_message": msg,
                }]
            if src_val.dtype != tgt_val.dtype:
                return [{
                    "case_id": case_idx,
                    "error_type": "output_dtype_mismatch",
                    "root_cause": "Output Dtype Mismatch",
                    "error_message": (
                        f"Output #{output_index} dtype mismatch: "
                        f"Source {src_val.dtype} vs Target {tgt_val.dtype}"
                    ),
                }]

            src64 = src_val.astype(np.float64, copy=False)
            tgt64 = tgt_val.astype(np.float64, copy=False)
            diff = np.abs(src64 - tgt64)
            tolerance = atol + rtol * np.abs(src64)
            failed_elements = int(np.count_nonzero(diff > tolerance))
            output_metrics.append({
                "output_index": output_index,
                "shape": list(src_val.shape),
                "source_dtype": str(src_val.dtype),
                "target_dtype": str(tgt_val.dtype),
                "max_absolute_error": float(np.max(diff)) if diff.size else 0.0,
                "max_relative_error": (
                    float(np.max(diff / np.maximum(np.abs(src64), np.finfo(np.float64).tiny)))
                    if diff.size else 0.0
                ),
                "failed_elements": failed_elements,
                "element_count": int(diff.size),
            })

        metrics_record = {
            "case_index": int(case_idx),
            "outputs": output_metrics,
            "passed": all(item["failed_elements"] == 0 for item in output_metrics),
        }
        if isinstance(numerical_checks, list):
            numerical_checks.append(metrics_record)
        if not metrics_record["passed"]:
            max_diff = max(item["max_absolute_error"] for item in output_metrics)
            max_relative = max(item["max_relative_error"] for item in output_metrics)
            failed_elements = sum(item["failed_elements"] for item in output_metrics)
            element_count = sum(item["element_count"] for item in output_metrics)
            logger.warning(f"[DiffTest-LogicCheck] Case #{case_idx}: FAILED | Max Diff: {max_diff:.2e}")
            return [{
                "case_id": case_idx,
                "input_data": test_inputs,
                "source_output": src_out_dict,
                "target_output": tgt_out_dict,
                "error_type": "output_value_mismatch",
                "root_cause": "Operator Calculation Logic Error",
                "error_message": (
                    f"Numerical mismatch > atol={atol}, rtol={rtol}. "
                    f"Max absolute error: {max_diff:.6g}; max relative error: {max_relative:.6g}; "
                    f"failed elements: {failed_elements}/{element_count}."
                ),
            }]
        max_diff = max(item["max_absolute_error"] for item in output_metrics)
        logger.info(f"[DiffTest-LogicCheck] Case #{case_idx}: PASS (Max Diff: {max_diff:.2e})")
        return []

    except Exception as e:
        logger.error(f" [DiffTest-LogicCheck] Case #{case_idx} Crash: {str(e)}")
        return [{
            "case_id": case_idx,
            "input_data": test_inputs,
            "error_type": "runtime_crash",
            "root_cause": "Runtime Execution Failure",
            "error_message": str(e)
        }]

