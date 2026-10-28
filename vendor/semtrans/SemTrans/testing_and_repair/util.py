import math, inspect, re,  os
import shutil, psutil, ast
from typing import Any, List
import numpy as np
import subprocess, tempfile

from SemTrans.math_symbol_base.frontends.abstract import InitEvent


_DOT_EXIST = shutil.which("dot") is not None
_CONDA_EXIST = shutil.which("conda") is not None
_APT_EXIST = shutil.which("apt") is not None
_BREW_EXIST = shutil.which("brew") is not None

_CALL_ONCE = False


def detect_framework_type(model: Any) -> str:
    # Relay conversion commonly carries ``(IRModule, params)``.  Inspect the
    # module rather than forcing callers to unwrap a valid TVM artifact.
    probe = model[0] if isinstance(model, tuple) and model else model
    probe_module = type(probe).__module__
    if probe_module.startswith('tvm'):
        return 'tvm'

    model_class = model if isinstance(model, type) else type(model)
    for base in model_class.__mro__:
        module_name = base.__module__

        if module_name == 'builtins':
            continue

        if module_name.startswith('torch'):
            return 'pytorch'

        if module_name.startswith('paddle'):
            return 'paddle'

        if module_name.startswith('tensorflow') or module_name.startswith('keras'):
            return 'tensorflow'

        if module_name.startswith('mindspore'):
            return 'mindspore'

    class_str = f"{model_class.__module__}.{model_class.__name__}"
    raise ValueError(
        f"Unsupported framework or custom class without standard inheritance: {class_str}.\n"
        "Ensure your model inherits from standard base classes (e.g., torch.nn.Module, paddle.nn.Layer)."
    )


def get_memory_capacity(model):
    device_info = str(getattr(model, 'device', 'cpu')).lower()

    if 'cuda' in device_info or 'gpu' in device_info:
        # 1. Parse GPU ID
        gpu_index = 0
        if ':' in device_info:
            try:
                gpu_index = int(device_info.split(':')[-1].strip())
            except ValueError:
                gpu_index = 0

        # 2. Query memory
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.total',
                '--format=csv,noheader,nounits',
                '-i', str(gpu_index)
            ], encoding='utf-8')
            return int(output.strip()) * 1024 * 1024
        except Exception as e:
            print(f"[Warning] Failed to query GPU-{gpu_index} memory: {e}")
            return psutil.virtual_memory().total

    return psutil.virtual_memory().total


def convert_sat_inputs_to_numpy(sat_inputs):
    sat_inputs_numpy = {}

    for key, tensor in sat_inputs.items():
        if tensor.place.is_gpu_place():
            tensor = tensor.cpu()
        sat_inputs_numpy[key] = tensor.numpy()

    return sat_inputs_numpy


def distinct_dicts(dict_list):
    seen = set()
    result = []

    for d in dict_list:
        hashable_items = []

        for k, v in sorted(d.items()):
            if isinstance(v, np.ndarray):
                val_signature = (v.tobytes(), v.shape, str(v.dtype))
            elif isinstance(v, list):
                val_signature = tuple(v)
            else:
                val_signature = v

            hashable_items.append((k, val_signature))

        # Convert the list to a tuple to make it hashable
        dict_signature = tuple(hashable_items)

        if dict_signature not in seen:
            seen.add(dict_signature)
            result.append(d)

    return result


class StructureMismatchError(Exception):
    def __init__(self, param_name, src_shape, tgt_shape):
        self.message = f"Shape mismatch for '{param_name}': Source {src_shape} vs Target {tgt_shape}"
        super().__init__(self.message)


def get_static_analysis_report(code_content):
    if not code_content or len(code_content.strip()) == 0:
        return "No code content to analyze."

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write(code_content)
        tmp_path = tmp.name

    try:
        cmd = [
            "pylint",
            "--disable=all",
            "--enable=E,F",
            "--score=n",
            "--msg-template='Line {line}: {msg} ({symbol})'",
            tmp_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        report = result.stdout.strip()

        if not report:
            return "No static syntax/structural errors detected."

        return report

    except Exception as e:
        return f"Static analysis failed to run: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def compare_traces(src_events: List['InitEvent'], tgt_events: List['InitEvent']):
    def _guess_family(api_name: str) -> str:
        name = api_name.lower()
        if 'kaiming' in name: return 'kaiming'
        if 'xavier' in name or 'glorot' in name: return 'xavier'
        if 'constant' in name: return 'constant'
        if 'zeros' in name: return 'zeros'
        if 'ones' in name: return 'ones'
        if 'normal' in name and 'kaiming' not in name and 'xavier' not in name: return 'normal'
        if 'uniform' in name and 'kaiming' not in name and 'xavier' not in name: return 'uniform'
        return 'unknown'

    def _is_uniform(api_name: str) -> bool:
        return 'uniform' in api_name.lower()

    def _fuzzy_equal(val1, val2, param_name):
        if val1 is None: val1 = "N/A"
        if val2 is None: val2 = "N/A"

        v1_str = str(val1).lower()
        v2_str = str(val2).lower()

        if v1_str == v2_str: return True

        if param_name == 'nonlinearity':
            if 'leaky' in v1_str and 'leaky' in v2_str: return True
            if 'relu' in v1_str and 'relu' in v2_str and 'leaky' not in v1_str and 'leaky' not in v2_str: return True

        try:
            f1 = float(val1)
            f2 = float(val2)
            return math.isclose(f1, f2, rel_tol=1e-4)
        except ValueError:
            pass

        return False

    issues_found = []

    min_len = min(len(src_events), len(tgt_events))
    if len(src_events) != len(tgt_events):
        issues_found.append({
            "id": -1,
            "reason": "Event Count Mismatch",
            "detail": f"Source has {len(src_events)} events, Target has {len(tgt_events)}. Truncating to {min_len}.",
            "severity": "WARNING"
        })

    for i in range(min_len):
        src = src_events[i]
        tgt = tgt_events[i]

        s_args = src.normalized_args
        t_args = tgt.normalized_args

        s_fam = _guess_family(src.api_name)
        t_fam = _guess_family(tgt.api_name)

        if s_fam != t_fam and s_fam != 'unknown' and t_fam != 'unknown':
            if not (s_fam == 'zeros' and t_fam == 'constant') and\
                    not (s_fam == 'constant' and t_fam == 'zeros'):
                issues_found.append({
                    "id": i,
                    "reason": "Algorithm Family Mismatch",
                    "detail": f"Source uses {s_fam} ({src.api_name}), Target uses {t_fam} ({tgt.api_name})",
                    "src_event": src, "tgt_event": tgt, "severity": "CRITICAL"
                })
                continue

        if _is_uniform(src.api_name) != _is_uniform(tgt.api_name):
            if 'constant' not in s_fam and 'zeros' not in s_fam and 'ones' not in s_fam:
                issues_found.append({
                    "id": i,
                    "reason": "Distribution Mismatch",
                    "detail": f"One is Uniform, the other is Normal/Other.",
                    "src_event": src, "tgt_event": tgt, "severity": "HIGH"
                })

        keys_to_compare = ['mode', 'nonlinearity', 'a', 'mean', 'std', 'val']

        for k in keys_to_compare:
            s_val = s_args.get(k)
            t_val = t_args.get(k)

            if s_val is None and t_val is None:
                continue

            if not _fuzzy_equal(s_val, t_val, k):
                issues_found.append({
                    "id": i,
                    "reason": f"Param '{k}' Mismatch",
                    "detail": f"Source: {s_val} vs Target: {t_val}",
                    "src_event": src, "tgt_event": tgt, "severity": "MEDIUM"
                })

    return issues_found


def diagnose_initialization_mismatch(
        source_symbolnet,
        target_symbolnet,
        src_loader,
        tgt_loader,
        logger
):
    """
    [Final Revised Logic]
    Smart Reporting with Noise Filtering and Non-Blocking Suggestions.
    """
    from collections import defaultdict
    import os

    # --- Helper Functions (Internal) ---

    def _get_code_semantic(code_line):
        if not code_line: return 'UNKNOWN'
        code = code_line.strip()
        
        if 'nn.init.' in code or 'Op(' in code or code.startswith('init.') or code.endswith('_'):
            return 'EXPLICIT_OP'
        
        if '= nn.' in code or 'add_module' in code:
            return 'LAYER_DEF'
        return 'UNKNOWN'

    def _get_layer_type(code_line):
        if not code_line: return 'Unknown'
        code = code_line.lower()
        if 'conv' in code: return 'Conv'
        if 'linear' in code or 'fc' in code or 'classifier' in code: return 'Linear'
        if 'norm' in code or 'bn' in code: return 'Norm'
        return 'Unknown'

    def _classify_init_api(api_name):
        name = api_name.lower()
        if any(k in name for k in ['constant', 'zeros', 'ones', 'fill']): return 'CONST'
        if any(k in name for k in ['kaiming', 'xavier', 'normal', 'uniform', 'orthogonal', 'glorot']): return 'DIST'
        return 'UNKNOWN'

    def _compare_init_events(src_events, tgt_events):
        mismatch_groups = defaultdict(list)

        i = 0
        j = 0

        while i < len(src_events):
            src = src_events[i]

            
            if j >= len(tgt_events):
                key = ("Target Missing", f"Source has explicit {src.api_name}, but Target stream ended.")
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                i += 1
                continue

            tgt = tgt_events[j]

            
            src_semantic = _get_code_semantic(src.code_line)
            tgt_semantic = _get_code_semantic(tgt.code_line)

            
            
            if src_semantic == 'EXPLICIT_OP' and tgt_semantic == 'LAYER_DEF':
                
                key = ("Implicit/Lazy Init", f"Source uses explicit {src.api_name}, Target lacks corresponding init.")

                
                
                mismatch_groups[key].append({'role': 'Source', 'event': src})

                i += 1
                
                continue

            
            if src_semantic == 'LAYER_DEF' and tgt_semantic == 'EXPLICIT_OP':
                key = ("Structure Mismatch", f"Source defining layer, Target executing Op. Sync lost.")
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                i += 1;
                continue

            
            src_layer = _get_layer_type(src.code_line)
            tgt_layer = _get_layer_type(tgt.code_line)

            if src_layer != 'Unknown' and tgt_layer != 'Unknown' and src_layer != tgt_layer:
                key = ("Layer Mismatch", f"Source is {src_layer}, Target is {tgt_layer}.")
                
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                mismatch_groups[key].append({'role': 'Target (Context)', 'event': tgt})
                i += 1;
                continue  

            
            src_api_type = _classify_init_api(src.api_name)
            tgt_api_type = _classify_init_api(tgt.api_name)

            
            is_ones = (src.api_name == 'ones_' and 'constant' in tgt.api_name.lower())
            is_zeros = (src.api_name == 'zeros_' and 'constant' in tgt.api_name.lower())
            if is_ones or is_zeros: i += 1; j += 1; continue

            # Lazy Init check
            if src_api_type == 'DIST' and tgt_api_type == 'CONST':
                key = ("Implicit/Lazy Init", f"Source uses {src.api_name}, Target uses Implicit/Bias Init.")
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                mismatch_groups[key].append({'role': 'Target (Context)', 'event': tgt})
                i += 1;
                continue

                # Value check
            s_name = src.api_name.lower()
            t_name = tgt.api_name.lower()
            is_compatible = (s_name in t_name) or (t_name in s_name) or\
                            ('xavier' in s_name and 'glorot' in t_name) or\
                            ('glorot' in s_name and 'xavier' in t_name)

            param_diffs = []
            for k, v in src.normalized_args.items():
                tgt_v = tgt.normalized_args.get(k, 'N/A')
                if v != tgt_v: param_diffs.append(f"{k}:({v}!={tgt_v})")

            if not is_compatible:
                key = ("Strategy Mismatch", f"Source: {src.api_name} vs Target: {tgt.api_name}")
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                mismatch_groups[key].append({'role': 'Target', 'event': tgt})
                i += 1;
                j += 1
            elif param_diffs:
                key = ("Value Mismatch", f"Args differ: {', '.join(param_diffs)}")
                mismatch_groups[key].append({'role': 'Source', 'event': src})
                mismatch_groups[key].append({'role': 'Target', 'event': tgt})
                i += 1;
                j += 1
            else:
                i += 1;
                j += 1

        
        if not mismatch_groups:
            return None

        report = ["[INITIALIZATION DIAGNOSIS REPORT]"]
        report.append("=" * 80)

        for (err_type, desc), entries in mismatch_groups.items():
            src_entries = [e for e in entries if 'Source' in e['role']]
            unique_src_lines = sorted(list(set(e['event'].lineno for e in src_entries)))

            report.append(f"TYPE: [{err_type}]")
            report.append(f"DESCRIPTION: {desc}")
            report.append(f"STATS: {len(src_entries)} events across {len(unique_src_lines)} source locations.")
            report.append("CODE LOCATIONS (Sampled):")

            count = 0
            for entry in entries[:8]:
                evt = entry['event']
                role = entry['role']
                if 'Source' in role: count += 1
                if count > 5: break

                label = f"[{role}]".ljust(18)
                fname = os.path.basename(evt.filepath)
                code = evt.code_line.strip() if evt.code_line else "<N/A>"
                if len(code) > 80: code = code[:80] + "..."

                report.append(f"  {label} File: {fname}:{evt.lineno}")
                report.append(f"     Code: {code}")

            if len(unique_src_lines) > 5:
                report.append(f"     ... (+{len(unique_src_lines) - 5} other source locations)")
            report.append("-" * 40)

        report.append("\n[DIAGNOSIS SUMMARY]")
        if any("Implicit" in k[0] for k in mismatch_groups):
            report.append(">> STATUS: IMPLICIT INITIALIZATION DETECTED")
            report.append(
                ">> INFO:   Target model relies on Default/Lazy initialization, while Source uses Explicit strategies.")
            report.append(">> RISK:   Low for Inference (weights loaded). Medium for Fine-tuning (convergence).")
            report.append(
                ">> ACTION: No action needed for conversion/inference. For training from scratch, align init strategies.")
        else:
            report.append(">> STATUS: MISMATCH DETECTED (See details above)")

        return "\n".join(report)

    # --- Main Execution Flow ---
    src_events = []
    tgt_events = []

    # 1. Trace Source
    try:
        src_tracer = source_symbolnet.init_trace()
        with src_tracer:
            src_loader()
            
            
            src_events = [e for e in src_tracer.events if getattr(e, 'is_explicit', True)]
    except Exception as e:
        logger.warning(f"Source Trace Failed: {e}")

    # 2. Trace Target
    try:
        tgt_tracer = target_symbolnet.init_trace()
        with tgt_tracer:
            tgt_loader()
            tgt_events = [e for e in tgt_tracer.events if getattr(e, 'is_explicit', True)]
    except Exception as e:
        logger.warning(f"Target Trace Failed: {e}")

    if not src_events and not tgt_events:
        return None

    # 3. Compare
    return _compare_init_events(src_events, tgt_events)


def parse_layer_from_log(error_msg):
    match = re.search(r"(?:in layer|Layer:)[\s]*([a-zA-Z0-9_\.]+)", error_msg, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


class SourceLocator:
    def __init__(self, model_instance):
        self.model = model_instance
        self.file_path = "Unknown"
        self.start_line = 0
        self.attr_map = {}
        self._parse_source()

    def _parse_source(self):
        try:
            cls = self.model.__class__
            self.file_path = inspect.getsourcefile(cls)
            if not self.file_path: return
            lines, self.start_line = inspect.getsourcelines(cls)
            source_code = "".join(lines)

            tree = ast.parse(source_code)

            def record_assignment(target, value, owner_node):
                if isinstance(target, ast.Attribute) and \
                        isinstance(target.value, ast.Name) and \
                        target.value.id == 'self':
                    attr_name = target.attr
                    self.attr_map[attr_name] = {
                        'node': owner_node,
                        'is_seq': self._is_sequential(value),
                        'args': value.args if isinstance(value, ast.Call) else []
                    }

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        record_assignment(target, node.value, node)
                elif isinstance(node, ast.AnnAssign):
                    record_assignment(node.target, node.value, node)
        except Exception:
            pass

    def _is_sequential(self, node):
        if not isinstance(node, ast.Call): return False
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        return 'Sequential' in name

    def _resolve_attr_info(self, name):
        return self.attr_map.get(name)

    def _format_location(self, lineno, suffix=None):
        abs_line = self.start_line + lineno - 1
        base = f"{os.path.basename(self.file_path)}:{abs_line}"
        if suffix:
            return f"{base} ({suffix})"
        return base

    def get_location(self, param_name):
        if not self.file_path or self.file_path == "Unknown": return "Source N/A"

        parts = param_name.split('.')
        root = parts[0]
        info = self._resolve_attr_info(root)

        if not info: return "Unknown Loc"

        target_lineno = info['node'].lineno

        if info['is_seq'] and len(parts) > 1 and parts[1].isdigit():
            idx = int(parts[1])
            if idx < len(info['args']):
                target_lineno = info['args'][idx].lineno
                return self._format_location(target_lineno, f"{root}[{idx}]")

        if len(parts) > 2 and parts[1] == 'layers' and parts[2].isdigit():
            layer_idx = int(parts[2])
            return self._format_location(target_lineno, f"{root}.layers[{layer_idx}]")

        return self._format_location(target_lineno)

    def get_module_location(self, module_path):
        if not self.file_path or self.file_path == "Unknown":
            return "Source N/A"
        if not module_path:
            return self._format_location(1)

        parts = module_path.split('.')
        root = parts[0]
        info = self._resolve_attr_info(root)
        if not info:
            return "Unknown Loc"

        target_lineno = info['node'].lineno

        if info['is_seq'] and len(parts) > 1 and parts[1].isdigit():
            idx = int(parts[1])
            if idx < len(info['args']):
                target_lineno = info['args'][idx].lineno
                return self._format_location(target_lineno, f"{root}[{idx}]")

        rendered_parts = []
        i = 0
        has_layer_index = False
        while i < len(parts):
            if parts[i] == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                rendered_parts.append(f"layers[{int(parts[i + 1])}]")
                has_layer_index = True
                i += 2
                continue
            rendered_parts.append(parts[i])
            i += 1

        if has_layer_index:
            return self._format_location(target_lineno, ".".join(rendered_parts))

        return self._format_location(target_lineno)
