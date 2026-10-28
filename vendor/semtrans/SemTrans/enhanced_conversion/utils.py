import ast, os
import pprint


def save_cache_to_config(file_path, variable_name, new_data):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        source_code = "".join(source_lines)

        tree = ast.parse(source_code)
        target_node = None

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        target_node = node
                        break
            if target_node:
                break

        if not target_node:
            print(f"[Warning] Variable '{variable_name}' not found in {file_path}. Appending to end.")
            formatted_data = pprint.pformat(new_data, indent=4, width=120)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{variable_name} = {formatted_data}\n")
            return

        new_code_str = pprint.pformat(new_data, indent=4, width=120)

        new_assignment = f"{variable_name} = {new_code_str}\n"

        start_line = target_node.lineno - 1
        end_line = target_node.end_lineno

        source_lines[start_line:end_line] = [new_assignment]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(source_lines)

        print(f"[System] Cache successfully saved to {file_path}")

    except Exception as e:
        print(f"[Error] Failed to update config file: {e}")


def get_all_python_files(source_path):
    if os.path.isfile(source_path):
        return [source_path]
    elif os.path.isdir(source_path):
        files = []
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
        return files
    else:
        
        return [source_path]


def get_model_code_content(file_path):
    """
    Read the model file and return the code string.
    - .py: Return source code
    - .pt/.pth: Return TorchScript decompiled code
    - .onnx/.pdmodel: Return generated pseudo-code description
    """
    ext = os.path.splitext(file_path)[-1].lower()

    # 1. Python Source: Read file directly
    if ext == '.py':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # 2. TorchScript: Leverage official decompilation capabilities
    elif ext in ['.pt', '.pth']:
        try:
            import torch
            
            model = torch.jit.load(file_path, map_location='cpu')
            return model.code  # Python-style code automatically decompiled by Torch
        except Exception as e:
            return f"# Error loading TorchScript: {e}"

    # 3. Paddle Inference: Generate static graph pseudo-code
    elif ext == '.pdmodel':
        try:
            import paddle
            paddle.enable_static()
            prefix = file_path.replace(".pdmodel", "")

            exe = paddle.static.Executor(paddle.CPUPlace())
            # load model structure
            program, _, _ = paddle.static.load_inference_model(prefix, exe)

            lines = ["# Paddle Static Program", "def forward():"]

            for block in program.blocks:
                for op in block.ops:
                    # Format: output = OpType(input, attrs)
                    input_args = ", ".join(op.input_arg_names)
                    output_args = ", ".join(op.output_arg_names)

                    # Extract key attributes, filtering out internal attributes
                    attrs = []
                    for name in op.attr_names:
                        if name not in ['op_callstack', 'op_namescope', 'op_role', 'use_mkldnn']:
                            try:
                                val = op.attr(name)
                                # Filter out excessively long binary attributes
                                if isinstance(val, (int, float, bool, str)):
                                    attrs.append(f"{name}={val}")
                            except:
                                pass

                    lines.append(f"{output_args} = {op.type}({input_args}, {', '.join(attrs)})")

            paddle.disable_static()
            return "\n".join(lines)
        except Exception as e:
            return f"# Error loading Paddle model: {e}"

    else:
        return f"# Unsupported file extension: {ext}"

