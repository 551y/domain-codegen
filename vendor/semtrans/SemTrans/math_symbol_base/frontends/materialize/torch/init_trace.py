import types

import torch.nn.init
from SemTrans.math_symbol_base.frontends.abstract import InitEvent, BaseInitTracer


class InitTracer(BaseInitTracer):
    framework_name = 'pytorch'

    def __enter__(self):
        self._originals = {}
        target_module = torch.nn.init

        
        for name in dir(target_module):
            if name.startswith('_'): continue
            attr = getattr(target_module, name)
            if isinstance(attr, types.FunctionType):
                self._originals[name] = attr
                setattr(target_module, name, self._make_hook(name, attr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        target_module = torch.nn.init
        for name, original_func in self._originals.items():
            setattr(target_module, name, original_func)

    def _make_hook(self, name, original_func):
        def hook(*args, **kwargs):
            
            
            if self._processing:
                return original_func(*args, **kwargs)

            self._processing = True
            try:
                
                fname, line, code, is_explicit = self._get_call_origin()

                
                norm_args = self._normalize_args(name, args, kwargs)

                self.events.append(InitEvent(
                    framework='pytorch',
                    api_name=name,
                    args=[],
                    kwargs={},
                    filepath=fname,
                    lineno=line,
                    code_line=code,
                    normalized_args=norm_args,
                    is_explicit=is_explicit
                ))

                return original_func(*args, **kwargs)
            finally:
                self._processing = False

        return hook

    def _normalize_args(self, api_name, args, kwargs):
        params = {}
        api = api_name.lower()

        
        
        remaining_args = args[1:] if len(args) > 0 else []

        # === 1. Uniform (a, b) ===
        if 'uniform' in api and 'kaiming' not in api and 'xavier' not in api:
            # PyTorch sig: uniform_(tensor, a=0.0, b=1.0)
            
            params['a'] = str(remaining_args[0] if len(remaining_args) > 0 else kwargs.get('a', 0.0))
            params['b'] = str(remaining_args[1] if len(remaining_args) > 1 else kwargs.get('b', 1.0))

        # === 2. Normal (mean, std) ===
        elif 'normal' in api and 'kaiming' not in api:
            # PyTorch sig: normal_(tensor, mean=0.0, std=1.0)
            params['mean'] = str(remaining_args[0] if len(remaining_args) > 0 else kwargs.get('mean', 0.0))
            params['std'] = str(remaining_args[1] if len(remaining_args) > 1 else kwargs.get('std', 1.0))

        # === 3. Constant (val) ===
        elif 'constant' in api:
            # PyTorch sig: constant_(tensor, val)
            if len(remaining_args) > 0:
                params['val'] = str(remaining_args[0])
            else:
                params['val'] = str(kwargs.get('val', 0.0))

        # === 4. Kaiming (a, mode, nonlinearity) ===
        elif 'kaiming' in api:
            # PyTorch sig: kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
            params['a'] = str(kwargs.get('a', 0))
            params['mode'] = kwargs.get('mode', 'fan_in')  # Default
            params['nonlinearity'] = kwargs.get('nonlinearity', 'leaky_relu')

        # === 5. Xavier / Glorot (gain) ===
        elif 'xavier' in api or 'glorot' in api:
            # PyTorch sig: xavier_uniform_(tensor, gain=1.0)
            params['gain'] = str(kwargs.get('gain', 1.0))
            
            

        
        
        for k, v in kwargs.items():
            if k not in params:
                params[k] = str(v)

        return params