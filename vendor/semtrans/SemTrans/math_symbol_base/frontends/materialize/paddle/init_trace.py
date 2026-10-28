import paddle.nn.initializer

from SemTrans.math_symbol_base.frontends.abstract import InitEvent, BaseInitTracer


class InitTracer(BaseInitTracer):
    framework_name = 'paddle'

    def __enter__(self):
        self._originals = {}
        target_module = paddle.nn.initializer

        
        for name in dir(target_module):
            if name.startswith('_'): continue
            attr = getattr(target_module, name)
            if isinstance(attr, type):
                self._originals[name] = attr
                setattr(target_module, name, self._make_class_hook(name, attr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        target_module = paddle.nn.initializer
        for name, original_cls in self._originals.items():
            setattr(target_module, name, original_cls)

    def _make_class_hook(self, name, original_cls):
        tracer_self = self

        class HookedClass(original_cls):
            def __init__(self, *args, **kwargs):
                try:
                    fname, line, code, is_explicit = tracer_self._get_call_origin()
                    if fname != "tracer_internal":
                        norm_args = tracer_self._normalize_args(name, args, kwargs)

                        tracer_self.events.append(InitEvent(
                            framework='paddle',
                            api_name=name,
                            args=[],
                            kwargs={},
                            filepath=fname,
                            lineno=line,
                            code_line=code,
                            normalized_args=norm_args,
                            is_explicit=is_explicit
                        ))

                    super().__init__(*args, **kwargs)
                finally:
                    pass

        HookedClass.__name__ = original_cls.__name__
        return HookedClass

    def _normalize_args(self, api_name, args, kwargs):
        params = {}
        api = api_name.lower()

        # === 1. Uniform (low -> a, high -> b) ===
        if 'uniform' in api and 'kaiming' not in api and 'xavier' not in api:
            # Paddle: Uniform(low=0.0, high=1.0)
            
            params['a'] = str(kwargs.get('low', 0.0))
            params['b'] = str(kwargs.get('high', 1.0))

        # === 2. Normal (mean, std) ===
        elif 'normal' in api and 'kaiming' not in api:
            # Paddle: Normal(mean=0.0, std=1.0)
            params['mean'] = str(kwargs.get('mean', 0.0))
            params['std'] = str(kwargs.get('std', 1.0))

        # === 3. Constant (value -> val) ===
        elif 'constant' in api:
            # Paddle: Constant(value=0.0)
            val = kwargs.get('value', None)
            if val is None and len(args) > 0: val = args[0]
            params['val'] = str(val if val is not None else 0.0)

        
        elif 'kaiming' in api:
            
            
            
            
            params['mode'] = 'fan_in'
            
            

        
        elif 'xavier' in api:
            
            params['mode'] = 'fan_avg'

            
        for k, v in kwargs.items():
            
            if k in ['low', 'high', 'value']:
                continue
            if k not in params:
                params[k] = str(v)

        return params