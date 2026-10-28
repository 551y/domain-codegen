import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

import paddle

from SemTrans.math_symbol_base.frontends.abstract import DType
from SemTrans.testing_and_repair.util import StructureMismatchError
from .paddlenet import SymbolNet


class InputSearchBase(ABC):
    @staticmethod
    def apply_weights(net, weight_sample):
        with paddle.no_grad():
            for name, param in net.named_parameters():
                if name not in weight_sample:
                    continue

                val = weight_sample[name]
                if isinstance(val, paddle.Tensor):
                    val = val.numpy()

                param_shape = list(param.shape)
                val_shape = list(val.shape)

                if len(param_shape) == 2 and param_shape == list(val.T.shape):
                    val = val.T

                elif len(param_shape) == 4 and len(val_shape) == 4:
                    if param_shape != val_shape:
                        val_transposed = val.transpose(1, 0, 2, 3)
                        if param_shape == list(val_transposed.shape):
                            val = val_transposed
                        else:
                            raise StructureMismatchError(name, val_shape, param_shape)

                if list(param.shape) != list(val.shape):
                    raise StructureMismatchError(name, val_shape, param_shape)

                param.set_value(paddle.to_tensor(val))

    def __init__(
            self, net: SymbolNet, start_inputs=None, start_weights=None, use_cuda=False
    ):
        self.net = net
        self.start_inputs = start_inputs
        self.start_weights = start_weights
        self.use_cuda = use_cuda

    @abstractmethod
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, paddle.Tensor]:
        pass

    def search(
            self,
            max_time_ms: int = None,
            max_sample: int = 1,
            target_count: int = 50,
            logger=None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        [Paddle] Batch search for valid inputs AND weights.
        """
        n_try = 0
        valid_inputs_list = []
        start_time = time.time()
        DEFAULT_SINGLE_TIMEOUT = 10000

        if logger:
            logger.info(f"[Search] Start: Target={target_count}, MaxTry={max_sample}, TimeLimit={max_time_ms}ms")

        while n_try < max_sample:
            elapsed_ms = (time.time() - start_time) * 1000
            if max_time_ms is not None and elapsed_ms > max_time_ms:
                if logger: logger.info(f"[Search] Time limit reached ({elapsed_ms:.0f}ms). Stopping.")
                break

            current_weights = {}

            if self.start_weights is not None and n_try < len(self.start_weights):
                current_weights = self.start_weights[n_try]
            else:
                for name, param in self.net.named_parameters():
                    current_weights[name] = self.net.get_weight_tensor(
                        param.shape, dtype=param.dtype, use_gpu=self.use_cuda
                    )

            self.apply_weights(self.net, current_weights)

            if self.start_inputs is not None and n_try < len(self.start_inputs):
                cur_input = self.start_inputs[n_try]
            else:
                cur_input = self.net.get_random_inps(use_cuda=self.use_cuda)

            single_run_timeout = DEFAULT_SINGLE_TIMEOUT
            if max_time_ms is not None:
                remaining = int(max_time_ms - elapsed_ms)
                if remaining < single_run_timeout:
                    single_run_timeout = max(100, remaining)

            try:
                res = self.search_one(cur_input, single_run_timeout)
            except Exception as e:
                if logger: logger.error(f"[Search] Warning: Iteration {n_try} failed with error: {e}")
                res = None

            n_try += 1

            if res is not None:
                valid_inputs_list.append({
                    "inputs": res,
                    "weights": current_weights
                })

            if len(valid_inputs_list) >= target_count:
                break

        return n_try, valid_inputs_list


class SamplingSearch(InputSearchBase):
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, paddle.Tensor]:
        with paddle.no_grad():
            self.net.check_intermediate_numeric = True
            _ = self.net(*start_inp.values())
            if not self.net.invalid_found_last:
                return start_inp
            return None


class GradSearch(InputSearchBase):
    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, paddle.Tensor]:
        timeout_s = None if timeout_ms is None else timeout_ms / 1000
        return self.net.grad_input_gen(
            init_tensors=start_inp, use_cuda=self.use_cuda, max_time=timeout_s
        )


class PracticalHybridSearch(InputSearchBase):
    def __init__(
            self, net: SymbolNet, start_inputs=None, start_weights=None, use_cuda=False
    ):
        super().__init__(net, start_inputs, start_weights, use_cuda)

        self.differentiable = None

        if all([DType.is_float(v.dtype) for _, v in self.net.input_like.items()]):
            diff_test_inp = self.net.get_random_inps(use_cuda=self.use_cuda)
            for _, item in diff_test_inp.items():
                item.stop_gradient = False
            self.net.forward(*diff_test_inp.values())
            self.differentiable = self.net.differentiable
        else:
            self.differentiable = False

    def search_one(self, start_inp, timeout_ms: int = None) -> Dict[str, paddle.Tensor]:
        if self.differentiable:
            return GradSearch.search_one(start_inp, timeout_ms)
        else:
            return SamplingSearch.search_one(start_inp, timeout_ms)
