# pylint: disable=unused-argument
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_gather)
from vllm.distributed.utils import divide
from vllm.lora.paged.sgmv import add_lora_sgmv_cutlass
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

if TYPE_CHECKING:
    pass


def _get_lora_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/vllm/lora/layers.py#L34
    """Returns the device for where to place the LoRA tensors."""
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # GPTQ/AWQ/SqueezeLLM
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # marlin
    elif hasattr(base_layer, "B"):
        return base_layer.B.device
    else:
        raise ValueError(f"Unsupported base layer: {base_layer}")


def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop('decorate') if 'decorate' in kwargs else True
        condition = (not kwargs['lora_config'].fully_sharded_loras
                     if decorate else True)
        return can_replace(*args, **kwargs) and condition

    return dec


def _apply_lora(
    x: torch.Tensor,
    lora_a_ptr: torch.Tensor,
    lora_b_ptr: torch.Tensor,
    indices: torch.Tensor,
    rank: int,
    output: torch.Tensor,
):
    """Applies lora to each input.

    This method applies all loras to each input. Here indices is a list of tuple (start_idx[i], lora_idx[i]), representing that the lora_ptr[lora_idx[i]] should be applied to the input token[start_idx[i]:start_idx[i+1]].

    Input shapes:
        x:               (batch_size, hidden_dim)
        lora_a_ptr:      (max_loras, )
            DType: torch.int64
            The matrix it points to has shape (input_dim, lora_rank)
        lora_b_ptr:      (max_loras, )
            DType: torch.int64
            The matrix it points to has shape (lora_rank, output_dim)
        indices:         (num_lora_seqs + 1, )
        output:          (batch_size, output_dim)
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    wa_ptr = [None for _ in range(indices.size(0) - 1)]
    wb_ptr = [None for _ in range(indices.size(0) - 1)]
    s = [0 for _ in range(indices.size(0))]

    # TODO: Parallelize this loop
    for i in range(indices.size(0) - 1):
        start_idx, lora_idx = indices[i]
        end_idx = indices[i + 1][0]
        assert start_idx < end_idx, f"Start index {start_idx} is greater than end index {end_idx}"
        assert lora_a_ptr[lora_idx] is not None, f"lora_a_ptr[{lora_idx}] is None"
        assert lora_b_ptr[lora_idx] is not None, f"lora_b_ptr[{lora_idx}] is None"
        wa_ptr[i] = lora_a_ptr[lora_idx]
        wb_ptr[i] = lora_b_ptr[lora_idx]
        s[i + 1] = end_idx
        
    wa_ptr = torch.tensor(wa_ptr, device="cuda:0", dtype=torch.int64)
    wb_ptr = torch.tensor(wb_ptr, device="cuda:0", dtype=torch.int64)
    add_lora_sgmv_cutlass(output, x, wa_ptr, wb_ptr, s, 0, rank)
    return output.view_as(org_output)

def _apply_original_lora(
    x: torch.Tensor,
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    """Applies lora to each input.

    This method applies all loras to each input. It uses the
    indices vector to determine which lora yields the
    correct output. An index of -1 means no lora should be
    applied. This method adds the final lora results to the
    output.

    Input shapes:
        x:               (batch_size, hidden_dim)
        lora_a_stacked:  (num_loras, lora_rank, hidden_dim)
        lora_b_stacked:  (num_loras, output_dim, lora_rank)
        indices:         (batch_size)
        output:          (batch_size, output_dim)
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    
    # output_std = output.clone()
    # print(x.sum(), output.sum())
    # add_lora(output, x, lora_a_stacked, lora_b_stacked, indices, 0, 1.0)
    # print(output.sum())
    # return output
    
    # x = x.to(torch.float32)
    
    # import gc
    # gc.disable()
    
    lora_a_stacked_transposed = [lora_a_stacked[i].to(dtype=x.dtype).view(-1, lora_a_stacked.shape[-1]).T.contiguous() for i in range(lora_a_stacked.shape[0])]
    lora_b_stacked_transposed = [lora_b_stacked[i].to(dtype=x.dtype).view(-1, lora_b_stacked.shape[-1]).T.contiguous() for i in range(lora_b_stacked.shape[0])]
    lora_a_stacked_transposed_fp32 = [lora_a_stacked[i].to(dtype=torch.float32).view(-1, lora_a_stacked.shape[-1]).T.contiguous() for i in range(lora_a_stacked.shape[0])]
    lora_b_stacked_transposed_fp32 = [lora_b_stacked[i].to(dtype=torch.float32).view(-1, lora_b_stacked.shape[-1]).T.contiguous() for i in range(lora_b_stacked.shape[0])]


    # x = x.to(torch.float32)
    empty_lora_a = torch.zeros_like(lora_a_stacked_transposed[0], dtype=x.dtype, device="cuda:0")
    empty_lora_b = torch.zeros_like(lora_b_stacked_transposed[0], dtype=x.dtype, device="cuda:0")
    empty_lora_a_fp32 = torch.zeros_like(lora_a_stacked_transposed[0], dtype=torch.float32, device="cuda:0")
    empty_lora_b_fp32 = torch.zeros_like(lora_b_stacked_transposed[0], dtype=torch.float32, device="cuda:0")
    
    wa_ptr = []
    wb_ptr = []
    wa_ptr_fp32 = []
    wb_ptr_fp32 = []
    s = []

    for i in range(indices.shape[0]):
        if i == 0 or indices[i] != indices[i - 1]:
            if indices[i] == -1:
                wa_ptr.append(empty_lora_a.data_ptr())
                wb_ptr.append(empty_lora_b.data_ptr())
                wa_ptr_fp32.append(empty_lora_a_fp32.data_ptr())
                wb_ptr_fp32.append(empty_lora_b_fp32.data_ptr())
            else:
                wa_ptr.append(lora_a_stacked_transposed[indices[i]].data_ptr())
                wb_ptr.append(lora_b_stacked_transposed[indices[i]].data_ptr())
                wa_ptr_fp32.append(lora_a_stacked_transposed_fp32[indices[i]].data_ptr())
                wb_ptr_fp32.append(lora_b_stacked_transposed_fp32[indices[i]].data_ptr())
            s.append(i)
    s.append(indices.shape[0])
    
    wa_ptr = torch.tensor(wa_ptr, device="cuda:0", dtype=torch.int64)
    wb_ptr = torch.tensor(wb_ptr, device="cuda:0", dtype=torch.int64)
    wa_ptr_fp32 = torch.tensor(wa_ptr_fp32, device="cuda:0", dtype=torch.int64)
    wb_ptr_fp32 = torch.tensor(wb_ptr_fp32, device="cuda:0", dtype=torch.int64)
    s = torch.tensor(s, device="cuda:0", dtype=torch.int32)
    
    from vllm.lora.paged.sgmv import add_lora_sgmv_cutlass, add_lora_sgmv_cutlass_custom, add_lora_sgmv_cutlass_fp32
    output_tmp = output.clone()
    # output_fp32 = output.clone()
    # print(x.sum(), output.sum())
    add_lora_sgmv_cutlass(output, x, wa_ptr, wb_ptr, s, 0, empty_lora_a.shape[-1])
    # 25.93, 25.72
    # add_lora_sgmv_cutlass_custom(output_tmp, x, wa_ptr, wb_ptr_fp32, s, 0, empty_lora_a.shape[-1])
    # add_lora_sgmv_cutlass_fp32(output_fp32, x, wa_ptr_fp32, wb_ptr_fp32, s, 0, empty_lora_a.shape[-1])
    
    # print(output.sum())
    # gc.enable()
    # output.set_(output_fp32)
    # if torch.isnan(output.sum()):
        # exit(0)
    
    # assert torch.allclose(output, output_std, rtol=1e-2, atol=1e-3)
    return output.view_as(org_output)


@dataclass
class LoRAMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


class BaseLayerWithPagedLoRA(nn.Module):
    def slice_lora_a(
        self, lora_a: Union[torch.Tensor, List[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, None]]]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    def slice_lora_b(
        self, lora_b: Union[torch.Tensor, List[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, None]]]:
        """Slice lora b if splitting with tensor parallelism."""
        ...
    
    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.int64,
        lora_b: torch.int64,
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        indices_len: int, 
    ):
        """Sets the mapping indices."""
        ...

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError



class ColumnParallelLinearWithPagedLoRA(BaseLayerWithPagedLoRA):
    """
    LoRA on top of ColumnParallelLinear layer.
    """

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size = self.base_layer.input_size
        self.output_size = self.base_layer.output_size_per_partition
        self.device = _get_lora_device(self.base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_config = lora_config
        self.tp_size = get_tensor_model_parallel_world_size()
        if lora_config.fully_sharded_loras:
            raise ValueError(
                "Current implementation does not support fully sharded loras."
            )
        # self.lora_a_stacked = torch.zeros(
        #     max_loras,
        #     1,
        #     lora_config_max_lora_rank.
        #     self.input_size,
        #     dtype=lora_config.lora_dtype,
        #     device=self.device,
        # )
        self.lora_a_ptr = [ None for _ in range(max_loras) ]
        # self.lora_b_stacked = torch.zeros(
        #     max_loras,
        #     1,
        #     self.output_size,
        #     lora_config.max_lora_rank,
        #     dtype=lora_config.lora_dtype,
        #     device=self.device,
        # )
        self.lora_b_ptr = [ None for _ in range(max_loras) ]
        self.output_dim = self.output_size

        # lazily initialized.
        self.indices: torch.Tensor
        self.indices_len: int

    def reset_lora(self, index: int):
        self.lora_a_ptr[index] = None
        self.lora_b_ptr[index] = None

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_lora(
        self,
        index: int,
        lora_a: int,
        lora_b: int,
    ):
        self.reset_lora(index)
        self.lora_a_ptr[index] = lora_a
        self.lora_b_ptr[index] = lora_b


    def set_mapping(
        self,
        base_indices: torch.Tensor,
        indices_len: int
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        assert self.indices is not None, "Indices not set."
        _apply_lora(
            x,
            self.lora_a_ptr,
            self.lora_b_ptr,
            self.indices[:self.indices_len],
            self.lora_config.max_lora_rank,
            output,
        )
        return output

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1)

class RowParallelLinearWithPagedLoRA(BaseLayerWithPagedLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size
        self.device = _get_lora_device(self.base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_config = lora_config
        self.tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.lora_a_ptr = [ None for _ in range(max_loras) ]
        self.lora_b_ptr = [ None for _ in range(max_loras) ]
        # Lazily initialized
        self.indices: torch.Tensor
        self.indices_len: int

    def reset_lora(self, index: int):
        self.lora_a_ptr[index] = None
        self.lora_b_ptr[index] = None

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = self.input_size
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: int,
        lora_b: int
    ):
        self.reset_lora(index)

        if self.base_layer.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_ptr[index] = lora_a
        self.lora_b_ptr[index] = lora_b

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        indices_len: int
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)
        _apply_lora(
            x,
            self.lora_a_ptr,
            self.lora_b_ptr,
            self.indices[:self.indices_len],
            self.lora_config.max_lora_rank,
            output,
        )
        return output

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight if hasattr(
            self.base_layer, "weight") else self.base_layer.qweight

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is RowParallelLinear
