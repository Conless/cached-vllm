import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import safetensors.torch
import torch
from torch import nn

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import (BaseLayerWithLoRA,
                              LoRAMapping)
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.models import LoRALRUCache, LoRAModel, LoRAModelManager
from vllm.lora.paged.layers import BaseLayerWithPagedLoRA
from vllm.lora.paged.memory_pool import PageCacheMemoryPool
from vllm.lora.utils import (from_layer, from_layer_logits_processor,
                             parse_fine_tuned_lora_name, replace_submodule)
from vllm.utils import CudaMemoryProfiler, LRUCache, is_pin_memory_available

logger = init_logger(__name__)

_GLOBAL_LORA_ID = 0

def convert_mapping(
    mapping: LoRAMapping,
    lora_index_to_id: List[Optional[int]],
    lora_id_to_index: Dict[int, int],
    max_loras: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], List[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.
        long_lora_context: Passed if there are long context lora in a batch.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            long_lora_indices: Tensor of shape [batch_size] mapping
                requests to RoPE offsets and rot dims for long LoRAs.
                None if long context lora doesn't exist.
            indices_len: List of lengths of the above tensors.
                Used to index into each tensor. It contains length for
                (base_indices, sampler_indices, sampler_indices_padded,
                embeddings_indices, long_lora_indices). If long_lora doesn't
                exist, it only contains first 4 entries.
    """
    index_mapping_indices: List[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()
    long_lora_offsets: Optional[torch.Tensor] = None
    prompt_mapping: List[int] = [
        lora_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        # TODO index can be slow. optimize
        lora_idx = (lora_index_to_id.index(index_mapping_indices[i])
                    if index_mapping_indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx

    indices_list: List[Union[List[int], torch.Tensor]] = [
        index_mapping_indices, lora_indices, embedding_indices
    ]
    indices = torch.tensor(indices_list, dtype=torch.long, device="cuda")
    prompt_mapping_tensor = torch.tensor(prompt_mapping,
                                         device="cuda",
                                         dtype=torch.long)
    base_indices = indices[1]
    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = base_indices.shape[-1]

    # return (base_indices, sampler_indices, sampler_indices_padded)


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


class LoRAPagedModel(LoRAModel):
    """A paged LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: Dict[str, LoRALayerWeights],
        scaling_factor: Optional[float] = None,
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.
            scaling_factor: Scaling factor to support long context lora model.
                None if the lora is not tuned for long context support.
        """
        self.id = lora_model_id
        # Scaling factor for long context lora model. None if it is not
        # fine tuned for the long context.
        self.scaling_factor = scaling_factor
        assert (lora_model_id >
                0), f"a valid lora id should be greater than 0, got {self.id}"
        self.rank = rank
        self.loras: Dict[str, LoRALayerWeights] = loras
        self.lora_ptrs : Dict[str, Tuple[torch.int64, torch.int64]]= None
        self.lora_pages : List[int] = None
    
    def get_lora_ptr(self, module_name: str) -> Tuple[int, int]:
        if self.lora_ptrs is None:
            raise ValueError("LoRAModel is not activated.")
        return self.lora_ptrs[module_name]
    
    def profile(self, pool: PageCacheMemoryPool) -> int:
        """Profile the paged memory usage of the LoRAModel."""
        self.pages = 0
        current_page_row_index = 0
        for lora in self.loras.values():
            if lora.rank < pool.max_rank: # Align to max rank
                lora_a_replace = torch.zeros(
                    (lora.input_dim, pool.max_rank), dtype=lora.lora_a.dtype, device=lora.lora_a.device)
                lora_a_replace[:, :lora.rank].copy_(lora.lora_a)
                lora.lora_a = lora_a_replace
                lora_b_replace = torch.zeros(
                    (pool.max_rank, lora.output_dim), dtype=lora.lora_b.dtype, device=lora.lora_b.device)
                lora_b_replace[:lora.rank, :].copy_(lora.lora_b)
                lora.lora_b = lora_b_replace
                lora.rank = pool.max_rank
            for lora_tensor in [lora.lora_a, lora.lora_b]:
                lora_nums = lora_tensor.numel()
                lora_rows = lora_nums // pool.page_width
                assert lora_nums % pool.page_width == 0, f"lora_nums {lora_nums} is not a multiple of page_width {pool.page_width}"
                if current_page_row_index + lora_rows > pool.page_rows:
                    self.pages += 1
                    current_page_row_index = 0
                current_page_row_index += lora_rows
        return self.pages


    def activate(self, pool: PageCacheMemoryPool):
        if self.lora_ptrs is not None:
            raise ValueError("LoRAModel is already activated.")
        self.lora_ptrs = {}
        current_page_id, current_page = pool.allocate_page(self.id)
        self.lora_pages = [current_page_id]
        current_page_row_index = 0
        for lora in self.loras.values():
            # Init LoRA.A
            lora_a_nums = lora.lora_a.numel()
            assert lora_a_nums % pool.page_width == 0, f"lora_a_nums {lora_a_nums} is not a multiple of page_width {pool.page_width}"
            lora_a_rows = lora.lora_a.numel() // pool.page_width
            if current_page_row_index + lora_a_rows > pool.page_rows:
                current_page_id, current_page = pool.allocate_page(self.id)
                self.lora_pages.append(current_page_id)
                current_page_row_index = 0
            current_page[current_page_row_index:current_page_row_index + lora_a_rows].copy_(lora.lora_a.view(-1, pool.page_width))
            lora_a_ptr = current_page[current_page_row_index].data_ptr()
            current_page_row_index += lora_a_rows
            
            # Init LoRA.B, same as above
            lora_b_nums = lora.lora_b.numel()
            assert lora_b_nums % pool.page_width == 0, f"lora_b_nums {lora_b_nums} is not a multiple of page_width {pool.page_width}"
            lora_b_rows = lora.lora_b.numel() // pool.page_width
            if current_page_row_index + lora_b_rows > pool.page_rows:
                current_page_id, current_page = pool.allocate_page(self.id)
                current_page_row_index = 0
            current_page[current_page_row_index:current_page_row_index + lora_b_rows].copy_(lora.lora_b.view(-1, pool.page_width))
            lora_b_ptr = current_page[current_page_row_index].data_ptr()
            current_page_row_index += lora_b_rows

            self.lora_pages
            self.lora_ptrs[lora.module_name] = (lora_a_ptr, lora_a_ptr)
    
    def deactivate(self, pool: PageCacheMemoryPool):
        if self.lora_ptrs is None:
            raise ValueError("LoRAModel is not activated.")
        for page_id in self.lora_pages:
            pool.evict_page(self.id, page_id)
        self.lora_ptrs = None
        self.lora_pages = None
        
class PageCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with page cache."""


    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
    ):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.lora_config = lora_config
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: List[Optional[int]] = [None] * self.lora_slots
        self.lora_id_to_index: Dict[int, int] = {}
        self.base_indices = torch.empty(self.max_num_batched_tokens,
                                        dtype=torch.long,
                                        device="cuda")
        # Scaling factor -> offset to the sin_cos_cache to it.
        # Used for long context lora.
        self.scaling_factor_to_offset: Dict[float, int] = {}
        # 4 is the number of indicies tensors defined above
        # base_indices, sampler_indices, sampler_indices_padded,
        # embeddings_indices
        self.indices_len: int = 0

        self.model: nn.Module = model
        if hasattr(self.model, "supported_lora_modules"):
            self.supported_lora_modules = copy.deepcopy(
                self.model.supported_lora_modules)
        self.modules: Dict[str, "BaseLayerWithLoRA"] = {}
        self._registered_loras: Dict[int, LoRAModel] = {}
        # Dict instead of a Set for compatibility with LRUCache.
        self._active_loras: Dict[int, None] = {}
        self._last_mapping: Optional[LoRAMapping] = None
        self._create_lora_modules()
        self.model.lora_manager = self
        self._registered_loras: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_lora)
        self._active_loras: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_lora)
        self.memory_pool: PageCacheMemoryPool = PageCacheMemoryPool(lora_config) 

    def activate_lora(self, lora_id: int) -> bool:
        if lora_id in self._active_loras:
            self._active_loras.touch(lora_id)
            return False
        lora_model = self._registered_loras.get(lora_id)
        if not lora_model:
            return False
        while lora_model.pages >= self.memory_pool.available_page_count() or len(self._active_loras) >= self.lora_slots:
            self._active_loras.remove_oldest()
        lora_model.activate(self.memory_pool)
        first_free_slot = next(
            ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
             if lora_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_loras[lora_id] = None
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = lora_model.get_lora_ptr(module_name)
            if module_lora:
                module.set_lora(index, module_lora[0], module_lora[1])
            else:
                module.reset_lora(index)

    def _deactivate_lora(self, lora_id: int):
        try:
            index = self.lora_id_to_index[lora_id]
            self.lora_index_to_id[index] = None
            self.lora_id_to_index.pop(lora_id)
        except ValueError:
            pass

    def deactivate_lora(self, lora_id: int) -> bool:
        """Remove a LoRA from a GPU buffer."""
        if lora_id in self._active_loras:
            self._deactivate_lora(lora_id)
            self._active_loras.pop(lora_id)
            self._registered_loras[lora_id].lora_ptrs = None
            return True
        return False

    # This is a CPU operation, so the same as LRUCacheLoRAManager
    def add_lora(self, lora: LoRAPagedModel) -> bool:
        """Add a LoRAModel to the manager."""
        logger.debug(
            "Adding lora. Model id: %d, "
            "int id: %d, "
            "scaling factor: %s", lora.id, lora.id, lora.scaling_factor)
        if lora.id not in self._registered_loras:
            lora.profile(self.memory_pool)
            self._registered_loras[lora.id] = lora
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_loras.touch(lora.id)
            was_added = False
        return was_added

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRAModel from the manager CPU cache."""
        # TODO: should we check active lora?
        self.deactivate_lora(lora_id)
        return bool(self._registered_loras.pop(lora_id, None))

    # TODO see if this can be vectorized
    def _set_lora_mapping(self, mapping: LoRAMapping) -> None:
        (base_indices, sampler_indices, sampler_indices_padded,
         embeddings_indices, long_lora_offsets_tensor,
         indices_len) = convert_mapping(mapping, self.lora_index_to_id,
                                        self.lora_slots + 1, self.vocab_size,
                                        self.lora_config.lora_extra_vocab_size,
                                        self.long_lora_context)
        self.base_indices[:base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        self.embeddings_indices[:embeddings_indices.
                                shape[0], :embeddings_indices.shape[1]].copy_(
                                    embeddings_indices)
        if long_lora_offsets_tensor is not None:
            self.long_lora_indices[:long_lora_offsets_tensor.shape[0]].copy_(
                long_lora_offsets_tensor)
        else:
            self.long_lora_indices.zero_()
        # Maintain the reference
        self.indices_len[:] = indices_len

    def set_lora_mapping(self, lora_mapping: LoRAMapping) -> None:
        pass
        # if self._last_mapping != lora_mapping:
        #     self._set_lora_mapping(lora_mapping)
        # self._last_mapping = lora_mapping

    def list_loras(self) -> Dict[int, LoRAPagedModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras.cache)

    def remove_all_loras(self):
        """Remove all LoRAModels from the manager."""
        self._registered_loras.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_loras.clear()

    def _create_lora_modules(self):
        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            # with CudaMemoryProfiler() as m:
                if not self._match_target_modules(module_name):
                    continue
                parts = module_name.split(".")[-1]
                new_module = replace_submodule(
                        self.model, module_name,
                        from_layer(module, self.lora_slots, self.lora_config,
                                [], self.model.config))
                self.register_module(module_name, new_module)
                new_module.set_mapping(self.base_indices, self.indices_len)
            # logger.info("Loading module" + module_name + "took %.4f MB, used %.4f GB in total.", m.consumed_memory / float(2 ** 20), m.current_memory_usage() / float(2 ** 30))

    def register_module(self, module_name: str, module: "BaseLayerWithPagedLoRA"):
        assert isinstance(module, BaseLayerWithPagedLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            scaling_factor: Optional[float],
            embedding_modules: Optional[Dict[str, str]] = None) -> LoRAPagedModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAPagedModel(lora_id, rank, {}, scaling_factor)
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name) or not isinstance(
                    module, BaseLayerWithPagedLoRA):
                continue
            lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.input_size,
                        module.output_size,
                        rank,
                        module.lora_config.lora_dtype,
                        "cpu",
                    )
            lora.optimize()
            model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_lora_modules)


    
