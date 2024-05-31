import sys
from typing import Dict, Set

import torch
from vllm.config import LoRAConfig


class PageCacheMemoryPool:
    """ A memory pool that caches pages of VRAM
    """

    def __init__(
        self,
        lora_config: LoRAConfig,
    ):
        self.max_size = lora_config.memory_pool_size
        self.page_size = lora_config.memory_pool_page_size
        self.page_count = self.max_size // self.page_size
        self.max_loras = lora_config.max_loras

        # Configuration for an adapter page
        self.min_hidden_size = 1024
        self.max_rank = lora_config.max_lora_rank
        self.page_width = self.min_hidden_size * self.max_rank
        if lora_config.lora_dtype == torch.float16 or lora_config.lora_dtype == torch.bfloat16:
            page_width_size = self.page_width * 2
        elif lora_config.lora_dtype == torch.float32:
            page_width_size = self.page_width * 4
        else:
            raise ValueError(f"Unsupported dtype: {lora_config.lora_dtype}")

        self.page_rows = self.page_size * (2**20) // self.page_width
        
        print(f"Page cache init with total size: {self.max_size} MB, page size: {self.page_size} MB, each with {self.page_rows} entries")
        self.pages = [
            (
                i,
                torch.zeros((self.page_rows, self.min_hidden_size * self.max_rank), dtype=lora_config.lora_dtype, device="cuda")
            ) for i in range(self.page_count)
        ]
        self.available_pages = dict(self.pages)
        self.used_pages: Dict[int, Set[int]] = {}
    
    def available_page_count(self):
        return len(self.available_pages)
    
    def evict_page(self, lora_id: int, page_index: int):
        try:
            self.used_pages[lora_id].remove(page_index)
        except KeyError:
            raise ValueError(f"Page {page_index} is not used by LoRA {lora_id}")
        page = self.pages[page_index]
        self.available_pages[page_index] = page
    
    def allocate_page(self, lora_id: int) -> tuple[int, torch.Tensor]:
        if self.available_page_count() == 0:
            raise ValueError("No available pages")
        if lora_id not in self.used_pages:
            self.used_pages[lora_id] = set()
        page_index, page = self.available_pages.popitem()
        self.used_pages[lora_id].add(page_index)
        return page_index, page
        