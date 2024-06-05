# Based on test_punica.py

import random
import pytest
import torch
import math
import numpy as np

import vllm.lora.paged.sgmv as punica
from vllm.lora.punica import add_lora

def next_power_of_2(n):
    return 2**math.ceil(math.log2(n))

def assert_close(a, b, rtol=None, atol=None):
    if rtol is None or atol is None:
      rtol, atol = {
          torch.float16: (5e-3, 5e-3),
          torch.bfloat16: (3e-2, 2e-2),
          torch.float32: (None, None),
      }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

def _lora_ref_impl(
    y_final: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
):
    """
    Semantics:
      y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_ptr: Shape: `[S]`. DType: torch.Tensor.
        Weight matrix shape: `[num_layers, H1, R]`.
      wb_ptr: Shape: `[S]`. DType: torch.Tensor.
        Weight matrix shape: `[num_layers, R, H2]`.
      s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
        `s[0] == 0`, `s[-1] == B`.
      layer_idx: Layer index of the weight matrices.
    """

    x_slices = [ x[s[i]:s[i+1]].to(dtype=torch.float32).contiguous() for i in range(s.size(0) - 1) ]
    wa_ptr_fp32 = wa_ptr.to(dtype=torch.float32)
    tmp = [ (xi @ wa_ptr_fp32[i]) for i, xi in enumerate(x_slices) ]
    wb_ptr_fp32 = wb_ptr.to(dtype=torch.float32)
    y = [ (t @ wb_ptr_fp32[i]).to(dtype=torch.float16) for i, t in enumerate(tmp) ]
    tmp = torch.cat(tmp, dim=0).to(dtype=torch.float16)
    y_final[s[0]:s[-1]] += torch.cat(y, dim=0)
    return tmp, y_final


@pytest.mark.parametrize("dtype_str", ["float16"])
@pytest.mark.parametrize("h1", [128])
@pytest.mark.parametrize("h2", [128])
@torch.inference_mode()
def test_lora_sgmv_correctness(
    dtype_str: str,
    h1: int,
    h2: int,
):
    num_loras = 8
    r = 64
    dtype = getattr(torch, dtype_str)
    torch.set_default_device("cuda:0")

    wa = torch.stack([ torch.randn(h1, r, dtype=dtype) for _ in range(num_loras) ])
    wa_fp32 = wa.to(dtype=torch.float32)
    wa_ptr = torch.tensor([ wa[i].data_ptr() for i in range(num_loras) ])
    wa_ptr_fp32 = torch.tensor([ wa_fp32[i].data_ptr() for i in range(num_loras) ])
    wb = torch.stack([ torch.randn(r, h2, dtype=dtype) for _ in range(num_loras) ])
    wb_fp32 = wb.to(dtype=torch.float32)
    wb_ptr = torch.tensor([ wb[i].data_ptr() for i in range(num_loras) ])
    wb_ptr_fp32 = torch.tensor([ wb_fp32[i].data_ptr() for i in range(num_loras) ])
    
    token_for_loras = [ random.randint(8, 32) for _ in range(num_loras - 1)] 
    token_for_loras = np.cumsum([0] + token_for_loras).tolist()
    token_for_loras.append(next_power_of_2(token_for_loras[-1]))
    s = torch.tensor(
        token_for_loras,
        dtype=torch.int32,
    )
    

    x = torch.rand((s[-1], h1), dtype=dtype)
    y = torch.rand((s[-1], h2), dtype=dtype)

    y_ref = y.clone()
    mid_ref, y_ref = _lora_ref_impl(y_ref, x, wa, wb, s)

    # y_our = y.clone()
    # mid_our, y_our = punica.add_lora_sgmv_cutlass(y_our, x, wa_ptr, wb_ptr, s, 0, r)
    # assert_close(y_ref, y_our, rtol=1, atol=0.1)
    
    # y_custom = y.clone()
    # mid_custom, y_custom = punica.add_lora_sgmv_cutlass_custom(y_custom, x, wa_ptr, wb_ptr_fp32, s, 0, r)
    # assert_close(y_ref, y_custom, rtol=1, atol=0.1)
    
    y_fp32 = y.clone()
    mid_fp32, y_fp32 = punica.add_lora_sgmv_cutlass_fp32(y_fp32, x, wa_ptr_fp32, wb_ptr_fp32, s, 0, r)
    mid_fp32 = mid_fp32.to(mid_ref.dtype)
    assert_close(mid_ref, mid_fp32, rtol=1, atol=0.1)
    assert_close(y_ref, y_fp32, rtol=1, atol=0.1)


@pytest.mark.parametrize("batch_size", [8192])
@pytest.mark.parametrize("h1", [4096])
@pytest.mark.parametrize("h2", [4096])
@pytest.mark.parametrize("num_loras", [16])
@pytest.mark.parametrize("lora_rank", [64])
@torch.inference_mode()
def apply_lora(
    batch_size: int,
    h1: int,
    h2: int,
    num_loras: int,
    lora_rank: int
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
    dtype = torch.float16

    indices = []
    while len(indices) < batch_size:
        seq_len = random.randint(16, 32)
        lora_idx = random.randint(0, num_loras - 1)
        indices.extend([lora_idx] * seq_len)
    indices = torch.tensor(indices[:batch_size], dtype=torch.long, device="cuda:0")

    x = torch.randn((batch_size, h1), dtype=dtype, device="cuda:0")
    output = torch.randn((batch_size, h2), dtype=dtype, device="cuda:0")
    lora_a_stacked = torch.randn((num_loras, 1, lora_rank, h1), dtype=dtype, device="cuda:0")
    lora_b_stacked = torch.randn((num_loras, 1, h2, lora_rank), dtype=dtype, device="cuda:0")
    
    # v = v.to(torch.float16)
    
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
    
    wa = []
    wb = []
    wa_fp32 = []
    wb_fp32 = []
    wa_ptr = []
    wb_ptr = []
    wa_ptr_fp32 = []
    wb_ptr_fp32 = []
    s = []

    for i in range(indices.shape[0]):
        if i == 0 or indices[i] != indices[i - 1]:
            if indices[i] == -1:
                wa.append(empty_lora_a)
                wb.append(empty_lora_b)
                wa_fp32.append(empty_lora_a_fp32)
                wb_fp32.append(empty_lora_b_fp32)
                wa_ptr.append(empty_lora_a.data_ptr())
                wb_ptr.append(empty_lora_b.data_ptr())
                wa_ptr_fp32.append(empty_lora_a_fp32.data_ptr())
                wb_ptr_fp32.append(empty_lora_b_fp32.data_ptr())
            else:
                wa.append(lora_a_stacked_transposed[indices[i]])
                wb.append(lora_b_stacked_transposed[indices[i]])
                wa_fp32.append(lora_a_stacked_transposed_fp32[indices[i]])
                wb_fp32.append(lora_b_stacked_transposed_fp32[indices[i]])
                wa_ptr.append(lora_a_stacked_transposed[indices[i]].data_ptr())
                wb_ptr.append(lora_b_stacked_transposed[indices[i]].data_ptr())
                wa_ptr_fp32.append(lora_a_stacked_transposed_fp32[indices[i]].data_ptr())
                wb_ptr_fp32.append(lora_b_stacked_transposed_fp32[indices[i]].data_ptr())
            s.append(i)
    s.append(indices.shape[0])
    
    wa = torch.stack(wa)
    wb = torch.stack(wb)
    wa_fp32 = torch.stack(wa_fp32)
    wb_fp32 = torch.stack(wb_fp32)
    wa_ptr = torch.tensor(wa_ptr, device="cuda:0", dtype=torch.int64)
    wb_ptr = torch.tensor(wb_ptr, device="cuda:0", dtype=torch.int64)
    wa_ptr_fp32 = torch.tensor(wa_ptr_fp32, device="cuda:0", dtype=torch.int64)
    wb_ptr_fp32 = torch.tensor(wb_ptr_fp32, device="cuda:0", dtype=torch.int64)
    s = torch.tensor(s, device="cuda:0", dtype=torch.int32)
    
    # output_tmp = output.clone()
    # print(output.sum(), x.sum())
    # add_lora_sgmv_cutlass(output, x, wa_ptr, wb_ptr, s, 0, empty_lora_a.shape[-1])
    # 25.93, 25.72
    # add_lora_sgmv_cutlass_custom(output, x, wa_ptr, wb_ptr_fp32, s, 0, empty_lora_a.shape[-1])
    output_fp32 = output.clone()
    output_ref = output.clone()
    # print(x.sum(), output.sum())
    v, _ = add_lora(output, x, lora_a_stacked, lora_b_stacked, indices, 0, 1.0)
    v_fp32, _ = punica.add_lora_sgmv_cutlass(output_fp32, x, wa_ptr, wb_ptr, s, 0, empty_lora_a.shape[-1])
    v = v.to(v_fp32.dtype)
    
    v_ref, _ = _lora_ref_impl(output_ref, x, wa, wb, s)
    
    assert_close(v_ref, v_fp32, rtol=1, atol=0.1)
    # assert_close(output, output_fp32, rtol=1, atol=0.1)