# Based on test_punica.py

import random
import pytest
import torch

import vllm.lora.sgmv.punica as punica

def assert_close(a, b):
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

    x_slices = [ x[s[i]:s[i+1]] for i in range(s.size(0) - 1) ]
    tmp = [ xi @ wa_ptr[i] for i, xi in enumerate(x_slices) ]
    y = [ t @ wb_ptr[i] for i, t in enumerate(tmp) ]
    y_final[s[0]:s[-1]] += torch.cat(y, dim=0)
    return y_final


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("h1", [128, 256])
@pytest.mark.parametrize("h2", [128, 256])
@torch.inference_mode()
def test_lora_sgmv_correctness(
    dtype_str: str,
    h1: int,
    h2: int,
):
    num_loras = 8
    r = 16
    dtype = getattr(torch, dtype_str)
    torch.set_default_device("cuda:0")

    wa = torch.stack([ torch.randn(h1, r, dtype=dtype) for _ in range(num_loras) ])
    wa_ptr = torch.tensor([ wa[i].data_ptr() for i in range(num_loras) ])
    wb = torch.stack([ torch.randn(r, h2, dtype=dtype) for _ in range(num_loras) ])
    wb_ptr = torch.tensor([ wb[i].data_ptr() for i in range(num_loras) ])
    
    token_for_loras = [16 for _ in range(num_loras)] # Replacing 16 with a random number would cause accuracy issues
    s = torch.cumsum(
        torch.tensor([0] + token_for_loras),
        dim=0,
        dtype=torch.int32,
    )

    x = torch.rand((s[-1], h1), dtype=dtype)
    y = torch.rand((s[-1], h2), dtype=dtype)

    y_ref = y.clone()
    _lora_ref_impl(y_ref, x, wa, wb, s)

    y_our = y.clone()
    punica.add_lora_sgmv_cutlass(y_our, x, wa_ptr, wb_ptr, s, 0, r)
    assert_close(y_ref, y_our)
