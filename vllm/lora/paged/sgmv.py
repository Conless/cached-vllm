# Based on code from [Punica](https://github.com/punica-ai/punica)

import torch

from vllm.lora.punica import _raise_import_error

def sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(w_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    w_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    tmp_size = punica_kernels.sgmv_cutlass_tmp_size(w_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    punica_kernels.sgmv_cutlass(y, x, w_ptr, s, tmp, layer_idx)
    
def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    tmp_size = punica_kernels.sgmv_cutlass_tmp_size(len(wa_ptr))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    punica_kernels.sgmv_cutlass(v, x, wa_ptr, s, tmp, layer_idx)
    punica_kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp, layer_idx)
    return v, y
    # punica_kernels.sgmv_cutlass(y_tmp, v, wb_ptr, s, tmp, layer_idx)
    # y_tmp = y_tmp.to(torch.float16)
    # if not torch.allclose(y, y_tmp, rtol=1e-3, atol=1e-3):
        # print("Here")

def add_lora_sgmv_cutlass_fp32(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    x = x.to(torch.float32)
    tmp_size = punica_kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    punica_kernels.sgmv_cutlass(v, x, wa_ptr, s, tmp, layer_idx)
    y_tmp = y.clone().to(torch.float32)
    punica_kernels.sgmv_cutlass(y_tmp, v, wb_ptr, s, tmp, layer_idx)
    y.set_(y_tmp.to(y.dtype))
    return v.to(x.dtype), y

def add_lora_sgmv_cutlass_custom(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    tmp_size = punica_kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=torch.float32, device=x.device)
    punica_kernels.sgmv_cutlass_custom(v, x, wa_ptr, s, tmp, layer_idx)
    punica_kernels.sgmv_cutlass_custom(y, v, wb_ptr, s, tmp, layer_idx)
    return v.to(x.dtype), y


def add_lora_sgmv_cutlass_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]).T @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H1]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    tmp1 = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    tmp2_size = punica_kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp2 = torch.empty((tmp2_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    punica_kernels.sgmv_shrink(v, x, wa_ptr, s, tmp1, layer_idx)
    punica_kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp2, layer_idx)

# def add_lora_sgmv_triton(
#     y: torch.Tensor,
#     x: torch.Tensor,
    
    

  