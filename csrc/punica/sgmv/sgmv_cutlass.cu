#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/arch/mma.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "sgmv_cutlass.cuh"

template bool sgmv<nv_half>(nv_half* y, nv_half* x, nv_half** w, int32_t* s,
                            void* tmp_d, int num_problems, int d_in, int d_out,
                            int layer_idx, cudaStream_t stream);

template bool sgmv<nv_bfloat16>(nv_bfloat16* y, nv_bfloat16* x, nv_bfloat16** w,
                                int32_t* s, void* tmp_d, int num_problems,
                                int d_in, int d_out, int layer_idx,
                                cudaStream_t stream);

template bool sgmv<float>(float* y, float* x, float** w, int32_t* s,
                          void* tmp_d, int num_problems, int d_in, int d_out,
                          int layer_idx, cudaStream_t stream);

template bool sgmv_custom<float, nv_half, nv_half>(
    float* y, nv_half* x, nv_half** w, int32_t* s, void* tmp_d,
    int num_problems, int d_in, int d_out, int layer_idx, cudaStream_t stream);

template bool sgmv_custom<nv_half, float, float>(
    nv_half* y, float* x, float** w, int32_t* s, void* tmp_d, int num_problems,
    int d_in, int d_out, int layer_idx, cudaStream_t stream);

template bool sgmv_custom<float, nv_bfloat16, nv_bfloat16>(
    float* y, nv_bfloat16* x, nv_bfloat16** w, int32_t* s, void* tmp_d,
    int num_problems, int d_in, int d_out, int layer_idx, cudaStream_t stream);

template bool sgmv_custom<nv_bfloat16, float, float>(
    nv_bfloat16* y, float* x, float** w, int32_t* s, void* tmp_d,
    int num_problems, int d_in, int d_out, int layer_idx, cudaStream_t stream);

// void haha() {
//   using t = cutlass::gemm::threadblock::DefaultMma<
//       nv_half, cutlass::layout::RowMajor, 8, float,
//       cutlass::layout::RowMajor, 8, float, cutlass::layout::RowMajor,
//       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
//       cutlass::gemm::GemmShape<32, 128, 16>,
//       cutlass::gemm::GemmShape<32, 64, 16>, cutlass::gemm::GemmShape<16, 8,
//       8>, 2, cutlass::arch::OpMultiplyAddMixedInputUpcast, false,
//       cutlass::gemm::SharedMemoryClearOption::kNone, false, false,
//       cutlass::layout::NoPermute, cutlass::layout::NoPermute>;
//   t a;
// }
