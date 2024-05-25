#pragma once

#include <torch/extension.h>

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, float scale);

void dispatch_bgmv_low_level(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                             torch::Tensor indicies, int64_t layer_idx,
                             float scale, int64_t h_in, int64_t h_out,
                             int64_t y_offset);

void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x,
                           torch::Tensor w_ptr, torch::Tensor s,
                           torch::Tensor tmp, int layer_idx);

void dispatch_sgmv_shrink(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                          torch::Tensor s, torch::Tensor tmp, int layer_idx);

size_t sgmv_tmp_size(int num_problems);
