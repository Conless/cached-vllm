#include <torch/extension.h>

#include "punica_ops.h"

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dispatch_bgmv", &dispatch_bgmv, "dispatch_bgmv");
  m.def("dispatch_bgmv_low_level", &dispatch_bgmv_low_level,
        "dispatch_bgmv_low_level");
  m.def("sgmv_cutlass", &dispatch_sgmv_cutlass, "");
  m.def("sgmv_cutlass_custom", &dispatch_sgmv_cutlass_custom, "");
  m.def("sgmv_cutlass_tmp_size", &sgmv_tmp_size, "");
  m.def("sgmv_shrink", &dispatch_sgmv_shrink, "");
}
