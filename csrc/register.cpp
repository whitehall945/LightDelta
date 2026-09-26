#include <torch/library.h>
#include "lightdelta/ops.h"

TORCH_LIBRARY(lightdelta, m) {
  m.def("decode(Tensor qkv, Tensor a, Tensor b, Tensor a_log, Tensor dt_bias, "
        "Tensor indices, Tensor cu, Tensor accepted, Tensor(s!) state, Tensor gate, "
        "Tensor weight, Tensor(o!) out, float eps) -> ()");
  m.def("gather_state(Tensor pool, Tensor slots, Tensor valid, Tensor(o!) out) -> ()");
  m.def("scatter_state(Tensor packed, Tensor slots, Tensor(p!) pool) -> ()");
}
TORCH_LIBRARY_IMPL(lightdelta, CUDA, m) {
  m.impl("decode", lightdelta::decode);
  m.impl("gather_state", lightdelta::gather_state);
  m.impl("scatter_state", lightdelta::scatter_state);
}
