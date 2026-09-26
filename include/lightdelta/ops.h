#pragma once
#include <ATen/ATen.h>

namespace lightdelta {
void decode(const at::Tensor& qkv, const at::Tensor& a, const at::Tensor& b,
            const at::Tensor& a_log, const at::Tensor& dt_bias,
            const at::Tensor& indices, const at::Tensor& cu,
            const at::Tensor& accepted, at::Tensor state,
            const at::Tensor& gate, const at::Tensor& weight,
            at::Tensor out, double eps);
void gather_state(const at::Tensor& pool, const at::Tensor& slots,
                  const at::Tensor& valid, at::Tensor out);
void scatter_state(const at::Tensor& packed, const at::Tensor& slots, at::Tensor pool);
}  // namespace lightdelta
