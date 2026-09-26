#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "lightdelta/ops.h"

namespace lightdelta {
namespace {
constexpr int kTile = 1024;

template <bool Gather>
__global__ void state_copy(const float* input, float* output, const int* slots,
                           const bool* valid, int64_t elements,
                           int64_t input_stride, int64_t output_stride,
                           int64_t slot_stride) {
  const int row = blockIdx.x;
  const int slot = slots[row * slot_stride];
  if (!Gather && slot <= 0) return;
  const bool live = slot > 0 && (!Gather || valid[row]);
  const auto input_base = static_cast<int64_t>(Gather ? slot : row) * input_stride;
  const auto output_base = static_cast<int64_t>(Gather ? row : slot) * output_stride;
  for (int i = threadIdx.x; i < kTile; i += blockDim.x) {
    const int64_t element = static_cast<int64_t>(blockIdx.y) * kTile + i;
    if (element < elements)
      output[output_base + element] = live ? input[input_base + element] : 0.0f;
  }
}

void check(const at::Tensor& pool, const at::Tensor& slots, const at::Tensor& packed) {
  TORCH_CHECK(pool.is_cuda() && pool.scalar_type() == at::kFloat && pool.dim() == 4
              && pool.stride(3) == 1 && pool.stride(2) == pool.size(3)
              && pool.stride(1) == pool.size(2)*pool.size(3),
              "pool must be CUDA FP32 with dense inner dimensions");
  TORCH_CHECK(slots.device() == pool.device() && slots.scalar_type() == at::kInt
              && slots.dim() == 1, "slots must be a CUDA int32 vector");
  TORCH_CHECK(packed.device() == pool.device() && packed.scalar_type() == at::kFloat
              && packed.dim() == 4 && packed.size(0) == slots.numel()
              && packed.sizes().slice(1) == pool.sizes().slice(1)
              && packed.stride(3) == 1 && packed.stride(2) == packed.size(3)
              && packed.stride(1) == packed.size(2)*packed.size(3), "invalid packed layout");
}
}  // namespace

void gather_state(const at::Tensor& pool, const at::Tensor& slots,
                  const at::Tensor& valid, at::Tensor out) {
  check(pool, slots, out);
  TORCH_CHECK(valid.device() == pool.device() && valid.scalar_type() == at::kBool
              && valid.sizes() == slots.sizes() && valid.is_contiguous(),
              "valid must be a CUDA bool vector");
  if (slots.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(pool.device());
  const auto elements = pool.size(1)*pool.size(2)*pool.size(3);
  state_copy<true><<<dim3(slots.numel(),(elements+kTile-1)/kTile),256,0,
      c10::cuda::getCurrentCUDAStream()>>>(pool.data_ptr<float>(),out.data_ptr<float>(),
      slots.data_ptr<int>(),valid.data_ptr<bool>(),elements,pool.stride(0),out.stride(0),
      slots.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_state(const at::Tensor& packed, const at::Tensor& slots, at::Tensor pool) {
  check(pool, slots, packed);
  if (slots.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(pool.device());
  const auto elements = pool.size(1)*pool.size(2)*pool.size(3);
  state_copy<false><<<dim3(slots.numel(),(elements+kTile-1)/kTile),256,0,
      c10::cuda::getCurrentCUDAStream()>>>(packed.data_ptr<float>(),pool.data_ptr<float>(),
      slots.data_ptr<int>(),nullptr,elements,packed.stride(0),pool.stride(0),slots.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace lightdelta
