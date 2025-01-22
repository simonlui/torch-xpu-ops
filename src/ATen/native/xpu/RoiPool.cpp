#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/RoiPoolKernels.h>
#include <comm/XPUGuard.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>
namespace at::native::xpu {

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  TORCH_CHECK(input.is_xpu(), "input must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  c10::DeviceGuard device_guard(input.device());
  return roi_pool_kernel(
      input, rois, spatial_scale, pooled_height, pooled_width);
}

at::Tensor _roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(grad.is_xpu(), "grad must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");
  TORCH_CHECK(argmax.is_xpu(), "argmax must be a XPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "roi_pool_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  c10::DeviceGuard device_guard(grad.device());

  return roi_pool_backward_kernel(
      grad,
      rois,
      argmax,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width);
}

} // namespace at::native::xpu
