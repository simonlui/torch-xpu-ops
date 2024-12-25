#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <comm/xpu_aten.h>

namespace at::native {

Tensor _fft_c2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
}

Tensor& _fft_c2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
}

Tensor _fft_c2r_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2r_mkl(self, dim, normalization, last_dim_size);
}

Tensor _fft_c2r_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2r_mkl_out(self, dim, normalization, last_dim_size, out);
}

} // namespace at::native
