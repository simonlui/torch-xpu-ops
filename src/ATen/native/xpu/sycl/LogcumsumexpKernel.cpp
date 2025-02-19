#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBase.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>

#include <cmath>
#include <limits>

namespace at::native::xpu {

// custom min and max to be used in logcumsumexp for complex arguments
template <typename scalar_t, bool min>
c10::complex<scalar_t> _logcumsumexp_minmax(
    const c10::complex<scalar_t>& x,
    const c10::complex<scalar_t>& y) {
  scalar_t xr = std::real(x);
  scalar_t yr = std::real(y);
  if (at::_isnan(yr) || (at::_isnan(std::imag(y)))) {
    return y;
  } else if (at::_isnan(xr) || (at::_isnan(std::imag(x)))) {
    return x;
  } else if (min) { // min
    return (xr < yr) ? x : y;
  } else { // max
    return (xr >= yr) ? x : y;
  }
}

template <typename scalar_t>
scalar_t _log_add_exp_helper(const scalar_t& x, const scalar_t& y) {
  // Reference :
  // https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  // Using the original expression: `at::_isnan(y) ? y : std::min(x, y)` causes
  // an error in ROCM
  auto isnan_x = at::_isnan(x);
  auto isnan_y = at::_isnan(y);
  scalar_t min = isnan_y ? y : (isnan_x ? x : std::min(x, y));
  scalar_t max = isnan_y ? y : (isnan_x ? x : std::max(x, y));
  if (min != max || std::isfinite(min)) {
    // nan will be propagated here
    return ::log1p(std::exp(min - max)) + max;
  } else {
    // special case to correctly handle infinite cases
    return x;
  }
}

template <typename scalar_t>
c10::complex<scalar_t> _fast_build_exp(const c10::complex<scalar_t>& x) {
  // complex exponential function, but implemented manually to get fast
  // compilation time this function only handles the case where the x is finite
  // (not inf nor nan)
  auto xreal = std::real(x);
  auto ximag = std::imag(x);
  auto exp_x_abs = std::exp(xreal);
  auto exp_x_real = exp_x_abs * std::cos(ximag);
  auto exp_x_imag = exp_x_abs * std::sin(ximag);
  return {exp_x_real, exp_x_imag};
}

template <typename scalar_t>
c10::complex<scalar_t> _fast_build_exp_inf(const c10::complex<scalar_t>& x) {
  // complex exponential function, but implemented manually to get fast
  // compilation time this function only handles the case where the real part of
  // x is infinite
  auto ximag = std::imag(x);
  auto exp_x_abs = std::numeric_limits<scalar_t>::infinity();
  auto sin = std::sin(ximag);
  auto cos = std::cos(ximag);
  // special case if the angle is exactly the multiple of pi/2
  auto exp_x_real = (cos == 0) ? (scalar_t)0.0 : exp_x_abs * cos;
  auto exp_x_imag = (sin == 0) ? (scalar_t)0.0 : exp_x_abs * sin;
  return {exp_x_real, exp_x_imag};
}

template <typename scalar_t>
c10::complex<scalar_t> _log_add_exp_helper(
    const c10::complex<scalar_t>& x,
    const c10::complex<scalar_t>& y) {
  c10::complex<scalar_t> min =
      _logcumsumexp_minmax<scalar_t, /*min=*/true>(x, y);
  c10::complex<scalar_t> max =
      _logcumsumexp_minmax<scalar_t, /*min=*/false>(x, y);
  scalar_t min_real = std::real(min);
  scalar_t max_real = std::real(max);

  if (at::_isnan(min_real) || at::_isnan(std::imag(min))) {
    // handling the "infectious" NaNs
    return {
        std::numeric_limits<scalar_t>::quiet_NaN(),
        std::numeric_limits<scalar_t>::quiet_NaN()};
  } else if ((!std::isfinite(min_real)) && (min_real == max_real)) {
    if (min_real < 0) {
      // handle the -inf case, the imaginary part here does not really matter as
      // the exp(value) will be around 0.0 and the angle (i.e. the imaginary
      // part) cannot be determined. It does not matter if we're taking the exp
      // of this value
      return min;
    } else {
      // handle the +inf case, we don't need the special precision for log1p for
      // small values and to avoid producing nan in case of real(max) ==
      // real(min) == +inf
      auto exp_min = _fast_build_exp_inf(min);
      auto exp_max = _fast_build_exp_inf(max);
      return ::log1p(
          exp_min + exp_max - 1); // log1p(x - 1) builds faster than log
    }
  } else {
    auto minmax = min - max;
    auto exp_minmax = _fast_build_exp(minmax);
    return ::log1p(exp_minmax) + max;
  }
}

template <typename scalar_t, typename opmath_t>
struct _logcumsumexp_out_log_add_exp_functor {
  scalar_t operator()(const scalar_t x_, const scalar_t y_) const {
    const opmath_t x{x_}, y{y_};
    return _log_add_exp_helper(x, y);
  }
};

void launch_logcumsumexp_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "logcumsumexp_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        scalar_t init = -std::numeric_limits<scalar_t>::infinity();
        _logcumsumexp_out_log_add_exp_functor<scalar_t, opmath_t> log_add_exp;
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            result, self, dim, init, log_add_exp);
      });
}

} // namespace at::native::xpu
