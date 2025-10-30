#pragma once

// This file is modified from the pixel-perfect-sfm project

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <third-party/half.h>
using float16 = half_float::half;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Make pybind11 support float16 conversion
// https://github.com/eacousineau/repro/blob/43407e3/python/pybind11/custom_tests/test_numpy_issue1776.cc
namespace pybind11 {
namespace detail {

#define SINGLE_ARG(...) __VA_ARGS__

#define ASSIGN_PYDICT_ITEM(dict, key, type)                                    \
  if (dict.contains(#key))                                                     \
    key = dict[#key].cast<type>();

#define ASSIGN_PYDICT_ITEM_TO_MEMBER(obj, dict, key, type)                     \
  if (dict.contains(#key))                                                     \
    obj.key = dict[#key].cast<type>();

#define ASSIGN_PYDICT_ITEM_TKEY(dict, key, tkey, type)                         \
  if (dict.contains(#key))                                                     \
    tkey = dict[#key].cast<type>();

template <typename T> struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type"); // Could make more efficient.
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type))
      return false;
    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }
    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});
    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

} // namespace detail
} // namespace pybind11

static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 {
namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following:
// https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <> struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <> struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

} // namespace detail
} // namespace pybind11
