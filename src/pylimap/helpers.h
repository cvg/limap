#pragma once

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

} // namespace detail
} // namespace pybind11
