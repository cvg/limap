#include "features/featurepatch.h"

namespace limap {

namespace features {

template <typename DTYPE>
PatchInfo<DTYPE> FeaturePatch<DTYPE>::GetPatchInfo() const {
  size_t height, width, channels;
  height = FeatureMap<DTYPE>::Height();
  width = FeatureMap<DTYPE>::Width();
  channels = FeatureMap<DTYPE>::Channels();
  size_t size = FeatureMap<DTYPE>::Size();
  const DTYPE *data_ptr = FeatureMap<DTYPE>::Data();

  py::array_t<DTYPE, py::array::c_style> pyarray =
      py::array_t<DTYPE, py::array::c_style>(
          std::vector<size_t>{height, width, channels});
  py::buffer_info buffer_info = pyarray.request();
  DTYPE *data_ptr_array = static_cast<DTYPE *>(buffer_info.ptr);
  memcpy(data_ptr_array, data_ptr, sizeof(DTYPE) * size);
  return PatchInfo<DTYPE>(pyarray, R, tvec, img_hw);
}

template class FeaturePatch<float16>;
template class FeaturePatch<float>;
template class FeaturePatch<double>;

} // namespace features

} // namespace limap
