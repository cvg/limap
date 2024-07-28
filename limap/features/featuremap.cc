#include "features/featuremap.h"

namespace limap {

namespace features {

template <typename DTYPE>
FeatureMap<DTYPE>::FeatureMap(const Eigen::MatrixXd &array) {
  height = array.rows();
  width = array.cols();
  channels = 1;

  data_.resize(height * width);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      data_[row * width + col] = DTYPE(array(row, col));
    }
  }
  data_ptr_ = data_.data();
}

template <typename DTYPE>
FeatureMap<DTYPE>::FeatureMap(
    const py::array_t<DTYPE, py::array::c_style> &pyarray, bool do_copy) {
  py::buffer_info buffer_info = pyarray.request();

  data_ptr_ = static_cast<DTYPE *>(buffer_info.ptr);
  std::vector<ssize_t> shape = buffer_info.shape;
  if (shape.size() != 2 && shape.size() != 3) {
    throw std::runtime_error("Unsupported shape!");
  }
  height = shape[0];
  width = shape[1];
  if (shape.size() == 3)
    channels = shape[2];
  else
    channels = 1;

  if (do_copy) {
    ssize_t size = Size();
    THROW_CHECK_EQ(buffer_info.size, size);
    data_.assign(data_ptr_, data_ptr_ + size);
    data_ptr_ = &data_[0];
  }
}

template class FeatureMap<float16>;
template class FeatureMap<float>;
template class FeatureMap<double>;

} // namespace features

} // namespace limap
