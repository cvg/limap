
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <Eigen/Core>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "VLFeat/covdet.h"
#include "VLFeat/dsift.h"
#include "VLFeat/sift.h"
#include <cstdlib>

#include <chrono>

#include "util/simple_logger.h"
#include <colmap/util/logging.h>

namespace limap {

namespace features {

py::array_t<float> extract_dsift(const py::array_t<float> image,
                                 const double step, const double bin_size) {
  // Check that input is grayscale.
  // assert(image.ndim() == 2);
  THROW_CHECK_EQ(image.ndim(), 2);

  VlDsiftFilter *dsift =
      vl_dsift_new_basic(image.shape(1), image.shape(0), step, bin_size);

  // Recover pointer to image;
  py::buffer_info image_buf = image.request();
  float *image_ptr = (float *)image_buf.ptr;

  size_t width = image.shape(1) - 3 * bin_size;
  size_t height = image.shape(0) - 3 * bin_size;
  size_t channels = 128;

  size_t kdim = width * height * channels;
  auto t1 = std::chrono::high_resolution_clock::now();

  vl_dsift_process(dsift, image_ptr);
  const float *descriptors_f = vl_dsift_get_descriptors(dsift);

  auto t2 = std::chrono::high_resolution_clock::now();
  STDLOG(INFO)
      << "DSIFT:"
      << " "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms" << std::endl;
  // Descriptors.
  py::array_t<float> pydescriptors(
      py::detail::any_container<ssize_t>({height, width, channels}));
  py::buffer_info pydescriptors_buf = pydescriptors.request();
  float *pydescriptors_ptr = (float *)pydescriptors_buf.ptr;

  memcpy(pydescriptors_ptr, descriptors_f, kdim * sizeof(float));

  vl_dsift_delete(dsift);

  return pydescriptors;
}

} // namespace features

} // namespace limap
