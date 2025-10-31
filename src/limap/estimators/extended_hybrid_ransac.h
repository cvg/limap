#pragma once

#include <RansacLib/hybrid_ransac.h>

namespace limap {

namespace estimators {

class ExtendedHybridLORansacOptions : public ransac_lib::HybridLORansacOptions {
public:
  ExtendedHybridLORansacOptions() : non_min_sample_multiplier_(3) {}
  // We add this to do non minimal sampling in LO step in align with
  // the original definition of the LO step
  int non_min_sample_multiplier_;
};

} // namespace estimators

} // namespace limap
