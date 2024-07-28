#ifndef LIMAP_CERESBASE_LOSS_FUNCTIONS_H
#define LIMAP_CERESBASE_LOSS_FUNCTIONS_H

// Modified from the pixel-perfect-sfm project.

#include <ceres/loss_function.h>
#include <cmath>
#include <pybind11/pybind11.h>

namespace limap {

class CERES_EXPORT Trivial4Loss : public ceres::LossFunction {
public:
  explicit Trivial4Loss(double k) : k4_(1.0 / (k * k * k * k)) {}
  inline void Evaluate(double, double *) const override;

private:
  const double k4_;
};

inline void Trivial4Loss::Evaluate(double s, double rho[3]) const {
  rho[0] = k4_ * s * s;
  rho[1] = k4_ * s * 2.0;
  rho[0] = 2.0 * k4_;
}

class CERES_EXPORT CustomBerHuLoss : public ceres::LossFunction {
public:
  explicit CustomBerHuLoss(double a, double k)
      : a_(a), b_(k - 2.0 * a), c_(a * a), k_(k) {}
  inline void Evaluate(double, double *) const override;

private:
  const double a_;
  // b = a^2.
  const double b_;
  const double c_;

  const double k_;
};

inline void CustomBerHuLoss::Evaluate(double s, double rho[3]) const {
  const double r = ceres::sqrt(s) + 1.0e-5;
  if (s <= c_) {
    // Inlier region.
    rho[0] = k_ * r;
    rho[1] = 0.5 * k_ / r;
    rho[2] = -rho[1] * 0.5 / s;
  } else {
    // Outlier region.
    rho[0] = s + b_ * r + c_;
    rho[1] = 1 + 0.5 * b_ / r;
    rho[2] = -rho[1] * 0.5 / s;
  }
}

class CERES_EXPORT BerHuLoss : public ceres::LossFunction {
public:
  explicit BerHuLoss(double a) : a_(a), b_(a * a) {}
  inline void Evaluate(double, double *) const override;

private:
  const double a_;
  // b = a^2.
  const double b_;
};

inline void BerHuLoss::Evaluate(double s, double rho[3]) const {
  if (s <= b_) {
    // Outlier region.
    // 'r' is always positive.
    const double r = std::sqrt(s) + 1.0e-5;
    rho[0] = 2.0 * a_ * r;
    rho[1] = std::max(std::numeric_limits<double>::min(), a_ / r);
    rho[2] = -rho[1] / (2.0 * s);
  } else {
    // Inlier region.
    rho[0] = s + 2 * b_;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
}

class CERES_EXPORT GemanMcClureLoss : public ceres::LossFunction {
public:
  explicit GemanMcClureLoss(double tau) : tau2_(4.0 * tau * tau) {}
  inline void Evaluate(double, double *) const override;

private:
  const double tau2_;
  // b = a^2.
};

inline void GemanMcClureLoss::Evaluate(double s, double rho[3]) const {
  double sum = s + tau2_;
  rho[0] = s / sum;
  rho[1] = tau2_ / (sum * sum);
  rho[2] = -2.0 * tau2_ / (sum * sum * sum);
}

class CERES_EXPORT EpsilonLoss : public ceres::LossFunction {
public:
  explicit EpsilonLoss(double eps)
      : eps2_(0.5 * eps * eps) {
  } // Set eps to the maximal allowable width in pixel at featuremap resolution!
  inline void Evaluate(double, double *) const override;

private:
  const double eps2_;
  // b = a^2.
};

inline void EpsilonLoss::Evaluate(double s, double rho[3]) const {
  if (s <= eps2_) {
    // Outlier region.
    // 'r' is always positive.
    rho[0] = 0.0;
    rho[1] = 0.0;
    rho[2] = 0.0;
  } else {
    // Inlier region.
    rho[0] = s - eps2_;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
}

// Tukey fix
class CERES_EXPORT TukeyLoss : public ceres::LossFunction {
public:
  explicit TukeyLoss(double a) : a_squared_(a * a) {}
  void Evaluate(double, double *) const override;

private:
  const double a_squared_;
};

inline void TukeyLoss::Evaluate(double s, double *rho) const {
  if (s <= a_squared_) {
    // Inlier region.
    const double value = 1.0 - s / a_squared_;
    const double value_sq = value * value;
    rho[0] = a_squared_ / 3.0 * (1.0 - value_sq * value);
    rho[1] = value_sq;
    rho[2] = -2.0 / a_squared_ * value;
  } else {
    // Outlier region.
    rho[0] = a_squared_ / 3.0;
    rho[1] = 0.0;
    rho[2] = 0.0;
  }
}

enum class LossFunctionType {
  TRIVIAL = 0,
  TRIVIAL4,
  SOFT_L1,
  CAUCHY,
  GEMAN,
  BERHU,
  EPSILON,
  TOLERANT,
  ARCTAN,
  TUKEY
};

extern

    inline LossFunctionType
    ResolveLossName(std::string loss_name) {
  std::unordered_map<std::string, LossFunctionType> loss_type_table = {
      {"trivial", LossFunctionType::TRIVIAL},
      {"trivial4", LossFunctionType::TRIVIAL4},
      {"soft_l1", LossFunctionType::SOFT_L1},
      {"cauchy", LossFunctionType::CAUCHY},
      {"geman", LossFunctionType::GEMAN},
      {"berhu", LossFunctionType::BERHU},
      {"epsilon", LossFunctionType::EPSILON},
      {"tolerant", LossFunctionType::TOLERANT},
      {"arctan", LossFunctionType::ARCTAN},
      {"tukey", LossFunctionType::TUKEY},
  };
  return loss_type_table[loss_name];
}

inline ceres::LossFunction *CreateLossFunction(std::string loss_name,
                                               std::vector<double> scales) {
  LossFunctionType loss_type = ResolveLossName(loss_name);
  ceres::LossFunction *loss_function = nullptr;
  switch (loss_type) {
  case LossFunctionType::TRIVIAL:
    loss_function = new ceres::TrivialLoss();
    break;
  case LossFunctionType::TRIVIAL4:
    loss_function = new Trivial4Loss(scales.at(0));
    break;
  case LossFunctionType::SOFT_L1:
    loss_function = new ceres::SoftLOneLoss(scales.at(0));
    break;
  case LossFunctionType::CAUCHY:
    loss_function = new ceres::CauchyLoss(scales.at(0));
    break;
  case LossFunctionType::GEMAN:
    loss_function = new GemanMcClureLoss(scales.at(0));
    break;
  case LossFunctionType::BERHU:
    loss_function = new CustomBerHuLoss(scales.at(0), scales.at(1));
    break;
  case LossFunctionType::EPSILON:
    loss_function = new EpsilonLoss(scales.at(0));
    break;
  case LossFunctionType::TOLERANT:
    loss_function = new ceres::TolerantLoss(scales.at(0), scales.at(1));
    break;
  case LossFunctionType::ARCTAN:
    loss_function = new ceres::ArctanLoss(scales.at(0));
    break;
  case LossFunctionType::TUKEY:
    loss_function = new TukeyLoss(scales.at(0));
    break;
  }
  std::string failure_message = "Unknown loss_name " + loss_name;
  THROW_CUSTOM_CHECK_MSG(loss_function, std::invalid_argument,
                         failure_message.c_str());
  return loss_function;
}

inline ceres::LossFunction *CreateScaledLossFunction(std::string loss_name,
                                                     std::vector<double> scales,
                                                     double magnitude) {
  return new ceres::ScaledLoss(CreateLossFunction(loss_name, scales), magnitude,
                               ceres::TAKE_OWNERSHIP);
}

inline ceres::LossFunction *CreateLossFunction(py::dict dict) {
  THROW_CHECK(dict.contains("name"));
  std::string loss_name = dict["name"].cast<std::string>();

  if (loss_name != std::string("trivial")) {
    THROW_CHECK(dict.contains("params"));
  }
  if (dict.contains("magnitude")) {
    return CreateScaledLossFunction(dict["name"].cast<std::string>(),
                                    dict["params"].cast<std::vector<double>>(),
                                    dict["magnitude"].cast<double>());
  }
  return CreateLossFunction(dict["name"].cast<std::string>(),
                            dict["params"].cast<std::vector<double>>());
}

} // namespace limap

#endif
