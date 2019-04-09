/**
 * Copyright 2019 Jeremy Aguilon
 */

#pragma once

#include <Eigen/Core>

template <int NumMeasurements, int NumParameters>
class DataManipulator {
 public:
  virtual ~DataManipulator() = default;

  typedef Eigen::Map<Eigen::MatrixXd> MatrixMap;
  typedef Eigen::Map<Eigen::VectorXd> VectorMap;

  virtual void fillJacobian(MatrixMap &jacobian, VectorMap &params, int m,
                            int n) const = 0;
  virtual void fillResiduals(VectorMap &residuals, VectorMap &params, int m,
                             int n) const = 0;

  virtual void fillJacobian(double *jacobian, double *params, int m,
                            int n) const {
    MatrixMap j(&jacobian[0], m, n);
    VectorMap p(&params[0], n, 1);
    fillJacobian(j, p, m, n);
  }

  virtual void fillResiduals(double *residuals, double *params, int m,
                             int n) const {
    VectorMap r(&residuals[0], m, 1);
    VectorMap p(&params[0], n, 1);
    fillResiduals(r, p, m, n);
  }
};
