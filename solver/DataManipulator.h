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

  virtual void fillJacobian(
          const int m, const int n, MatrixMap *jacobian, VectorMap *params) const = 0;
  virtual void fillResiduals(
          const int m, const int n, VectorMap *residuals, VectorMap *params) const = 0;

  virtual void fillJacobian(const int m, const int n, double *jacobian, double *params) const {
    MatrixMap j(&jacobian[0], m, n);
    VectorMap p(&params[0], n, 1);
    fillJacobian(m, n, &j, &p);
  }

  virtual void fillResiduals(const int m, const int n, double *residuals, double *params) const {
    VectorMap r(&residuals[0], m, 1);
    VectorMap p(&params[0], n, 1);
    fillResiduals(m, n, &r, &p);
  }
};
