#ifndef DATA_MANIPULATOR_H
#define DATA_MANIPULATOR_H

#include <Eigen/Core>

template<int M, int N>
class DataManipulator {
public:
    virtual ~DataManipulator() = default;

    typedef Eigen::Map<Eigen::Matrix<double, N, 1>> ParamMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, M, 1>> ResidualMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, M, N, Eigen::RowMajor>> JacobianMatrix;
    virtual void fillJacobian(JacobianMatrix &jacobian, ParamMatrix &param) const = 0;
    virtual void fillResiduals(ResidualMatrix &residuals, ParamMatrix &params) const = 0;
};


#endif