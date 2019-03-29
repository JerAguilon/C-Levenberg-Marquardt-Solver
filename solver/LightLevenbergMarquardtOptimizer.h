#ifndef LIGHT_LEVENBERG_MARQUARDT_OPTIIMIZER
#define LIGHT_LEVENBERG_MARQUARDT_OPTIIMIZER

#include <iostream>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>

#include "DataManipulator.h"

/**
 *  Solves the equation X[RowsMeasurements _x RowsParam] * P[RowsParam] = Y[RowsMeasurements]
 */
template <int RowsMeasurements, int RowsParams>
class LightLevenbergMarquardtOptimizer
{

public:
    typedef Eigen::Map<Eigen::Matrix<double, RowsParams, 1>> ParamMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, RowsMeasurements, RowsParams, Eigen::RowMajor>> JacobianMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, RowsMeasurements, 1>> ResidualMatrix;

    const DataManipulator<RowsMeasurements, RowsParams> &dataManipulator;

    LightLevenbergMarquardtOptimizer(
        const DataManipulator<RowsMeasurements, RowsParams> &dataManipulator,
        double (&initialParams)[RowsParams]);

    bool fit();

protected:
    // Used internally for Hessians, and Cholesky Triangle Matrices
    typedef Eigen::Map<Eigen::Matrix<double, RowsParams, RowsParams, Eigen::RowMajor>> SquareParamMatrix;

    double (&_parameters)[RowsParams];

// Useful for large matrices that could cause a stack overflow. Alternatively, allocate this object in the heap,
// although that somewhat breaks the malloc-free paradigm.
#ifdef OPTIMIZER_USE_STATIC_MEMORY
    static double(_residuals)[RowsMeasurements];

    static double _hessian[RowsParams][RowsParams],
        _lowerTriangle[RowsParams][RowsParams];

    static double _derivative[RowsParams],
        _jacobianMatrix[RowsMeasurements][RowsParams];

    static double _newParameters[RowsParams],
        _delta[RowsParams];
#else
    double(_residuals)[RowsMeasurements];

    double _hessian[RowsParams][RowsParams],
        _lowerTriangle[RowsParams][RowsParams];

    double _derivative[RowsParams],
        _jacobianMatrix[RowsMeasurements][RowsParams];

    double _newParameters[RowsParams],
        _delta[RowsParams];
#endif

    double getError(ResidualMatrix residuals);
};

#ifdef OPTIMIZER_USE_STATIC_MEMORY

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_residuals[RowsMeasurements] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_hessian[RowsParameters][RowsParameters] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_lowerTriangle[RowsParameters][RowsParameters] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_derivative[RowsParameters] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_jacobianMatrix[RowsMeasurements][RowsParameters] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_newParameters[RowsParameters] = {0};

template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::_delta[RowsParameters] = {0};

#endif

template <int RowsMeasurements, int RowsParameters>
LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::LightLevenbergMarquardtOptimizer(
    const DataManipulator<RowsMeasurements, RowsParameters> &dataManipulator,
    double (&initialParams)[RowsParameters]) :
#ifdef OPTIMIZER_USE_STATIC_MEMORY
    dataManipulator(dataManipulator),
    _parameters(initialParams)
#else
    dataManipulator(dataManipulator),
    _parameters(initialParams),
    _hessian{},
    _derivative{},
    _jacobianMatrix{},
    _delta{},
    _residuals{},
    _newParameters{}
{}
#endif

/**
 * Computes the sum of of square residuals error of a given parameter
 * configuration.
 */
template <int RowsMeasurements, int RowsParameters>
double LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::getError(ResidualMatrix residuals)
{
    return (residuals.transpose() * residuals)(0, 0);
}

/**
 * Computes the Levenberg-Marquadt solution to a nonlinear system.
 */
template <int RowsMeasurements, int RowsParameters>
bool LightLevenbergMarquardtOptimizer<RowsMeasurements, RowsParameters>::fit()
{
    bool success = false;

    ParamMatrix parameters(&_parameters[0], RowsParameters, 1);
    ParamMatrix newParameters(&_newParameters[0], RowsParameters, 1);

    ResidualMatrix residuals(&_residuals[0], RowsMeasurements, 1);

    ParamMatrix derivative(&_derivative[0], RowsParameters);
    ParamMatrix delta(&_delta[0], RowsParameters);

    SquareParamMatrix hessian(&_hessian[0][0], RowsParameters, RowsParameters);
    SquareParamMatrix lowerTriangle(&_lowerTriangle[0][0], RowsParameters, RowsParameters);

    JacobianMatrix jacobianMatrix(&_jacobianMatrix[0][0], RowsMeasurements, RowsParameters);

    // TODO: make these input arguments
    int maxIterations = 500;
    double lambda = 0.01;
    double upFactor = 10.0;
    double downFactor = 1.0 / 10.0;
    double targetDeltaError = 0.01;

    dataManipulator.fillResiduals(residuals, parameters);
    double currentError = getError(residuals);

    int iteration;
    for (iteration = 0; iteration < maxIterations; iteration++)
    {
        std::cout << "Current Error: " << currentError << std::endl;
        std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl
                  << std::endl;

        derivative.setZero();
        jacobianMatrix.setZero();
        hessian.setZero();

        // Build out the jacobian and the hessian matrices
        // H = J^T * J
        dataManipulator.fillJacobian(jacobianMatrix, parameters);
        hessian.noalias() = jacobianMatrix.transpose() * jacobianMatrix;

        // Compute the right hand side of the update equation:
        // -∇f(x) = J^T(y - y_hat)
        // In other words, this is a  negative derivative of the evaluation
        // function w.r.t. the model parameters
        derivative.noalias() = jacobianMatrix.transpose() * -residuals;

        double multFactor = 1 + lambda;
        bool illConditioned = true;
        double newError = 0;
        double deltaError = 0;

        while (illConditioned && iteration < maxIterations)
        {
            for (int i = 0; i < RowsParameters; i++)
            {
                hessian(i, i) = hessian(i, i) * multFactor;
            }

            // Computes an LLT decomposition inplace, which saves on some memory
            // overhead
            lowerTriangle = hessian;
            Eigen::LLT<Eigen::Ref<SquareParamMatrix>> llt(lowerTriangle);

            // The decoposition fails if the matrix is not positive semi definite
            illConditioned = (llt.info() != Eigen::ComputationInfo::Success);
            if (!illConditioned)
            {
                delta = llt.solve(derivative);
                for (int i = 0; i < RowsParameters; i++)
                {
                    newParameters(i) = parameters(i) + delta(i);
                }
                dataManipulator.fillResiduals(residuals, newParameters);
                newError = getError(residuals);
                deltaError = newError - currentError;
                illConditioned = deltaError > 0;
            }

            if (illConditioned)
            {
                multFactor = (1 + lambda * upFactor) / (1 + lambda);
                lambda *= upFactor;
                iteration++;
            }
        }

        for (int i = 0; i < RowsParameters; i++)
        {
            parameters(i) = newParameters(i);
        }

        currentError = newError;
        lambda *= downFactor;

        if (!illConditioned && (-deltaError < targetDeltaError))
        {
            success = true;
            break;
        };
    }
    std::cout << "Current Error: " << currentError << std::endl;
    std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl;
    std::cout << "Total iterations: " << iteration + 1 << std::endl
              << std::endl;
    return success;
}

#endif