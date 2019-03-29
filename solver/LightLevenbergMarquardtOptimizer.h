#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>

#include "DataManipulator.h"

/**
 *  Solves the equation X[NumMeasurements _x RowsParam] * P[RowsParam] = Y[NumMeasurements]
 */
template <int NumMeasurements, int NumParameters>
class LightLevenbergMarquardtOptimizer
{

public:
    typedef Eigen::Map<Eigen::Matrix<double, -1, 1>> ParamMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, -1, -1>> JacobianMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, -1, 1>> ResidualMatrix;

    const DataManipulator<NumMeasurements, NumParameters> &dataManipulator;

    LightLevenbergMarquardtOptimizer(
        const DataManipulator<NumMeasurements, NumParameters> &dataManipulator,
        double (&initialParams)[NumParameters]);

    bool fit();

protected:
    // Used internally for Hessians, and Cholesky Triangle Matrices
    typedef Eigen::Map<Eigen::Matrix<double, NumParameters, NumParameters>> SquareParamMatrix;

    double (&_parameters)[NumParameters];

// Useful for large matrices that could cause a stack overflow. Alternatively, allocate this object in the heap,
// although that somewhat breaks the malloc-free paradigm.
#ifdef OPTIMIZER_USE_STATIC_MEMORY
    static double(_residuals)[NumMeasurements];

    static double _hessian[NumParameters * NumParameters],
        _lowerTriangle[NumParameters * NumParameters];

    static double _derivative[NumParameters],
        _jacobianMatrix[NumMeasurements * NumParameters];

    static double _newParameters[NumParameters],
        _delta[NumParameters];
#else
    double(_residuals)[NumMeasurements];

    double _hessian[NumParameters * NumParameters],
        _lowerTriangle[NumParameters * NumParameters];

    double _derivative[NumParameters],
        _jacobianMatrix[NumMeasurements * NumParameters];

    double _newParameters[NumParameters],
        _delta[NumParameters];
#endif

    double getError(ResidualMatrix residuals);
};

#ifdef OPTIMIZER_USE_STATIC_MEMORY

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_residuals[NumMeasurements] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_hessian[NumParameters * NumParameters] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_lowerTriangle[NumParameters * NumParameters] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_derivative[NumParameters] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_jacobianMatrix[NumMeasurements * NumParameters] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_newParameters[NumParameters] = {0};

template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::_delta[NumParameters] = {0};

#endif

template <int NumMeasurements, int NumParameters>
LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::LightLevenbergMarquardtOptimizer(
    const DataManipulator<NumMeasurements, NumParameters> &dataManipulator,
    double (&initialParams)[NumParameters]) :
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
#endif
{}

/**
 * Computes the sum of of square residuals error of a given parameter
 * configuration.
 */
template <int NumMeasurements, int NumParameters>
double LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::getError(ResidualMatrix residuals)
{
    return (residuals.transpose() * residuals)(0, 0);
}

/**
 * Computes the Levenberg-Marquadt solution to a nonlinear system.
 */
template <int NumMeasurements, int NumParameters>
bool LightLevenbergMarquardtOptimizer<NumMeasurements, NumParameters>::fit()
{
    bool success = false;

    ParamMatrix parameters(&_parameters[0], NumParameters, 1);
    ParamMatrix newParameters(&_newParameters[0], NumParameters, 1);

    ResidualMatrix residuals(&_residuals[0], NumMeasurements, 1);

    ParamMatrix derivative(&_derivative[0], NumParameters);
    ParamMatrix delta(&_delta[0], NumParameters);

    SquareParamMatrix hessian(&_hessian[0], NumParameters, NumParameters);
    SquareParamMatrix lowerTriangle(&_lowerTriangle[0], NumParameters, NumParameters);

    JacobianMatrix jacobianMatrix(&_jacobianMatrix[0], NumMeasurements, NumParameters);

    // TODO: make these input arguments
    int maxIterations = 500;
    double lambda = 0.01;
    double upFactor = 10.0;
    double downFactor = 1.0 / 10.0;
    double targetDeltaError = 0.01;

    double *residualsPtr = residuals.data();
    double *newParamPtr = newParameters.data();
    double *paramPtr = parameters.data();
    double *jacobianPtr = jacobianMatrix.data();

    dataManipulator.fillResiduals(residualsPtr, paramPtr, NumMeasurements, NumParameters);
    double currentError = getError(residuals);

    int iteration;
    for (iteration = 0; iteration < maxIterations; iteration++)
    {
        std::cout << "Current Error: " << currentError << std::endl;
        std::cout << "Mean Error: " << currentError / NumMeasurements << std::endl
                  << std::endl;

        derivative.setZero();
        jacobianMatrix.setZero();
        hessian.setZero();

        // Build out the jacobian and the hessian matrices
        // H = J^T * J
        dataManipulator.fillJacobian(jacobianPtr, paramPtr, NumMeasurements, NumParameters);
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
            for (int i = 0; i < NumParameters; i++)
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
                for (int i = 0; i < NumParameters; i++)
                {
                    newParameters(i) = parameters(i) + delta(i);
                }
                dataManipulator.fillResiduals(residualsPtr, newParamPtr, NumMeasurements, NumParameters);
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

        for (int i = 0; i < NumParameters; i++)
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
    std::cout << "Mean Error: " << currentError / NumMeasurements << std::endl;
    std::cout << "Total iterations: " << iteration + 1 << std::endl
              << std::endl;
    return success;
}