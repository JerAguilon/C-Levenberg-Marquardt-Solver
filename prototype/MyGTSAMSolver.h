#ifndef MY_GTSAM_SOLVER_H
#define MY_GTSAM_SOLVER_H
#define TOL 1e-30

#include <iostream>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>


/**
 *  Solves the equation X[RowsMeasurements _x RowsParam] * P[RowsParam] = Y[RowsMeasurements]
 */
template<int RowsMeasurements, int RowsParams>
class MyGTSAMSolver {

public:
    typedef Eigen::Map<Eigen::Matrix<double, RowsParams, 1>> ParamMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, RowsMeasurements, RowsParams, Eigen::RowMajor>> XMatrix;
    typedef Eigen::Map<Eigen::Matrix<double, RowsMeasurements, 1>> YMatrix;
    typedef Eigen::Matrix<double, RowsParams, 1> XRow;
    typedef Eigen::Map<Eigen::Matrix<double, RowsMeasurements, RowsParams, Eigen::RowMajor>> JacobianMatrix;

    typedef double (*EvaluationFunction)(const ParamMatrix &params, const XRow &x);
    typedef void (*GradientFunction)(JacobianMatrix &jacobian, ParamMatrix &params, const XMatrix &x);

    EvaluationFunction evaluationFunction;
    GradientFunction gradientFunction;

    MyGTSAMSolver(
        EvaluationFunction evaluationFunction,
        GradientFunction gradientFunction,
        double (&initialParams)[RowsParams],
        double (&x)[RowsMeasurements][RowsParams],
        double (&y)[RowsMeasurements]
    );

    bool fit();

private:
    // Used internally for Hessians, and Cholesky Triangle Matrices
    typedef Eigen::Map<Eigen::Matrix<double, RowsParams, RowsParams, Eigen::RowMajor>> SquareParamMatrix;

    double (&_parameters)[RowsParams];
    double (&_x)[RowsMeasurements][RowsParams];
    double (&_y)[RowsMeasurements];

    double _hessian[RowsParams][RowsParams];

    double _derivative[RowsParams],
           _jacobianMatrix[RowsMeasurements][RowsParams];

    double _newParameters[RowsParams],
           _delta[RowsParams];

    double getError(
        ParamMatrix parameters,
        XMatrix x,
        YMatrix y
    );

    void solveCholesky(
        SquareParamMatrix &choleskyDecomposition,
        ParamMatrix &derivative,
        ParamMatrix &delta
    );
};

template<int RowsMeasurements, int RowsParameters>
MyGTSAMSolver<RowsMeasurements, RowsParameters>::MyGTSAMSolver(
    EvaluationFunction evaluationFunction,
    GradientFunction gradientFunction,
    double (&initialParams)[RowsParameters],
    double (&x)[RowsMeasurements][RowsParameters],
    double (&y)[RowsMeasurements]
):
    evaluationFunction(evaluationFunction),
    gradientFunction(gradientFunction),
    _parameters(initialParams),
    _x(x),
    _y(y),
    _hessian{},
    _derivative{},
    _jacobianMatrix{},
    _delta{},
    _newParameters{}
{}

/**
 * Computes the sum of of square residuals error of a given parameter
 * configuration.
 */
template<int RowsMeasurements, int RowsParameters>
double MyGTSAMSolver<RowsMeasurements, RowsParameters>::getError(
    ParamMatrix parameters,
    XMatrix x,
    YMatrix y)
{
    double residual;
    double error = 0;

    for (int i = 0; i < RowsMeasurements; i++) {
        residual = evaluationFunction(parameters, x.row(i)) - y[i];
        error += residual * residual;
    }
    return error;
}


/**
 * Computes the Levenberg-Marquadt solution to a nonlinear system.
 */
template<int RowsMeasurements, int RowsParameters>
bool MyGTSAMSolver<RowsMeasurements, RowsParameters>::fit() {
    bool success = false;

    ParamMatrix parameters(&_parameters[0], RowsParameters, 1);
    ParamMatrix newParameters(&_newParameters[0], RowsParameters, 1);

    XMatrix x(&_x[0][0], RowsMeasurements, RowsParameters);
    YMatrix y(&_y[0], RowsMeasurements, 1);

    ParamMatrix derivative(&_derivative[0], RowsParameters);
    ParamMatrix delta(&_delta[0], RowsParameters);

    SquareParamMatrix hessian(&_hessian[0][0], RowsParameters, RowsParameters);

    JacobianMatrix jacobianMatrix(&_jacobianMatrix[0][0], RowsMeasurements, RowsParameters);

    // TODO: make these input arguments
    int maxIterations = 500;
    double lambda = 0.1;
    double upFactor = 10.0;
    double downFactor = 1.0/10.0;
    double targetDeltaError = 0.01;

    double currentError = getError(parameters, x, y);

    int iteration;
    for (iteration=0; iteration < maxIterations; iteration++) {
        std::cout << "Current Error: " << currentError << std::endl;
        std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl << std::endl;

        derivative.setZero();
        jacobianMatrix.setZero();
        hessian.setZero();

        // Build out the jacobian and the _hessian matrices
        // H = J^T * J
        gradientFunction(jacobianMatrix, parameters, x);
        hessian.noalias() = jacobianMatrix.transpose() * jacobianMatrix;

        // Compute the right hand side of the update equation:
        // -∇f(x) = J^T(y - y_hat)
        // In other words, this is a  negative derivative of the evaluation
        // function w.r.t. the model parameters
        for (int m = 0; m < RowsMeasurements; m++) {
            double currY = y[m];
            auto currentGradient = jacobianMatrix.row(m);
            for (int i = 0; i < RowsParameters; i++) {
                derivative[i] += (currY - evaluationFunction(parameters, x.row(m))) * currentGradient(i);
            }
        }

        double multFactor = 1 + lambda;
        bool illConditioned = true;
        double newError = 0;
        double deltaError = 0;

        while (illConditioned && iteration < maxIterations) {
            // TODO(Jeremy): Debug this line
            // for (int i = 0; i < RowsParameters; i++) {
            //     hessian(i, i) *= multFactor;
            // }

            // Computes an in-place LL^T Cholesky decomposition, which saves
            // on some memory overhead
            Eigen::LLT<Eigen::Ref<SquareParamMatrix> > lu(hessian);

            // The decoposition failed if info() returns anything other than 0
            illConditioned = lu.info() != 0;
            if (!illConditioned) {
                solveCholesky(hessian, derivative, delta);
                for (int i = 0; i < RowsParameters; i++) {
                    newParameters(i) = parameters(i) + delta(i);
                }
                newError = getError(newParameters, x, y);
                deltaError = newError - currentError;
                illConditioned = deltaError > 0;
            }

            if (illConditioned) {
                multFactor = (1 + lambda * upFactor) / (1 + lambda);
                lambda *= upFactor;
                iteration++;
            }
        }

        for (int i = 0; i < RowsParameters; i++) {
            parameters(i) = newParameters(i);
        }

        currentError = newError;
        lambda *= downFactor;

        if (!illConditioned && (-deltaError < targetDeltaError)) {
            success = true;
            break;
        };
    }
    std::cout << "Current Error: " << currentError << std::endl;
    std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl << std::endl;
    return success;
}


/**
 * Solves an L*L^T decomposed hessian matrix and stores the result into
 * delta
 */
template<int RowsMeasurements, int RowsParameters>
void MyGTSAMSolver<RowsMeasurements, RowsParameters>::solveCholesky(
    SquareParamMatrix &choleskyDecomposition,
    ParamMatrix &derivative,
    ParamMatrix &delta
) {
    int i, j;
    double sum;

    int n = RowsParameters;

    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < i; j++) {
            sum += choleskyDecomposition(i, j) * delta(j);
        }
        delta(j) = (derivative(i) - sum) / choleskyDecomposition(i, i);
    }

    for (i = n - 1; i >= 0; i--) {
        sum = 0;
        for (j = i + 1; j < n; j++) {
            sum += choleskyDecomposition(j, i) * delta(j);
        }
        delta(i) = (delta(i) - sum) / choleskyDecomposition(i, i);
    }
}

#endif