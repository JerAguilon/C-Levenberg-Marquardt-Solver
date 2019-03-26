#ifndef MY_GTSAM_SOLVER_H
#define MY_GTSAM_SOLVER_H
#define TOL 1e-30

#include <iostream>
#include <algorithm>
#include <math.h>

#include <Eigen/Core>


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

    typedef double (*EvaluationFunction)(ParamMatrix params, XRow x);
    typedef void (*GradientFunction)(double *gradient, ParamMatrix params, XRow x);

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
    double (&_parameters)[RowsParams];
    double (&_x)[RowsMeasurements][RowsParams];
    double (&_y)[RowsMeasurements];

    double hessian[RowsParams][RowsParams],
           choleskyDecomposition[RowsParams][RowsParams];

    double derivative[RowsParams],
           gradient[RowsParams];

    double _newParameters[RowsParams],
           delta[RowsParams];

    double getError(
        ParamMatrix parameters,
        XMatrix x,
        YMatrix y
    );

    bool getCholeskyDecomposition();
    void solveCholesky();
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
    hessian{},
    choleskyDecomposition{},
    derivative{},
    gradient{},
    delta{},
    _newParameters{}
{}

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

template<int RowsMeasurements, int RowsParameters>
bool MyGTSAMSolver<RowsMeasurements, RowsParameters>::fit() {
    ParamMatrix parameters(&_parameters[0], RowsParameters, 1);
    ParamMatrix newParameters(&_newParameters[0], RowsParameters, 1);

    XMatrix x(&_x[0][0], RowsMeasurements, RowsParameters);
    YMatrix y(&_y[0], RowsMeasurements, 1);

    // TODO: make these input arguments
    int maxIterations = 10000;
    double lambda = 0.1;
    double upFactor = 10.0;
    double downFactor = 1.0/10.0;
    double targetDeltaError = 0.01;

    double currentError = getError(parameters, x, y);

    int iteration;
    for (iteration=0; iteration < maxIterations; iteration++) {
        std::cout << "Current Error: " << currentError << std::endl;
        std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl << std::endl;


        for (int i = 0; i < RowsParameters; i++) {
            derivative[i] = 0.0;
        }

        for (int i = 0; i < RowsParameters; i++) {
            for (int j = 0; j < RowsParameters; j++) {
                hessian[i][j] = 0.0;
            }
        }

        // Build out the jacobian and the hessian matrices
        for (int m = 0; m < RowsMeasurements; m++) {
            XRow currX(_x[m]);
            double currY = y[m];
            gradientFunction(gradient, parameters, currX);

            for (int i = 0; i < RowsParameters; i++) {
                // J_i = residual * gradient
                derivative[i] += (currY - evaluationFunction(parameters, currX)) * gradient[i];
                // H = J^T * J
                for (int j = 0; j <=i; j++) {
                    hessian[i][j] += gradient[i]*gradient[j];
                }
            }
        }

        double multFactor = 1 + lambda;
        bool illConditioned = true;
        double newError = 0;
        double deltaError = 0;

        while (illConditioned && iteration < maxIterations) {
            illConditioned = getCholeskyDecomposition();
            if (!illConditioned) {
                solveCholesky();
                for (int i = 0; i < RowsParameters; i++) {
                    _newParameters[i] = parameters(i) + delta[i];
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
            parameters(i) = _newParameters[i];
        }

        currentError = newError;
        lambda *= downFactor;

        if (!illConditioned && (-deltaError < targetDeltaError)) break;
    }
    std::cout << "Current Error: " << currentError << std::endl;
    std::cout << "Mean Error: " << currentError / RowsMeasurements << std::endl << std::endl;
    return true;
}

template<int RowsMeasurements, int RowsParameters>
bool MyGTSAMSolver<RowsMeasurements, RowsParameters>::getCholeskyDecomposition() {
    int i, j, k;
    double sum;

    int n = RowsParameters;

    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum += choleskyDecomposition[i][k] * choleskyDecomposition[j][k];
            }
            choleskyDecomposition[i][j] = (hessian[i][j] - sum) / choleskyDecomposition[j][j];
        }

        sum = 0;
        for (k = 0; k < i; k++) {
            sum += choleskyDecomposition[i][k] * choleskyDecomposition[i][k];
        }
        sum = hessian[i][i] - sum;
        if (sum < TOL) return true;
        choleskyDecomposition[i][i] = sqrt(sum);
    }
    return 0;
}

template<int RowsMeasurements, int RowsParameters>
void MyGTSAMSolver<RowsMeasurements, RowsParameters>::solveCholesky() {
    int i, j;
    double sum;

    int n = RowsParameters;

    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < i; j++) {
            sum += choleskyDecomposition[i][j] * delta[j];
        }
        delta[j] = (derivative[i] - sum) / choleskyDecomposition[i][i];
    }

    for (i = n - 1; i >= 0; i--) {
        sum = 0;
        for (j = i + 1; j < n; j++) {
            sum += choleskyDecomposition[j][i] * delta[j];
        }
        delta[i] = (delta[i] - sum) / choleskyDecomposition[i][i];
    }
}

#endif