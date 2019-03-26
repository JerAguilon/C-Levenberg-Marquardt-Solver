#include <iostream>
#include <algorithm>
#include <math.h>

#include "MyGTSAMSolver.h"

template<int NumParameters, int NumMeasurements>
MyGTSAMSolver<NumParameters, NumMeasurements>::MyGTSAMSolver(
    EvaluationFunction evaluationFunction,
    GradientFunction gradientFunction,
    double (&initialParams)[NumParameters],
    double (&x)[NumMeasurements][NumParameters],
    double (&y)[NumMeasurements]
):
    evaluationFunction(evaluationFunction),
    gradientFunction(gradientFunction),
    parameters(initialParams),
    x(x),
    y(y),
    hessian{},
    choleskyDecomposition{},
    derivative{},
    gradient{},
    delta{},
    newParameters{}
{}

template<int NumParameters, int NumMeasurements>
double MyGTSAMSolver<NumParameters, NumMeasurements>::getError(
    double (&parameters)[NumParameters],
    double (&x)[NumMeasurements][NumParameters],
    double (&y)[NumMeasurements])
{
    double residual;
    double error = 0;

    for (int i = 0; i < NumMeasurements; i++) {
        residual = evaluationFunction(parameters, x[i]) - y[i];
        error += residual * residual;
    }
    return error;
}

template<int NumParameters, int NumMeasurements>
bool MyGTSAMSolver<NumParameters, NumMeasurements>::fit() {
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
        std::cout << "Mean Error: " << currentError / NumMeasurements << std::endl << std::endl;


        for (int i = 0; i < NumParameters; i++) {
            derivative[i] = 0.0;
        }

        for (int i = 0; i < NumParameters; i++) {
            for (int j = 0; j < NumParameters; j++) {
                hessian[i][j] = 0.0;
            }
        }

        // Build out the jacobian and the hessian matrices
        for (int m = 0; m < NumMeasurements; m++) {
            double *currX = x[m];
            double currY = y[m];
            gradientFunction(gradient, parameters, currX);

            for (int i = 0; i < NumParameters; i++) {
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
                for (int i = 0; i < NumParameters; i++) {
                    newParameters[i] = parameters[i] + delta[i];
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

        for (int i = 0; i < NumParameters; i++) {
            parameters[i] = newParameters[i];
        }

        currentError = newError;
        lambda *= downFactor;

        if (!illConditioned && (-deltaError < targetDeltaError)) break;
    }
    std::cout << "Current Error: " << currentError << std::endl;
    std::cout << "Mean Error: " << currentError / NumMeasurements << std::endl << std::endl;
    return true;
}

template<int NumParameters, int NumMeasurements>
bool MyGTSAMSolver<NumParameters, NumMeasurements>::getCholeskyDecomposition() {
    int i, j, k;
    double sum;

    int n = NumParameters;

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

template<int NumParameters, int NumMeasurements>
void MyGTSAMSolver<NumParameters, NumMeasurements>::solveCholesky() {
    int i, j;
    double sum;

    int n = NumParameters;

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