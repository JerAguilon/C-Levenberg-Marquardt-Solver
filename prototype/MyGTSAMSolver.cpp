#include <iostream>
#include <algorithm>

#include "MyGTSAMSolver.h"

template<int NumParameters, int NumMeasurements>
MyGTSAMSolver<NumParameters, NumMeasurements>::MyGTSAMSolver(
    EvaluationFunction evaluationFunction,
    GradientFunction gradientFunction,
    double (&initialParams)[NumParameters],
    double (&x)[NumMeasurements],
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
    double (&x)[NumMeasurements],
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

        for (int i = 0; i < NumParameters; i++) {
            derivative[i] = 0.0;
        }

        for (int i = 0; i < NumParameters; i++) {
            for (int j = 0; j < NumMeasurements; j++) {
                hessian[i][j] = 0.0;
            }
        }

        // Build out the jacobian and the hessian matrices
        for (int m = 0; m < NumMeasurements; m++) {
            double currX = x[m];
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
}

template<int NumParameters, int NumMeasurements>
bool MyGTSAMSolver<NumParameters, NumMeasurements>::getCholeskyDecomposition() {


}

template<int NumParameters, int NumMeasurements>
void MyGTSAMSolver<NumParameters, NumMeasurements>::solveCholesky() {
    std::cout << "FOO";
}