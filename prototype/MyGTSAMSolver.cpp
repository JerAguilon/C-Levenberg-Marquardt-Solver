#include <algorithm>

#include "prototype/MyGTSAMSolver.h"

template<int NumParameters, int NumMeasurements>
MyGTSAMSolver<NumParameters, NumMeasurements>::MyGTSAMSolver(
    double (*evaluationFunction)(double *parameters, double x),
    double (*gradientFunction)(double *gradient, double *parameters, double x),
    double (&initialParams)[NumParameters]
):
    evaluationFunction(evaluationFunction),
    gradientFunction(gradientFunction),
    parameters{initialParams},
    hessian{},
    choleskyDecomposition{},
    derivative{},
    gradient{},
    delta{},
    newParameters{},
{}

template<int NumParameters, int NumMeasurements>
bool MyGTSAMSolver<NumParameters, NumMeasurements>::fit(
    double (&x)[NumMeasurements],
    double (&y)[NumMeasurements])
{
    // TODO: make these input arguments
    int maxIterations = 10000;
    double lambda = 0.1;
    double upFactor = 10.0;
    double downFactor = 1.0/10.0;
    double targetDeltaError = 0.01;

    double currentError = getError(parameters, x, y);

    int iteration;
    for (iteration=0; iteration < maxIterations; iteration++) {
        std::fill(derivative[0], derivative[0] + NumParameters, 0);
        std::fill(hessian[0], hessian[0] + NumParameters * NumParameters, 0);

        // Build out the jacobian and the hessian matrices
        for (int m = 0; m < NumMeasurements; m++) {
            double x = x[m];
            double y = y[m];
            gradientFunction(gradient, parameters, x);

            for (int i = 0; i < NumParameters; i++) {
                // J_i = residual * gradient
                d[i] += (y - evaluationFunction(parameters, x)) * g[i];
                // H = J^T * J
                for (int j = 0; j <=i; j++) {
                    hessian[i][j] += g[i]*g[j];
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
                multFactor = (1 + lambda * up) / (1 + lambda);
                lambda *= up;
                iteration++;
            }
        }

        for (int i = 0; i < NumParameters; i++) {
            parameters[i] = newParameters[i];
        }

        currentError = newError;
        lambda *= down;

        if (!illConditioned && (-deltaError < targetDeltaError)) break;
    }
}

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