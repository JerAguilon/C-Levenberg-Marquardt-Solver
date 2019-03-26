#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>

#include <Eigen/Core>

#include "prototype/MyGTSAMSolver.h"

const int M = 1000; // Number of measurements
const int N = 3; // Number of parameters: a, b, c


/**
 * A simple linear evaluation function. Using this will likely yield quite low errors.
 */
double dotProductEvaluationFunction(double *params, double *x) {
    return params[0] * x[0] + params[1] * x[1] + params[2] * x[2];
}


/**
 * An arbitrary (meaningless) nonlinear function for demonstration purposes
 */
double evaluationFunction(double *params, double *x) {
    return x[0] * params[0] / params[1] + dotProductEvaluationFunction(params, x) + exp(
        (params[0] - params[1] + params[2]) / 30
    );
}


/**
 * Simple gradient function for an estimator. For demo purposes,
 * the gradients are not calculated analytically. Of course, you
 * can get significant runtime benefits by implementing your gradient
 * analytically.
 */
void gradientFunction(double *gradient, double *params, double *x) {
    float epsilon = 1e-5f;

    for (int iParam = 0; iParam < 3; iParam++) {
        double currParam = params[iParam];

        double paramPlus = currParam + epsilon;
        double paramMinus = currParam - epsilon;


        params[iParam] = paramPlus;
        double evalPlus = evaluationFunction(params, x);

        params[iParam] = paramMinus;
        double evalMinus = evaluationFunction(params, x);

        params[iParam] = currParam;

        double derivative = (evalPlus - evalMinus) / (2 * epsilon);

        gradient[iParam] = derivative;
    }
}



/**
 * Generates N (N=3 for this demo) random parameters and uses them to create
 * M (M=100) quadratic points with some Gaussian noise. Fiddle with the noise
 * and notice how the final error increases when the noise increases
 */
void generatePoints(double (&xValues)[M][N], double (&yValues)[M], double(&oracleParams)[N]) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> paramDistribution(0, 50);
    std::normal_distribution<double> yDistribution(0, .5);
    std::normal_distribution<double> xDistribution(0, .5);

    double a = paramDistribution(generator);
    double b = paramDistribution(generator);
    double c = paramDistribution(generator);
    oracleParams[0] = a;
    oracleParams[1] = b;
    oracleParams[2] = c;

    double paramArr[N] = {a, b, c};

    std::cout << "Randomly Generated Params" << std::endl;
    std::cout <<  "\t a:" << a << std::endl;
    std::cout <<  "\t b:" << b << std::endl;
    std::cout <<  "\t c:" << c << std::endl << std::endl;

    for (int x = 0; x < M; x++) {
        double x2 = x * x + xDistribution(generator);
        double x1 = x + xDistribution(generator);
        double x0 = 1 + xDistribution(generator);
        double xArr[N] = {x2, x1, x0};

        double noisyY = evaluationFunction(paramArr, xArr) + yDistribution(generator);

        xValues[x][0] = xArr[0];
        xValues[x][1] = xArr[1];
        xValues[x][2] = xArr[2];
        yValues[x] = noisyY;
    }
}

int main() {

    // Generate random points to fit
    double oracleParams[N] = {0}; // The actual parameters used to generate the points
    double xValues[M][N] = {0};
    double yValues[M] = {0};
    generatePoints(xValues, yValues, oracleParams);

    double xFlattened[M * N] = {0};
    int k = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            xFlattened[k] = xValues[m][n];
            k++;
        }
    }

    double *yValuesPtr = yValues;
    double *xValuesPtr = xFlattened;

    using YMatrix = Eigen::Matrix<double, M, 1>;
    using XMatrix = Eigen::Matrix<double, M, N>;
    using ParamMatrix = Eigen::Matrix<double, N, 1>;


    // initialize some parameters to some bad estimate of the oracle
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> paramDistribution(-1, 1);
    double initialParams[N] = {
        oracleParams[0] + paramDistribution(generator),
        oracleParams[1] + paramDistribution(generator),
        oracleParams[2] + paramDistribution(generator)
    };
    double *initialParamsPtr = initialParams;



    // Define pointers towards our evaluation function and gradient function
    EvaluationFunction e =  evaluationFunction; GradientFunction g = &gradientFunction;

    // Create Eigen Matrices by passing pointers to the data. This avoids a
    // Malloc as we are simply reusing the data address
    Eigen::Map<XMatrix> xMatrix(xValuesPtr, M, N);
    Eigen::Map<YMatrix> yMatrix(yValuesPtr, M, 1);
    Eigen::Map<ParamMatrix> paramMatrix(initialParams, N, 1);

    // Initialize the solver and fit, which updates initialParams
    // to have the final result
    MyGTSAMSolver<M, N> mysolver(e, g, initialParams, xValues, yValues);
    mysolver.fit();

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << initialParams[0] << std::endl;
    std::cout << "\tb: " << initialParams[1] << std::endl;
    std::cout << "\tc: " << initialParams[2] << std::endl;
    std::cout << "Error from oracle" << std::endl;
    std::cout << "\ta: " << oracleParams[0] - initialParams[0]  << std::endl;
    std::cout << "\tb: " << oracleParams[1] - initialParams[1]  << std::endl;
    std::cout << "\tc: " << oracleParams[2] - initialParams[2]  << std::endl;
}