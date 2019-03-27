#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>

#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>

#include "prototype/MyGTSAMSolver.h"

const int M = 7000; // Number of measurements
const int N = 3; // Number of parameters: a, b, c

using Solver = MyGTSAMSolver<M, N>;
using EvaluationFunction = Solver::EvaluationFunction;
using GradientFunction = Solver::GradientFunction;
using JacobianMatrix = Solver::JacobianMatrix;
using ParamMatrix = Solver::ParamMatrix;
using XRow = Solver::XRow;
using XMatrix = Solver::XMatrix;


/**
 * A simple linear evaluation function. Using this will likely yield quite low errors.
 */
double dotProductEvaluationFunction(const ParamMatrix &params, const XRow &x) {
    return params[0] * x[0] + params[1] * x[1] + params[2] * x[2];
}


/**
 * An arbitrary (meaningless) nonlinear function for demonstration purposes
 */
double evaluationFunction(const ParamMatrix &params, const XRow &x) {
    return x[0] * params[0] / params[1] + dotProductEvaluationFunction(params, x) + exp(
        (params[0] - params[1] + params[2]) / 30
    );
}


/**
 * Simple gradient function for an estimator. For demo purposes,
 * the gradients are not calculated analytically. Of course, you
 * can get significant runtime benefits by implementing your gradient
 * analytically and vectorize this.
 */
void gradientFunction(JacobianMatrix &jacobian, ParamMatrix &params, const XMatrix &x) {
    float epsilon = 1e-5f;
    XRow jacobianRow;
    for (int m = 0; m < M; m++) {
        for (int iParam = 0; iParam < 3; iParam++) {
            double currParam = params[iParam];

            double paramPlus = currParam + epsilon;
            double paramMinus = currParam - epsilon;

            params[iParam] = paramPlus;
            double evalPlus = evaluationFunction(params, x.row(m));

            params[iParam] = paramMinus;
            double evalMinus = evaluationFunction(params, x.row(m));

            params[iParam] = currParam;

            double derivative = (evalPlus - evalMinus) / (2 * epsilon);

            jacobianRow[iParam] = derivative;
        }
        jacobian.row(m).noalias() = jacobianRow;
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
    std::uniform_real_distribution<double> paramDistribution(0, 10);
    std::normal_distribution<double> yDistribution(0, .5);
    std::normal_distribution<double> xDistribution(0, .5);
    std::uniform_real_distribution<double> xCoords(-100, 100);

    double a = paramDistribution(generator);
    double b = paramDistribution(generator);
    double c = paramDistribution(generator);
    oracleParams[0] = a;
    oracleParams[1] = b;
    oracleParams[2] = c;

    double paramArr[N] = {a, b, c};
    ParamMatrix parameters(&paramArr[0], N, 1);

    std::cout << "Randomly Generated Params" << std::endl;
    std::cout <<  "\t a:" << a << std::endl;
    std::cout <<  "\t b:" << b << std::endl;
    std::cout <<  "\t c:" << c << std::endl << std::endl;

    for (int i = 0; i < M; i++) {
        double x = xCoords(generator);
        double x2 = x * x + xDistribution(generator);
        double x1 = x + xDistribution(generator);
        double x0 = 1 + xDistribution(generator);
        double xArr[N] = {x2, x1, x0};

        XRow xRow(&xArr[0]);

        double noisyY = evaluationFunction(parameters, xRow) + yDistribution(generator);

        xValues[i][0] = xArr[0];
        xValues[i][1] = xArr[1];
        xValues[i][2] = xArr[2];
        yValues[i] = noisyY;
    }
}

int main() {

    // Generate random points to fit
    double oracleParams[N] = {0}; // The actual parameters used to generate the points
    double xValues[M][N] = {0};
    double yValues[M] = {0};
    generatePoints(xValues, yValues, oracleParams);


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
    EvaluationFunction e =  &evaluationFunction;
    GradientFunction g = &gradientFunction;

    // Initialize the solver and fit, which updates initialParams
    // to have the final result. Add a flag so that we crash if
    // a malloc occurs.
    Eigen::internal::set_is_malloc_allowed(false);
    Solver mysolver(e, g, initialParams, xValues, yValues);
    bool result = mysolver.fit();
    std::cout << "Fit success: " <<  result << std::endl;;
    Eigen::internal::set_is_malloc_allowed(true);

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << initialParams[0] << std::endl;
    std::cout << "\tb: " << initialParams[1] << std::endl;
    std::cout << "\tc: " << initialParams[2] << std::endl;
    std::cout << "Error from oracle" << std::endl;
    std::cout << "\ta: " << oracleParams[0] - initialParams[0]  << std::endl;
    std::cout << "\tb: " << oracleParams[1] - initialParams[1]  << std::endl;
    std::cout << "\tc: " << oracleParams[2] - initialParams[2]  << std::endl;
}