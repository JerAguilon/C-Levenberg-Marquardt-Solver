#include <iostream>
#include <fstream>
#include <random>
#include <utility>
#include <chrono>

#include "solver/GTSAMSolver.h"
#include "prototype/MyGTSAMSolver.cpp"

const int M = 100; // Number of measurements
const int N = 3; // Number of parameters: a, b, c


/**
 * Simple dot product evaluation function given a sample x and its parameters
 */
double dotProductEvaluationFunction(double *params, double *x) {
    double total = 0;

    for (int i = 0; i < 3; i++) {
        total += params[i] * x[i];
    }
    return total;
}


/**
 * Simple gradient function for an estimator. For demo purposes,
 * the gradients are not calculated analytically. Of course, you
 * can get significant runtime benefits by implementing your gradient
 * analytically.
 */
void gradientFunction(double *gradient, double *params, double *x) {
    float epsilon = 1e-4f;

    for (int iParam = 0; iParam < 3; iParam++) {
        double currParam = params[iParam];

        double paramPlus = currParam + epsilon;
        double paramMinus = currParam - epsilon;


        params[iParam] = paramPlus;
        double evalPlus = dotProductEvaluationFunction(params, x);

        params[iParam] = paramMinus;
        double evalMinus = dotProductEvaluationFunction(params, x);

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
void generatePoints(double (&xValues)[M][N], double (&yValues)[M]) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> paramDistribution(0, 50);
    std::normal_distribution<double> yDistribution(0, .5);
    std::normal_distribution<double> xDistribution(0, .5);

    double a = paramDistribution(generator);
    double b = paramDistribution(generator);
    double c = paramDistribution(generator);

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

        double noisyY = dotProductEvaluationFunction(paramArr, xArr) + yDistribution(generator);

        xValues[x][0] = xArr[0];
        xValues[x][1] = xArr[1];
        xValues[x][2] = xArr[2];
        yValues[x] = noisyY;
    }
}

int main() {

    // Generate random points to fit
    double xValues[M][N] = {0};
    double yValues[M] = {0};
    generatePoints(xValues, yValues);


    // initialize some parameters
    double initialParams[N] = {0};


    // Define pointers towards our evaluation function and gradient function
    EvaluationFunction e = &dotProductEvaluationFunction;
    GradientFunction g = &gradientFunction;

    // Initialize the solver and fit, which updates initialParams
    // to have the final result
    MyGTSAMSolver<N, M> mysolver(e, g, initialParams, xValues, yValues);
    mysolver.fit();

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << initialParams[0] << std::endl;
    std::cout << "\tb: " << initialParams[1] << std::endl;
    std::cout << "\tc: " << initialParams[2] << std::endl;
}