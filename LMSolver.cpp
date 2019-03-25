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
 * Simple quadratic evaluation function given a point x and its parameters
 */
double evaluationFunction(double *params, double x) {
    float a = params[0];
    float b = params[1];
    float c = params[2];
    return a * x * x + b * x + c;
}


/**
 * Simple gradient function for an estimator. For demo purposes,
 * the gradients are not calculated analytically. Of course, you
 * can get significant runtime benefits by implementing your gradient
 * analytically.
 */
void gradientFunction(double *gradient, double *params, double x) {
    float epsilon = 1e-4f;

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
void generatePoints(double (&xValues)[M], double (&yValues)[M]) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> paramDistribution(0, 2);
    std::normal_distribution<double> yDistribution(0, .7);

    double a = paramDistribution(generator);
    double b = paramDistribution(generator);
    double c = paramDistribution(generator);

    std::cout << "Params: " << a << "," << b << "," << c << std::endl;

    for (int x = 0; x < M; x++) {
        double y = a * x * x + b * x + c;
        y += yDistribution(generator);

        xValues[x] = x;
        yValues[x] = y;
    }
}

int main() {

    // Generate random points to fit
    double xValues[M] = {0};
    double yValues[M] = {0};
    generatePoints(xValues, yValues);


    // initialize some random parameters for a quadratic
    double initialParams[N] = {-1.99854, 1, 7.90917};


    // Define pointers towards our evaluation function and gradient function
    EvaluationFunction e = &evaluationFunction;
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