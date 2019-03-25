#include <iostream>
#include <fstream>

#include "solver/GTSAMSolver.h"
#include "prototype/MyGTSAMSolver.cpp"

const int M = 3; // 100; // Number of measurements
const int N = 3; // Number of parameters: a, b, c

double evaluationFunction(double *params, double x) {
    float a = params[0];
    float b = params[1];
    float c = params[2];
    return a * x * x + b * x + c;
}

void gradientFunction(double *gradient, double *params, double x) {
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


bool fit(
    double (&x)[M],
    double (&y)[M])
{
    return true;
}

int main() {
    double xValues[M] = {0, 1, 2};
    double yValues[M] = {1, 2, 100};

    double initialParams[N] = {-1.99854, 50.0322, 7.90917};

    EvaluationFunction e = &evaluationFunction;
    GradientFunction g = &gradientFunction;

    MyGTSAMSolver<N, M> mysolver(e, g, initialParams, xValues, yValues);

    mysolver.fit();
    // fit(xValues, yValues);

    // Opt result
    //         a: -1.99854
    //         b: 50.0322
    //         c: 7.90917

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << initialParams[0] << std::endl;
    std::cout << "\tb: " << initialParams[1] << std::endl;
    std::cout << "\tc: " << initialParams[2] << std::endl;
}