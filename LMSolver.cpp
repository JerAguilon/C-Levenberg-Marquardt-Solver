#include <iostream>
#include <fstream>

#include "solver/GTSAMSolver.h"
#include "prototype/MyGTSAMSolver.cpp"

const int M = 100; // Number of measurements
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
    double xValues[M] = { -10.00, -9.50, -9.00, -8.50, -8.00, -7.50, -7.00, -6.50, -6.00, -5.50, -5.00, -4.50, -4.00, -3.50, -3.00, -2.50, -2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50, 14.00, 14.50, 15.00, 15.50, 16.00, 16.50, 17.00, 17.50, 18.00, 18.50, 19.00, 19.50, 20.00, 20.50, 21.00, 21.50, 22.00, 22.50, 23.00, 23.50, 24.00, 24.50, 25.00, 25.50, 26.00, 26.50, 27.00, 27.50, 28.00, 28.50, 29.00, 29.50, 30.00, 30.50, 31.00, 31.50, 32.00, 32.50, 33.00, 33.50, 34.00, 34.50, 35.00, 35.50, 36.00, 36.50, 37.00, 37.50, 38.00, 38.50, 39.00, 39.50 };
    double yValues[M] = {
    }


    double initialParams[N] = {-1.99854, 1, 7.90917};

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