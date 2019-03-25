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
    double yValues[M] = { -685.80, -647.10, -602.00, -548.90, -524.20, -490.10, -430.60, -412.10, -345.20, -344.10, -306.00, -255.50, -226.00, -195.90, -139.00, -129.50, -84.80, -86.50, -61.00, -18.70, -3.40, 20.90, 41.60, 86.30, 98.80, 122.70, 149.00, 149.10, 158.80, 188.10, 214.20, 223.70, 231.40, 257.50, 269.80, 274.50, 287.40, 278.70, 304.40, 323.30, 316.60, 329.10, 299.20, 333.50, 340.60, 311.10, 313.80, 302.50, 324.80, 316.70, 298.20, 292.10, 286.00, 297.30, 266.80, 269.30, 276.20, 257.90, 221.80, 216.50, 196.80, 211.50, 194.20, 152.70, 148.80, 128.10, 93.60, 83.50, 50.40, 28.30, 20.40, -34.10, -53.80, -80.90, -78.60, -128.10, -164.20, -178.30, -207.40, -241.50, -299.00, -324.10, -348.60, -408.30, -446.80, -486.10, -499.40, -558.90, -582.60, -651.70, -706.20, -727.10, -799.60, -839.50, -879.00, -931.50, -969.80, -1041.50 -1066.40, -1119.30 };

    double initialParams[N] = {0, 1, 2};

    EvaluationFunction e = &evaluationFunction;
    GradientFunction g = &gradientFunction;

    MyGTSAMSolver<N, M> mysolver(e, g, initialParams, xValues, yValues);

    mysolver.fit();
    // fit(xValues, yValues);

    // Opt result
    //         a: -1.99854
    //         b: 50.0322
    //         c: 7.90917

    // std::cout << "Opt result" << std::endl;
    // std::cout << "\ta: " << x(0) << std::endl;
    // std::cout << "\tb: " << x(1) << std::endl;
    // std::cout << "\tc: " << x(2) << std::endl;
}