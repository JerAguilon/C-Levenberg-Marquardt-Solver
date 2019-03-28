#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>

#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>

#include "solver/LightLevenbergMarquardtOptimizer.h"
#include "solver/DataManipulator.h"

const int M = 7000; // Number of measurements
const int N = 3; // Number of parameters: a, b, c

using Solver = LightLevenbergMarquardtOptimizer<M, N>;
using JacobianMatrix = Solver::JacobianMatrix;
using ParamMatrix = Solver::ParamMatrix;
using XRow = Eigen::Matrix<double, N, 1>;
using XMatrix = Eigen::Map<Eigen::Matrix<double, M, N, Eigen::RowMajor>>;
using YMatrix = Eigen::Map<Eigen::Matrix<double, M, 1>>;

/**
 * A dot product between the parameters and a row of data.
 */
double dotProductEvaluationFunction(const ParamMatrix &params, const XRow &x) {
    return params[0] * x[0] + params[1] * x[1] + params[2] * x[2];
}


/**
 * An arbitrary (meaningless) nonlinear function for demonstration purposes. We will
 * fit this nonlinear function using LM-optimization
 */
double evaluationFunction(const ParamMatrix &params, const XRow &x) {
    return x[0] * params[0] / params[1] + dotProductEvaluationFunction(params, x) + exp(
        (params[0] - params[1] + x[1]) / 30
    );
}


/**
 * This is an implementation of a class that is fed to the LM Solver.
 * It simply implements the logic to fill out a jacobian matrix
 * and a residual matrix, which GTSAM uses to calculate residuals.
 */
class MyManipulator : public DataManipulator<M, N> {
public:
    MyManipulator(double (&x)[M][N], double (&y)[M]):
        x(&x[0][0], M, N),
        y(&y[0], M, 1)
    {}

    /**
     * Fills the jacobian matrix from first principles. Computing a derivative
     * analytically would speed up this operation.
     */
    void fillJacobian(JacobianMatrix &jacobian, ParamMatrix &params) override {
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
     * Fills a residual matrix with according to the evaluation function.
     */
    void fillResiduals(ResidualMatrix &residuals, ParamMatrix &params) override {
        for (int i = 0; i < M; i++) {
            double residual = evaluationFunction(params, x.row(i)) - y[i];
            residuals(i) = residual;
        }
    };
private:
    XMatrix x;
    YMatrix y;

};


/**
 * Generates N (N=3 for this demo) random parameters and uses them to create
 * M points with some Gaussian noise. Fiddle with the noise
 * and notice how the final error increases when the noise increases
 */
void generatePoints(double (&xValues)[M][N], double (&yValues)[M], double(&oracleParams)[N]) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> paramDistribution(0, 10);
    std::normal_distribution<double> yNoise(0, .5);
    std::normal_distribution<double> xNoise(0, .5);
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
        double x2 = x * x + xNoise(generator);
        double x1 = x + xNoise(generator);
        double x0 = 1 + xNoise(generator);
        double xArr[N] = {x2, x1, x0};

        XRow xRow(&xArr[0]);

        double noisyY = evaluationFunction(parameters, xRow) + yNoise(generator);

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


    // Create our data manipulator, which is used to feed the jacobian and residuals
    // to the LM solver
    MyManipulator manipulator(xValues, yValues);

    // Initialize the solver and fit, which updates initialParams
    // to have the final result. Add a flag so that we crash if
    // a malloc occurs.
    Eigen::internal::set_is_malloc_allowed(false);
    Solver mysolver(&manipulator, initialParams);
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