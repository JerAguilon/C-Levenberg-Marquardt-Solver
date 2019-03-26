#ifndef MY_GTSAM_SOLVER_H
#define MY_GTSAM_SOLVER_H

#define TOL 1e-30

typedef double (*EvaluationFunction)(double *params, double *x);
typedef void (*GradientFunction)(double *gradient, double *params, double *x);

/**
 *  Solves the equation X[RowsMeasurements x RowsParam] * P[RowsParam] = Y[RowsMeasurements]
 *  todo(Jeremy): Add generic support for ColsY
 */
template<int RowsParams, int RowsMeasurements, int ColsY = 1 /* Stubbed since it is unsupported for now */>
class MyGTSAMSolver {
public:

    EvaluationFunction evaluationFunction;
    GradientFunction gradientFunction;

    double (&parameters)[RowsParams];
    double (&x)[RowsMeasurements][RowsParams];
    double (&y)[RowsMeasurements];

    MyGTSAMSolver(
        EvaluationFunction evaluationFunction,
        GradientFunction gradientFunction,
        double (&initialParams)[RowsParams],
        double (&x)[RowsMeasurements][RowsParams],
        double (&y)[RowsMeasurements]
    );

    bool fit();

private:
    double hessian[RowsParams][RowsParams],
           choleskyDecomposition[RowsParams][RowsParams];

    double derivative[RowsParams],
           gradient[RowsParams];

    double newParameters[RowsParams],
           delta[RowsParams];

    double getError(
        double (&parameters)[RowsParams],
        double (&x)[RowsMeasurements][RowsParams],
        double (&y)[RowsMeasurements]);

    bool getCholeskyDecomposition();
    void solveCholesky();
};

#endif