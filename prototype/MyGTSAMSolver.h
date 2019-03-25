#ifndef MY_GTSAM_SOLVER_H
#define MY_GTSAM_SOLVER_H

#define TOL 1e-30

typedef double (*EvaluationFunction)(double *params, double x);
typedef void (*GradientFunction)(double *gradient, double *params, double x);

template<int NumParameters, int NumMeasurements>
class MyGTSAMSolver {
public:

    EvaluationFunction evaluationFunction;
    GradientFunction gradientFunction;

    double (&parameters)[NumParameters];
    double (&x)[NumMeasurements];
    double (&y)[NumMeasurements];

    MyGTSAMSolver(
        EvaluationFunction evaluationFunction,
        GradientFunction gradientFunction,
        double (&initialParams)[NumParameters],
        double (&x)[NumMeasurements],
        double (&y)[NumMeasurements]
    );

    bool fit();

private:
    double hessian[NumParameters][NumParameters],
           choleskyDecomposition[NumParameters][NumParameters];

    double derivative[NumParameters],
           gradient[NumParameters];

    double newParameters[NumParameters],
           delta[NumParameters];

    double getError(
        double (&parameters)[NumParameters],
        double (&x)[NumMeasurements],
        double (&y)[NumMeasurements]);

    bool getCholeskyDecomposition();
    void solveCholesky();
};

#endif