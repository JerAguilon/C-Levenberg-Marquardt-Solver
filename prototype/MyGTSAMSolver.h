#ifndef MY_GTSAM_SOLVER_H
#define MY_GTSAM_SOLVER_H

#define TOL 1e-30

template<int NumParameters, int NumMeasurements>
class MyGTSAMSolver {
public:
    double (*evaluationFunction)(double *, int);
    double (*gradientFunction)(double *, int);

    double parameters[NumParameters];

    MyGTSAMSolver(
        double (*evaluationFunction)(double *parameters, double x),
        double (*gradientFunction)(double *gradient, double *parameters, double x),
        double (&initialParams)[NumParameters]
    );

    bool fit(
        double (&x)[NumMeasurements],
        double (&y)[NumMeasurements]
    );

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
        double (&y)[NumMeasurements])
    );

    bool getCholeskyDecomposition();
    void solveCholesky();
};

#endif