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
        double (*evaluationFunction)(double *, double),
        double (*gradientFunction)(double *, double),
        double initialParams[NumParameters]
    );

    bool fit(
        double x[NumMeasurements],
        double y[NumMeasurements]
    );

private:
    double hessian[NumParameters][NumParameters],
           choleskyDecomposition[NumParameters][NumParameters];

    double derivative[NumParameters],
           gradient[NumParameters];

    double newParameters[NumParameters];

    double getError();
};

#endif