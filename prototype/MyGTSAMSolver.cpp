#include "prototype/MyGTSAMSolver.h"

template<int NumParameters, int NumMeasurements>
MyGTSAMSolver<NumParameters, NumMeasurements>::MyGTSAMSolver(
    double (*evaluationFunction)(double *, double),
    double (*gradientFunction)(double *, double),
    double initialParams[NumParameters]
):
    evaluationFunction(evaluationFunction),
    gradientFunction(gradientFunction),
    parameters{initialParams},
    hessian{},
    choleskyDecomposition{},
    derivative{},
    gradient{},
    newParameters{},
{}

template<int NumParameters, int NumMeasurements>
bool MyGTSAMSolver<NumParameters, NumMeasurements>::fit(
    double x[NumMeasurements],
    double y[NumMeasurements])
{
    // TODO: make these input arguments
    final int maxIterations = 10000;
    double lambda = 0.1;
    double upFactor = 10.0;
    double downFactor = 1.0/10.0;
    double targetErr = 0.01;




}
