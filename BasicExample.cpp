#include <iostream>
#include <fstream>

#include <Eigen/Eigen>

#include <unsupported/Eigen/NonLinearOptimization>


struct QuadraticEvaluationFunction {
    float operator()(const Eigen::VectorXf &params, const Eigen::VectorXf &x) const
    {
        float xf = x(0);
        float a = params(0);
        float b = params(1);
        float c = params(2);
        return a * xf * xf + b * xf + c; 
    }
};

typedef float (*EvaluationFunction)(const Eigen::VectorXf, const Eigen::VectorXf);

template<typename EvaluationFunction>
class MyFunctor
{
    Eigen::MatrixXf measuredValues;
    const int m;
    const int n;
    const EvaluationFunction evalFunction;

public:
    MyFunctor(Eigen::MatrixXf measuredValues, const int m, const int n):
        measuredValues(measuredValues), m(m), n(n), evalFunction(EvaluationFunction())
    {}

    int operator()(const Eigen::VectorXf &params, Eigen::VectorXf &fvec) const
    {

        for (int i = 0; i < values(); i++) {
            float xValue = measuredValues(i, 0);
            float yValue = measuredValues(i, 1);

            Eigen::VectorXf xVector(1);
            xVector(0) = xValue;

            float residual = yValue - evalFunction(params, xVector);
            fvec(i) = residual;
        }
        return 0;
    }

    int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjacobian) {
        float epsilon = 1e-5f;        

        for (int i = 0; i < x.size(); i++) {
            Eigen::VectorXf xPlus(x);
            Eigen::VectorXf xMinus(x);
            xPlus(i) += epsilon;
            xMinus(i) -= epsilon;

            Eigen::VectorXf fVecPlus(values());
            Eigen::VectorXf fVecMinus(values());

            operator()(xPlus, fVecPlus);
            operator()(xMinus, fVecMinus);

            Eigen::VectorXf fvecDiff(values());
            fvecDiff = (fVecPlus - fVecMinus) / (2 * epsilon);

            fjacobian.block(0, i, values(), 1) = fvecDiff;
        }
        return 0;
    }

    int values() const { return m; }

    int inputs() const { return n; }
};

const int N = 3;

int main() {
    std::ifstream infile("measurements.txt");

    if (!infile) {
        std::cout << "measurements.txt could not be read" << std::endl;
        return -1;
    }

    std::vector<float> xValues;
    std::vector<float> yValues;

    std::string currLine;
    while(getline(infile, currLine)) {
        std::istringstream ss(currLine);
        float x, y;
        ss >> x >> y;
        xValues.push_back(x);
        yValues.push_back(y);
    }

    int m = xValues.size();

    Eigen::MatrixXf measuredValues(m, 2);
    for (int i = 0; i < m; i++) {
        measuredValues(i, 0) = xValues[i];            
        measuredValues(i, 1) = yValues[i];
    }

    Eigen::VectorXf x(N);
    x(0) = 0.0;
    x(1) = 0.0;
    x(2) = 0.0;

    using QuadraticFunctor = MyFunctor<QuadraticEvaluationFunction>;
    QuadraticFunctor functor(measuredValues, m, N);

    Eigen::LevenbergMarquardt<QuadraticFunctor, float> lm(functor);
    int result = lm.minimize(x);

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << x(0) << std::endl;
    std::cout << "\tb: " << x(1) << std::endl;
    std::cout << "\tc: " << x(2) << std::endl;
}