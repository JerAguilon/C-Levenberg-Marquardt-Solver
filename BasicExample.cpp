#include <iostream>
#include <fstream>

#include <Eigen/Eigen>

#include <unsupported/Eigen/NonLinearOptimization>

class MyFunctor
{
    Eigen::MatrixXf measuredValues;
    int m;
    int n;


public:

    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
    {
        float aParam = x(0);
        float bParam = x(1);
        float cParam = x(2);

        for (int i = 0; i < values(); i++) {
            float xValue = measuredValues(i, 0);
            float yValue = measuredValues(i, 1);
            
            fvec(i) = getResidual(aParam, bParam, cParam, xValue, yValue);

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

private:
    float getResidual(float a, float b, float c, float x, float y) const {
        return y - (a * x * x + b * x + c);
    }
};

const int N = 3;
const int M = 5;

int main() {
    std::ifstream

    MyFunctor functor;
    Eigen::VectorXf x(N);

    Eigen::LevenbergMarquardt<MyFunctor, float> lm(functor);
    int result = lm.minimize(x);

    std::cout << "Opt result" << std::endl;
    std::cout << "\ta: " << x(0) << std::endl;
    std::cout << "\tb: " << x(1) << std::endl;
    std::cout << "\tc: " << x(2) << std::endl;
}