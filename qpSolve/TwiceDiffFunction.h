#ifndef QPSOLVE_TWICEDIFFFUNCTION_H_
#define QPSOLVE_TWICEDIFFFUNCTION_H_

#include <Eigen/Dense>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace qpSolve {

    typedef double (*ObjectiveFunctionPtr)(const VectorXd &x);
    typedef VectorXd (*GradientFunctionPtr)(const VectorXd &x);
    typedef MatrixXd (*HessianFunctionPtr)(const VectorXd &x);

    class TwiceDiffFunction {
    public:
        TwiceDiffFunction();
        TwiceDiffFunction(
            const int &size, 
            const ObjectiveFunctionPtr objectiveFunction,
            const GradientFunctionPtr gradientFunction, 
            const HessianFunctionPtr hessianFunction, 
            const double &derivativeEps,
            const int &numberOfThreads
        );
        TwiceDiffFunction(const TwiceDiffFunction& twiceDiffFunction);

        int getSize() const {return size;}
		double getDerivativeEps() const {return derivativeEps;}

        double evaluateFunction(const VectorXd &x);
        VectorXd evaluateGradient(const VectorXd &x);
        MatrixXd evaluateHessian(const VectorXd &x);
        void evaluateFunctionAndGradient(const VectorXd &x, double &f, VectorXd &g);
        void evaluateFunctionGradientAndHessian(const VectorXd &x, double &f, VectorXd &g, MatrixXd &H);

        void operator=(const TwiceDiffFunction &twiceDiffFunction);

    private:
        class ThreadPool {
            private:
                int m_threads;
                std::vector<std::thread>threads;
                std::queue<std::function<void()>>tasks;
                std::mutex mtx;
                std::condition_variable cv;
                bool stop;

            public:
                explicit ThreadPool(int numThreads);
                ~ThreadPool();

                template<class F, class... Args>
                auto executeTask(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;
        };

    private:
        int size;
        int numberOfThreads;
        double derivativeEps;
        ObjectiveFunctionPtr objectiveFunction;
        GradientFunctionPtr gradientFunction;
        HessianFunctionPtr hessianFunction;
        ThreadPool *pool;

        void numericalGradient(const VectorXd& x, double& f, VectorXd& g);
        void numericalGradientParallel(const VectorXd& x, double& f, VectorXd& g);
        void hessianUsingGradient(const VectorXd& x, VectorXd& g, MatrixXd& H);
        void hessianUsingGradientParallel(const VectorXd& x, VectorXd& g, MatrixXd& H);
        void numericalGradientAndHessian(const VectorXd& x, double& f, VectorXd& g, MatrixXd& H);
        void numericalGradientAndHessianParallel(const VectorXd& x, double& f, VectorXd& g, MatrixXd& H);
        bool isParallel() const {return numberOfThreads > 1;}
    };
}

#endif /* QPSOLVE_TWICEDIFFFUNCTION_H_ */