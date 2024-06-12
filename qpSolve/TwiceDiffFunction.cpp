#include "TwiceDiffFunction.h"
#include <boost/asio/post.hpp>
#include <boost/bind/bind.hpp>

namespace qpSolve {

    TwiceDiffFunction::TwiceDiffFunction() {
        this->size = 0;
        this->objectiveFunction = NULL;
        this->gradientFunction = NULL;
        this->hessianFunction = NULL;
        this->derivativeEps = 1;
        this->numberOfThreads = 1;
    }

    TwiceDiffFunction::TwiceDiffFunction(
            const int &size, 
            const ObjectiveFunctionPtr objectiveFunction,
            const GradientFunctionPtr gradientFunction, 
            const HessianFunctionPtr hessianFunction, 
            const double &derivativeEps,
            const int &numberOfThreads
        ) {
        this->size = size;
        this->objectiveFunction = objectiveFunction;
        this->gradientFunction = gradientFunction;
        this->hessianFunction = hessianFunction;
        this->derivativeEps = derivativeEps;
        this->numberOfThreads = numberOfThreads;
        if (isParallel()) {
            this->pool = new ThreadPool(numberOfThreads);
        }
    }

    TwiceDiffFunction::TwiceDiffFunction(const TwiceDiffFunction &twiceDiffFunction) {
        this->size = twiceDiffFunction.getSize();
        this->objectiveFunction = twiceDiffFunction.objectiveFunction;
        this->gradientFunction = twiceDiffFunction.gradientFunction;
        this->hessianFunction = twiceDiffFunction.hessianFunction;
        this->derivativeEps = twiceDiffFunction.getDerivativeEps();
        this->numberOfThreads = twiceDiffFunction.numberOfThreads;
        if (this->isParallel()) {
            this->pool = new ThreadPool(numberOfThreads);
        }
    }

    double TwiceDiffFunction::evaluateFunction(const VectorXd &x) {
        if (size != x.size()) {
            throw std::invalid_argument("qpSolve::TwiceDiffFunction::evaluateFunction: size != x.size()");
        }

        return objectiveFunction(x);
    }

    VectorXd TwiceDiffFunction::evaluateGradient(const VectorXd &x) {
        if (size != x.size()) {
            throw std::invalid_argument("qpSolve::TwiceDiffFunction::evaluateGradient: size != x.size()");
        }

        if (gradientFunction != NULL) {
            return gradientFunction(x);
        } else {
            double f;
            VectorXd g(size);
            if (isParallel()) {
                numericalGradientParallel(x, f, g);
            } else {
                numericalGradient(x, f, g);
            }
            return g;
        }
    }

    MatrixXd TwiceDiffFunction::evaluateHessian(const VectorXd &x) {
        if (size != x.size()) {
            throw std::invalid_argument("qpSolve::TwiceDiffFunction::evaluateHessian: size != x.size()");
        }

        if (hessianFunction != NULL) {
            return hessianFunction(x);
        } else if (gradientFunction != NULL) {
            VectorXd g(size);
            MatrixXd H(size, size);
            if (isParallel()) {
                hessianUsingGradientParallel(x, g, H);
            } else {
                hessianUsingGradient(x, g, H);
            }
            return H;
        } else {
            double f;
            VectorXd g(size);
            MatrixXd H(size, size);
            if (isParallel()) {
                numericalGradientAndHessianParallel(x, f, g, H);
            } else {
                numericalGradientAndHessian(x, f, g, H);
            }
            return H;
        }
        
    }

    void TwiceDiffFunction::evaluateFunctionAndGradient(const VectorXd &x, double &f, VectorXd &g) {
        if (this->getSize() != x.size()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != x.size()");
        }
        if (this->getSize() != g.size()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != g.size()");
        }

        if (gradientFunction != NULL) {
            f = objectiveFunction(x);
            g = gradientFunction(x);
        } else {
            if (isParallel()) {
                numericalGradientParallel(x, f, g);
            } else {
                numericalGradient(x, f, g);
            }
        }
    }

    void TwiceDiffFunction::evaluateFunctionGradientAndHessian(const VectorXd &x, double &f, VectorXd &g, MatrixXd &H) {
        if (this->getSize() != x.size()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != x.size()");
        }
        if (this->getSize() != g.size()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != g.size()");
        }
        if (this->getSize() != H.rows()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != H.rows()");
        }
        if (this->getSize() != H.cols()) {
            throw std::invalid_argument("qpSolve::Model::evaluateFunctionAndGradient: this->getSize() != H.cols()");
        }

        if (hessianFunction == NULL && gradientFunction == NULL) {
            if (isParallel()) {
                numericalGradientAndHessianParallel(x, f, g, H);
            } else {
                numericalGradientAndHessian(x, f, g, H);
            }
        } else if (hessianFunction == NULL && gradientFunction != NULL) {
            f = objectiveFunction(x);
            if (isParallel()) {
                hessianUsingGradientParallel(x, g, H);
            } else {
                hessianUsingGradient(x, g, H);
            }
        } else if (hessianFunction != NULL && gradientFunction != NULL) {
            f = objectiveFunction(x);
            g = gradientFunction(x);
            H = hessianFunction(x);
        } else {
            H = hessianFunction(x);
            if (isParallel()) {
                numericalGradientParallel(x, f, g);
            } else {
                numericalGradient(x, f, g);
            }
        }
    }

    void TwiceDiffFunction::operator=(const TwiceDiffFunction &twiceDiffFunction) {
        this->size = twiceDiffFunction.getSize();
        this->objectiveFunction = twiceDiffFunction.objectiveFunction;
        this->gradientFunction = twiceDiffFunction.gradientFunction;
        this->hessianFunction = twiceDiffFunction.hessianFunction;
        this->derivativeEps = twiceDiffFunction.getDerivativeEps();
        this->numberOfThreads = twiceDiffFunction.numberOfThreads;
        if (this->isParallel()) {
            this->pool = new ThreadPool(numberOfThreads);
        }
    }

    void TwiceDiffFunction::numericalGradient(const VectorXd &x, double &f, VectorXd &g) {

        VectorXd x_n(x);
        double f_n;

        f = objectiveFunction(x);
        for (int i=0; i<size; i++) {
            x_n(i) += derivativeEps;
            f_n = objectiveFunction(x_n);
            g(i) = (f_n - f) / derivativeEps;
            x_n(i) = x(i);
        }
    }

    void TwiceDiffFunction::numericalGradientParallel(const VectorXd &x, double &f, VectorXd &g) {

        VectorXd x_n(x);
        double f_n;

        std::vector<std::future<double>> results;
        results.reserve(size + 1);

        results.push_back(
            pool->executeTask([this, x]() -> double {
                return objectiveFunction(x);
            })
        );

        for (int i=0; i<size; i++) {
            x_n(i) += derivativeEps;
            results.push_back(
                pool->executeTask([this, x_n]() -> double {
                    return objectiveFunction(x_n);
                })
            );
            x_n(i) = x(i);
        }

        f = results.at(0).get();
        for (int i=0; i<size; i++) {
            f_n = results.at(i+1).get();
            g(i) = (f_n - f) / derivativeEps;
        }
    }

    void TwiceDiffFunction::hessianUsingGradient(const VectorXd &x, VectorXd &g, MatrixXd &H) {
        VectorXd x1(x);
        VectorXd gN(size);
        VectorXd column(size);

        g = gradientFunction(x);

        for (int i=0; i<size; i++) {
            x1(i) += derivativeEps;
            gN = gradientFunction(x1);
            column = (gN - g) / derivativeEps;
            H.col(i) = column;
            x1(i) = x(0);
        }
    }

    void TwiceDiffFunction::hessianUsingGradientParallel(const VectorXd &x, VectorXd &g, MatrixXd &H) {
        VectorXd x1(x);
        VectorXd gN(size);
        VectorXd column(size);

        std::vector<std::future<VectorXd>> results;
        results.reserve(size + 1);

        results.push_back(
            pool->executeTask([this, x]() -> VectorXd {
                return gradientFunction(x);
            })
        );

        for (int i=0; i<size; i++) {
            x1(i) += derivativeEps;
            results.push_back(
                pool->executeTask([this, x1]() -> VectorXd {
                    return gradientFunction(x1);
                })
            );
            x1(i) = x(0);
        }

        g = results.at(0).get();

        for (int i=0; i<size; i++) {
            gN = results.at(i+1).get();
            column = (gN - g) / derivativeEps;
            H.col(i) = column;
        }
    }

    void TwiceDiffFunction::numericalGradientAndHessian(const VectorXd &x, double &f, VectorXd &g, MatrixXd &H) {
        int i,j;
        int combination;
        MatrixXd functionValueForXX(size, 2);
        VectorXd functionValueForXY((int)(size * (size - 1) / 2));

        VectorXd x1(x);
        VectorXd x2(x);

        double derivativeEpsSquare = derivativeEps * derivativeEps;

        f = objectiveFunction(x);

        for (i=0; i<size; i++) {
            x1(i) += derivativeEps;
            x2(i) += derivativeEps * 2;
            functionValueForXX(i, 0) = objectiveFunction(x1);
            functionValueForXX(i, 1) = objectiveFunction(x2);
            g(i) = (functionValueForXX(i, 0) - f) / derivativeEps;
            x1(i) = x(i);
            x2(i) = x(i);
        }

        combination = 0;
        for (i=0; i<size; i++) {
            x1(i) += derivativeEps;
            for (j=i+1; j<size; j++) {
                x1(j) += derivativeEps;
                functionValueForXY(combination) = objectiveFunction(x1);
                combination++;
                x1(j) = x(j);
            }
            x1(i) = x(i);
        }

        combination = 0;
        for (i=0; i<size; i++) {
            for (j=i; j<size; j++) {
                if (j == i) {
                    H(i, j) = (functionValueForXX(i, 1) - 2*functionValueForXX(i, 0) + f) / derivativeEpsSquare;
                }
                else if (j > i) {
                    H(i, j) = (functionValueForXY(combination) - functionValueForXX(j, 0) - functionValueForXX(i, 0) + f) / derivativeEpsSquare;
                    H(j, i) = H(i, j);
                    combination++;
                }
            }
        }
    }

    void TwiceDiffFunction::numericalGradientAndHessianParallel(const VectorXd &x, double &f, VectorXd &g, MatrixXd &H) {
        int i,j;
        int combination;
        MatrixXd functionValueForXX(size, 2);
        VectorXd functionValueForXY((int)(size * (size - 1) / 2));

        VectorXd x1(x);
        VectorXd x2(x);

        std::vector<std::future<double>> results;

        double derivativeEpsSquare = derivativeEps * derivativeEps;
        results.reserve(size*2 + (int)(size * (size - 1) / 2) + 1);

        results.push_back(
            pool->executeTask([this, x]() -> double {
                return objectiveFunction(x);
            })
        );

        for (int i=0; i<size; i++) {
            x1(i) += derivativeEps;
            x2(i) += derivativeEps * 2;
            results.push_back(
                pool->executeTask([this, x1]() -> double {
                    return objectiveFunction(x1);
                })
            );
            results.push_back(
                pool->executeTask([this, x2]() -> double {
                    return objectiveFunction(x2);
                })
            );
            x1(i) = x(i);
            x2(i) = x(i);
        }

        for (i=0; i<size; i++) {
            x1(i) += derivativeEps;
            for (j=i+1; j<size; j++) {
                x1(j) += derivativeEps;
                results.push_back(
                    pool->executeTask([this, x1]() -> double {
                        return objectiveFunction(x1);
                    })
                );
                x1(j) = x(j);
            }
            x1(i) = x(i);
        }

        f = results.at(0).get();

        for (i=0; i<size; i++) {
            functionValueForXX(i, 0) = results.at((i*2)+1).get();
            functionValueForXX(i, 1) = results.at((i*2)+2).get();
            g(i) = (functionValueForXX(i, 0) - f) / derivativeEps;
        }

        combination = 0;
        for (i=0; i<size; i++) {
            for (j=i+1; j<size; j++) {
                functionValueForXY(combination) = results.at(combination + 2*size + 1).get();
                combination++;
            }
        }

        combination = 0;
        for (i=0; i<size; i++) {
            for (j=i; j<size; j++) {
                if (j == i) {
                    H(i, j) = (functionValueForXX(i, 1) - 2*functionValueForXX(i, 0) + f) / derivativeEpsSquare;
                }
                else if (j > i) {
                    H(i, j) = (functionValueForXY(combination) - functionValueForXX(j, 0) - functionValueForXX(i, 0) + f) / derivativeEpsSquare;
                    H(j, i) = H(i, j);
                    combination++;
                }
            }
        }
    }

    TwiceDiffFunction::ThreadPool::ThreadPool(int numThreads) {
        this->m_threads = numThreads;
        this->stop = false;
        for(int i = 0; i<m_threads; i++) {
            threads.emplace_back([this] {
                std::function<void()>task;
                while(1){
                std::unique_lock<std::mutex>lock(mtx);
                cv.wait(lock,[this] {
                    return !tasks.empty()|| stop;
                });
                if(stop)
                    return;
                task = move(tasks.front());
                tasks.pop();
                lock.unlock();
                task();
                }
            });
        }
    }

    TwiceDiffFunction::ThreadPool::~ThreadPool(){
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
        lock.unlock();
        cv.notify_all();
        for(auto& th: threads) {
            th.join();
        }
    }

    template<class F, class... Args>
    auto TwiceDiffFunction::ThreadPool::executeTask(F&& f, Args&&... args)-> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(
                std::forward<F>(f),
                std::forward<Args>(args)...)
        );

        std::future<return_type> res =task->get_future();

        std::unique_lock<std::mutex>lock(mtx);
        tasks.emplace([task]()-> void {(*task)();});
        lock.unlock();
        cv.notify_one();
        return res;
    }
}
