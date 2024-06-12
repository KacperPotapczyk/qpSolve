#include "TestFunctions.h"
#include <thread>
#include <chrono>
#include <iostream>

// simple f(0, 0) = 0
double function(const VectorXd &x) {
    return pow(x(0), 2) + 3*pow(x(1), 2);
}

VectorXd gradient(const VectorXd &x) {
    VectorXd g(2);
    g(0) = 2*x(0);
    g(1) = 6*x(1);
    return g;
}

MatrixXd hessian(const VectorXd &x) {
    MatrixXd H {{2.0, 0.0}, {0.0, 6.0}};
    return H;
}

// rosenbrock f(a, a^2) = 0
double rosenbrock(const VectorXd &x) {
    double a = 1.0;
    double b = 100;
    return pow(a-x(0), 2) + b*pow(x(1)-pow(x(0), 2), 2);
}

// constrained
double constrainedFunction(const VectorXd &x) {
    //	solution:
    //	x(0) = -1
    //	x(1) = 0
    //	f(x) = 0
	return 2*std::pow(x(0), 2) + 4*x(0) + 2 + 0.5*std::pow(x(1), 2);
}

// constrained 3D
double constrainedFunction3D(const VectorXd& x) {
//	solution:
//	x(0) = -1
//	x(1) = 0
//	x(2) = 0
//	f(x) = 0
	return 2*std::pow(x(0), 2) + 4*x(0) + 2 + 0.5*std::pow(x(1), 2) + std::pow(x(2), 2);
}