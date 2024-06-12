#ifndef TESTS_TEST_FUNCTIONS_H_
#define TESTS_TEST_FUNCTIONS_H_

#include <math.h>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

double function(const VectorXd& x);
VectorXd gradient(const VectorXd& x);
MatrixXd hessian(const VectorXd& x);

double rosenbrock(const VectorXd& x);

double constrainedFunction(const VectorXd &x);
double constrainedFunction3D(const VectorXd& x);

#endif /* TESTS_TEST_FUNCTIONS_H_ */