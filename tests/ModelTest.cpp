#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <Model.h>
#include <ModelBuilder.h>
#include "TestFunctions.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

BOOST_AUTO_TEST_SUITE(Model)

BOOST_AUTO_TEST_CASE(ModelUsingTwiceDiffFunction) {

    qpSolve::TwiceDiffFunction twiceDiffFunction(2, function, NULL, NULL, 1e-4, 1);
    qpSolve::Model model(twiceDiffFunction);

    BOOST_CHECK_EQUAL(2, model.getProblemSize());

    qpSolve::TwiceDiffFunction objective = model.getObjectiveFunction();
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;
    BOOST_CHECK_CLOSE(7, objective.evaluateFunction(x), 0.1);
}

BOOST_AUTO_TEST_CASE(ModelBuilderMinimal) {

    int size = 2;
    qpSolve::ModelBuilder modelBuilder = qpSolve::Model::getBuilder();

    qpSolve::Model model = modelBuilder
            .setProblemSize(size)
            .setObjectiveFunction(function)
            .build();

    BOOST_CHECK_EQUAL(size, model.getProblemSize());

    qpSolve::TwiceDiffFunction objective = model.getObjectiveFunction();
    BOOST_CHECK_CLOSE(1e-3, objective.getDerivativeEps(), 0.001);

    VectorXd x(size);
    x(0) = 2;
    x(1) = 1;
    BOOST_CHECK_CLOSE(7, objective.evaluateFunction(x), 0.1);
}

BOOST_AUTO_TEST_CASE(ModelBuilderFull) {

    int size = 2;
    double eps = 1e-4;
    double tolerancePercent = 0.001;

    qpSolve::Model model = qpSolve::Model::getBuilder()
            .setProblemSize(size)
            .setObjectiveFunction(function)
            .setGradientFunction(gradient)
            .setHessianFunction(hessian)
            .setDerivativeEps(eps)
            .build();

    qpSolve::TwiceDiffFunction objective = model.getObjectiveFunction();
    BOOST_CHECK_CLOSE(eps, objective.getDerivativeEps(), tolerancePercent);

    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    objective.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 1), hessian(0, 1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(1, 0), hessian(1, 0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_AUTO_TEST_CASE(ModelBuilderInvalidArguments) {

    qpSolve::ModelBuilder modelBuilder = qpSolve::Model::getBuilder();

    modelBuilder.setProblemSize(-1);
    BOOST_CHECK_THROW(modelBuilder.build(), std::invalid_argument);

    modelBuilder.setProblemSize(2);
    BOOST_CHECK_THROW(modelBuilder.build(), std::invalid_argument);

    modelBuilder.setObjectiveFunction(function);
    modelBuilder.setDerivativeEps(-0.01);
    BOOST_CHECK_THROW(modelBuilder.build(), std::invalid_argument);

    modelBuilder.setDerivativeEps(1e-4);
    BOOST_CHECK_NO_THROW(modelBuilder.build());
}

BOOST_AUTO_TEST_CASE(ModelBuilderWithConstraint) {

    int size = 2;
    qpSolve::ModelBuilder modelBuilder = qpSolve::Model::getBuilder();

    std::string name = "test EQ";
    double rhs = 3;
    qpSolve::ConstraintSign sign = qpSolve::ConstraintSign::EQ;
    VectorXd coefficients(2);
    coefficients(0) = 2;
    coefficients(1) = -1;
    qpSolve::Constraint constraint(coefficients, rhs, sign, name);

    qpSolve::Model model = modelBuilder
            .setProblemSize(size)
            .setObjectiveFunction(function)
            .addConstraint(constraint)
            .build();

    BOOST_CHECK_EQUAL(size, model.getProblemSize());

    std::vector<qpSolve::Constraint> constraints = model.getConstraints();
    BOOST_CHECK_EQUAL(1, constraints.size());

    VectorXd x(size);
    x(0) = 2;
    x(1) = 1;
    BOOST_CHECK_EQUAL(0, constraints.at(0).residual(x));
    BOOST_CHECK_EQUAL(name, constraints.at(0).getName());
    BOOST_CHECK_EQUAL(sign, constraints.at(0).getSign());
}

BOOST_AUTO_TEST_CASE(ModelBuilderWithConstraints) {

    int size = 2;
    qpSolve::ModelBuilder modelBuilder = qpSolve::Model::getBuilder();

    std::string name = "test EQ";
    double rhs = 3;
    qpSolve::ConstraintSign sign = qpSolve::ConstraintSign::EQ;
    VectorXd coefficients(2);
    coefficients(0) = 2;
    coefficients(1) = -1;
    qpSolve::Constraint constraint(coefficients, rhs, sign, name);

    std::vector<qpSolve::Constraint> inputConstraints;
    inputConstraints.push_back(constraint);
    inputConstraints.push_back(constraint);
    inputConstraints.push_back(constraint);

    qpSolve::Model model = modelBuilder
            .setProblemSize(size)
            .setObjectiveFunction(function)
            .addConstraints(inputConstraints)
            .build();

    std::vector<qpSolve::Constraint> constraints = model.getConstraints();
    BOOST_CHECK_EQUAL(3, constraints.size());
}

BOOST_AUTO_TEST_CASE(ModelBuilderWithInvalidConstraintSize) {

    int size = 2;
    qpSolve::ModelBuilder modelBuilder = qpSolve::Model::getBuilder();

    std::string name = "test EQ";
    double rhs = 3;
    qpSolve::ConstraintSign sign = qpSolve::ConstraintSign::EQ;
    VectorXd coefficients(3);
    coefficients(0) = 2;
    coefficients(1) = -1;
    coefficients(2) = -10;
    qpSolve::Constraint constraint(coefficients, rhs, sign, name);

     modelBuilder
            .setProblemSize(size)
            .setObjectiveFunction(function)
            .addConstraint(constraint);

    BOOST_CHECK_THROW(modelBuilder.build(), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()