#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "Constraint.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace qpSolve;

BOOST_AUTO_TEST_SUITE(Constraints)

BOOST_AUTO_TEST_CASE(EqConstraint) {

    std::string name = "test EQ";
    double rhs = 3;
    ConstraintSign sign = ConstraintSign::EQ;

    VectorXd coefficients(2);
    coefficients(0) = 2;
    coefficients(1) = -1;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(coefficients, rhs, sign, name);

    BOOST_CHECK_EQUAL(sign, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(1.0, constraint.residual(x));
    BOOST_CHECK_EQUAL(4.0, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(LeqConstraint) {

    std::string name = "test LEQ";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::LEQ;

    VectorXd coefficients(2);
    coefficients(0) = -4;
    coefficients(1) = 3;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(coefficients, rhs, sign, name);

    BOOST_CHECK_EQUAL(false, constraint.isSignSwapped());
    BOOST_CHECK_EQUAL(sign, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(-9.0, constraint.residual(x));
    BOOST_CHECK_EQUAL(-10.0, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(GeqConstraint) {

    std::string name = "test GEQ";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::GEQ;

    VectorXd coefficients(2);
    coefficients(0) = -4;
    coefficients(1) = 3;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(coefficients, rhs, sign, name);

    BOOST_CHECK_EQUAL(sign, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(-9.0, constraint.residual(x));
    BOOST_CHECK_EQUAL(-10.0, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(GeqBoundConstraint) {

    std::string name = "test GEQ bound";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::GEQ;
    int n = 2;
    int index = 0;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(n, index, rhs, sign, name);

    BOOST_CHECK_EQUAL(true, constraint.isBoundConstarint());
    BOOST_CHECK_EQUAL(sign, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(2, constraint.residual(x));
    BOOST_CHECK_EQUAL(1, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(LeqBoundConstraint) {

    std::string name = "test LEQ bound";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::LEQ;
    int n = 2;
    int index = 1;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(n, index, rhs, sign, name);

    BOOST_CHECK_EQUAL(true, constraint.isBoundConstarint());
    BOOST_CHECK_EQUAL(false, constraint.isSignSwapped());
    BOOST_CHECK_EQUAL(sign, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(-1, constraint.residual(x));
    BOOST_CHECK_EQUAL(-2, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(LeqConstraint_swapSign) {

    std::string name = "test LEQ bound";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::LEQ;
    int n = 2;
    int index = 1;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(n, index, rhs, sign, name);
    constraint.swapSign();

    BOOST_CHECK_EQUAL(true, constraint.isSignSwapped());
    BOOST_CHECK_EQUAL(ConstraintSign::GEQ, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(1, constraint.residual(x));
    BOOST_CHECK_EQUAL(2, constraint.value(x));
}

BOOST_AUTO_TEST_CASE(LeqBoundConstraint_swapSign) {

    std::string name = "test LEQ";
    double rhs = -1;
    ConstraintSign sign = ConstraintSign::LEQ;

    VectorXd coefficients(2);
    coefficients(0) = -4;
    coefficients(1) = 3;

    VectorXd x(2);
    x(0) = 1;
    x(1) = -2;

    Constraint constraint(coefficients, rhs, sign, name);
    constraint.swapSign();

    BOOST_CHECK_EQUAL(true, constraint.isSignSwapped());
    BOOST_CHECK_EQUAL(ConstraintSign::GEQ, constraint.getSign());
    BOOST_CHECK_EQUAL(name, constraint.getName());
    BOOST_CHECK_EQUAL(9, constraint.residual(x));
    BOOST_CHECK_EQUAL(10, constraint.value(x));
}

BOOST_AUTO_TEST_SUITE_END()