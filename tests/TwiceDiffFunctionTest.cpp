#include <boost/test/unit_test.hpp>
#include <math.h>
#include <Eigen/Dense>
#include <TwiceDiffFunction.h>
#include "TestFunctions.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct functionOnly {
    functionOnly() : twiceDiffFunction(2, function, NULL, NULL, 1e-4, 1) {}
    ~functionOnly() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

struct functionOnlyParallel {
    functionOnlyParallel() : twiceDiffFunction(2, function, NULL, NULL, 1e-4, 4) {}
    ~functionOnlyParallel() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

struct functionAndGradient {
    functionAndGradient() : twiceDiffFunction(2, function, gradient, NULL, 1e-4, 1) {}
    ~functionAndGradient() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

struct functionAndGradientParallel {
    functionAndGradientParallel() : twiceDiffFunction(2, function, gradient, NULL, 1e-4, 3) {}
    ~functionAndGradientParallel() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

struct functionGradientAndHessian {
    functionGradientAndHessian() : twiceDiffFunction(2, function, gradient, hessian, 1e-4, 1) {}
    ~functionGradientAndHessian() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

struct functionAndHessian {
    functionAndHessian() : twiceDiffFunction(2, function, NULL, hessian, 1e-4, 1) {}
    ~functionAndHessian() {}

    qpSolve::TwiceDiffFunction twiceDiffFunction;
    double tolerancePercent = 0.1; // 0.1% -> 1e-3
    double toleranceAbs = 1e-4;
};

BOOST_AUTO_TEST_SUITE(TwiceDiffFunction)

BOOST_FIXTURE_TEST_CASE(getSize, functionOnly) {

    BOOST_CHECK_EQUAL(2, twiceDiffFunction.getSize());
}

BOOST_FIXTURE_TEST_CASE(getDerivativeEps, functionOnly) {

    BOOST_CHECK_CLOSE(1e-4, twiceDiffFunction.getDerivativeEps(), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunction, functionOnly) {

    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;
    BOOST_CHECK_CLOSE(7, twiceDiffFunction.evaluateFunction(x), tolerancePercent);

    x(0) = 3;
    x(1) = -1.5;
    BOOST_CHECK_CLOSE(15.75, twiceDiffFunction.evaluateFunction(x), tolerancePercent);

    VectorXd xFail(1);
    xFail(0) = 5;
    BOOST_CHECK_THROW(twiceDiffFunction.evaluateFunction(xFail), std::invalid_argument);
}

BOOST_FIXTURE_TEST_CASE(evaluateGradient, functionOnly) {
    
    VectorXd x(2);
    x(0) = -3;
    x(1) = 3;

    VectorXd gradient(2);
    gradient = twiceDiffFunction.evaluateGradient(x);

    VectorXd expectedGradinet {{-6.0, 18.0}};
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateGradientParallel, functionOnlyParallel) {
    
    VectorXd x(2);
    x(0) = -3;
    x(1) = 3;

    VectorXd gradient(2);
    gradient = twiceDiffFunction.evaluateGradient(x);

    VectorXd expectedGradinet {{-6.0, 18.0}};
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradient, functionOnly) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    twiceDiffFunction.evaluateFunctionAndGradient(x, f, gradient);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradientParallel, functionOnlyParallel) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    twiceDiffFunction.evaluateFunctionAndGradient(x, f, gradient);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateHessian, functionOnly) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    MatrixXd hessian = twiceDiffFunction.evaluateHessian(x);
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateHessianParallel, functionOnlyParallel) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    MatrixXd hessian = twiceDiffFunction.evaluateHessian(x);
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionGradientAndHessian, functionOnly) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionGradientAndHessianParallel, functionOnlyParallel) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateGradient_WithGradientFunction, functionAndGradient) {
    
    VectorXd x(2);
    x(0) = -3;
    x(1) = 3;

    VectorXd gradient(2);
    gradient = twiceDiffFunction.evaluateGradient(x);

    VectorXd expectedGradinet {{-6.0, 18.0}};
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradient_WithGradientFunction, functionAndGradient) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    twiceDiffFunction.evaluateFunctionAndGradient(x, f, gradient);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateHessian_WithGradientFunction, functionAndGradient) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    MatrixXd hessian = twiceDiffFunction.evaluateHessian(x);
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateHessian_WithGradientFunctionParallel, functionAndGradientParallel) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    MatrixXd hessian = twiceDiffFunction.evaluateHessian(x);
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradientAndHessian_WithGradientFunction, functionAndGradient) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradientAndHessian_WithGradientFunctionParallel, functionAndGradientParallel) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateHessian_WithGradientAndHessianFunctions, functionGradientAndHessian) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    MatrixXd hessian = twiceDiffFunction.evaluateHessian(x);
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradientAndHessian_WithGradientAndHessianFunctions, functionGradientAndHessian) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_FIXTURE_TEST_CASE(evaluateFunctionAndGradientAndHessian_WithHessianFunction, functionAndHessian) {
    
    VectorXd x(2);
    x(0) = 2;
    x(1) = 1;

    double f;
    VectorXd gradient(2);
    MatrixXd hessian(2, 2);
    twiceDiffFunction.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

    double expectedF = 7;
    VectorXd expectedGradinet {{4.0, 6.0}};
    MatrixXd expectedHessian {{2.0, 0.0}, {0.0, 6.0}};

    BOOST_CHECK_CLOSE(expectedF, f, tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(0), gradient(0), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedGradinet(1), gradient(1), tolerancePercent);
    BOOST_CHECK_CLOSE(expectedHessian(0, 0), hessian(0, 0), tolerancePercent);
    BOOST_CHECK_SMALL(hessian(0, 1) - expectedHessian(0, 1), toleranceAbs);
    BOOST_CHECK_SMALL(hessian(1, 0) - expectedHessian(1, 0), toleranceAbs);
    BOOST_CHECK_CLOSE(expectedHessian(1, 1), hessian(1, 1), tolerancePercent);
}

BOOST_AUTO_TEST_SUITE_END() 