#include <boost/test/unit_test.hpp>
#include <math.h>
#include <Eigen/Dense>
#include <Solver.h>
#include <ModelBuilder.h>
#include "TestFunctions.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace qpSolve;

BOOST_AUTO_TEST_SUITE(QP)

BOOST_AUTO_TEST_CASE(Simple) {

    double tolerance = 0.001;
    SolverSettings settings;
    // settings.enableDebugInfo();
    int n = 2;

    VectorXd x(n);
    x(0) = 2;
    x(1) = 1;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 0;
    expectedSolution(1) = 0;
    double expectedObjectiveValue = 0.0;

    Model model = Model::getBuilder()
            .setProblemSize(n)
            .setObjectiveFunction(function)
            .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(Simple_parallel) {

    double tolerance = 0.001;
    SolverSettings settings;
    // settings.enableDebugInfo();
    int n = 2;
    int numberOfThreads = 4;

    VectorXd x(n);
    x(0) = 2;
    x(1) = 1;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 0;
    expectedSolution(1) = 0;
    double expectedObjectiveValue = 0.0;

    Model model = Model::getBuilder()
            .setProblemSize(n)
            .setObjectiveFunction(function)
            .setNumberOfThreads(numberOfThreads)
            .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(Rosenbrock) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;

    VectorXd x(n);
    x(0) = -1;
    x(1) = -1;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 1;
    expectedSolution(1) = 1;
    double expectedObjectiveValue = 0.0;

    Model model = Model::getBuilder()
            .setProblemSize(n)
            .setObjectiveFunction(rosenbrock)
            .setDerivativeEps(1e-6)
            .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(Rosenbrock_parallel) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;
    int numberOfThreads = 4;

    VectorXd x(n);
    x(0) = -1;
    x(1) = -1;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 1;
    expectedSolution(1) = 1;
    double expectedObjectiveValue = 0.0;

    Model model = Model::getBuilder()
            .setProblemSize(n)
            .setObjectiveFunction(rosenbrock)
            .setDerivativeEps(1e-6)
            .setNumberOfThreads(numberOfThreads)
            .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(Rosenbrock_RestrictStep) {

    int n = 2;
    double tolerance = 0.001;
    SolverSettings settings;
    settings.enableRestrictedStep(0.5);
    settings.setMaxIterations(20);
    settings.enableSaveIntermediateSolutions();

    VectorXd x(n);
    x(0) = -1;
    x(1) = -1;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 1;
    expectedSolution(1) = 1;
    double expectedObjectiveValue = 0.0;

    Model model = Model::getBuilder()
            .setProblemSize(n)
            .setObjectiveFunction(rosenbrock)
            .setDerivativeEps(1e-6)
            .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    BOOST_CHECK_EQUAL(solution.getIteration()+1, solution.getIntermediateSolutions().size());   
}

BOOST_AUTO_TEST_CASE(constrainedFunction_OneGeqConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;
    
    VectorXd x(n);
    x(0) = 20;
    x(1) = 20;

    VectorXd expectedSolution(n);
    expectedSolution(0) = 0;
    expectedSolution(1) = 0;
    double expectedObjectiveValue = 2.0;

    // x0 >= rhs
    VectorXd coefficients(n);
    coefficients(0) = 1;
    coefficients(1) = 0;
    double rhs = 0.0;
    std::string name = "x0 >= 0";
    Constraint constraint(coefficients, rhs, ConstraintSign::GEQ, name);

    Model model = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6)
        .addConstraint(constraint)
        .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(constrainedFunction_OneLeqConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;
    
    VectorXd x(n);
    x(0) = -20;
    x(1) = -20;

    VectorXd expectedSolution(n);
    expectedSolution(0) = -2;
    expectedSolution(1) = 0;
    double expectedObjectiveValue = 2.0;

    // x0 >= rhs
    VectorXd coefficients(n);
    coefficients(0) = 1;
    coefficients(1) = 0;
    double rhs = -2.0;
    std::string name = "x0 <= -2";
    Constraint constraint(coefficients, rhs, ConstraintSign::LEQ, name);

    Model model = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6)
        .addConstraint(constraint)
        .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(constrainedFunction_TwoGeqConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;
    VectorXd x(n);
    x(0) = 20;
    x(1) = 20;

    ModelBuilder builder = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6);

    VectorXd coefficients(n);
    double rhs;

    // x0 >= rhs
    coefficients(0) = 1;
    coefficients(1) = 0;
    rhs = 0.0;
    Constraint constraint1(coefficients, rhs, ConstraintSign::GEQ, "x0 >= 0.0");
    builder.addConstraint(constraint1);

    // x1 >= rhs
    coefficients(0) = 0;
    coefficients(1) = 1;
    rhs = 1.0;
    Constraint constraint2(coefficients, rhs, ConstraintSign::GEQ, "x1 >= 1.0");
    builder.addConstraint(constraint2);

    VectorXd expectedSolution(n);
    expectedSolution(0) = 0;
    expectedSolution(1) = 1;
    double expectedObjectiveValue = 2.5;

    Model model = builder.build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());
    BOOST_CHECK_EQUAL(SolutionStatus::fixed_on_constraints, solution.getStatus());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
}

BOOST_AUTO_TEST_CASE(constrainedFunction_TwoLeqConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    int n = 2;
    
    VectorXd x(n);
    x(0) = -20;
    x(1) = -20;

    VectorXd expectedSolution(n);
    expectedSolution(0) = -2;
    expectedSolution(1) = -1;
    double expectedObjectiveValue = 2.5;

    VectorXd coefficients(n);
    double rhs;
    std::string name;

    coefficients(0) = 1;
    coefficients(1) = 0;
    rhs = -2.0;
    name = "x0 <= -2";
    Constraint constraint1(coefficients, rhs, ConstraintSign::LEQ, name);

    coefficients(0) = 0;
    coefficients(1) = 1;
    rhs = -1.0;
    name = "x1 <= -1";
    Constraint constraint2(coefficients, rhs, ConstraintSign::LEQ, name);

    Model model = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6)
        .addConstraint(constraint1)
        .addConstraint(constraint2)
        .build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());
    BOOST_CHECK_EQUAL(SolutionStatus::fixed_on_constraints, solution.getStatus());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    BOOST_CHECK_EQUAL(0, solution.getActiveConstraints().size());
}


BOOST_AUTO_TEST_CASE(constrainedFunction_OneEqAndTwoGeqConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    settings.enableSaveActiveConstraints();
    int n = 2;
    VectorXd x(n);
    x(0) = 20;
    x(1) = 20;

    ModelBuilder builder = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6);

    VectorXd coefficients(n);
    double rhs;

    // x0 >= 0.0
    coefficients(0) = 1;
    coefficients(1) = 0;
    rhs = 0.0;
    Constraint constraint1(coefficients, rhs, ConstraintSign::GEQ, "x0 >= 0.0");
    builder.addConstraint(constraint1);

    // x1 >= 1
    coefficients(0) = 0;
    coefficients(1) = 1;
    rhs = 1.0;
    Constraint constraint2(coefficients, rhs, ConstraintSign::GEQ, "x1 >= 1.0");
    builder.addConstraint(constraint2);

    // x0 - x1 == 0
    coefficients(0) = 1;
    coefficients(1) = -1;
    rhs = 0.0;
    Constraint constraint3(coefficients, rhs, ConstraintSign::EQ, "x0 == x1");
    builder.addConstraint(constraint3);

    VectorXd expectedSolution(n);
    expectedSolution(0) = 1;
    expectedSolution(1) = 1;
    double expectedObjectiveValue = 8.5;

    Model model = builder.build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());
    BOOST_CHECK_EQUAL(SolutionStatus::fixed_on_constraints, solution.getStatus());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    
    BOOST_CHECK_EQUAL(n, solution.getActiveConstraints().size());
}

BOOST_AUTO_TEST_CASE(constrainedFunction_OneEqAndGeqBoundAndLeqBoundConstraint) {

    double tolerance = 0.001;
    SolverSettings settings;
    settings.enableSaveActiveConstraints();
    int n = 2;
    VectorXd x(n);
    x(0) = -10;
    x(1) = -8;

    ModelBuilder builder = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction)
        .setDerivativeEps(1e-6);

    VectorXd coefficients(n);
    double rhs;
    int index;

    // x0 >= -11.0
    coefficients(0) = 1;
    coefficients(1) = 0;
    rhs = -11.0;
    index = 0;
    Constraint constraint1(n, index, rhs, ConstraintSign::GEQ, "x0 >= -11.0");
    builder.addConstraint(constraint1);

    // x1 <= 0.5
    coefficients(0) = 0;
    coefficients(1) = 1;
    rhs = 0.5;
    index = 1;
    Constraint constraint2(n, index, rhs, ConstraintSign::LEQ, "x1 <= 0.5");
    builder.addConstraint(constraint2);

    // -x0 + x1 == 2
    coefficients(0) = -1;
    coefficients(1) = 1;
    rhs = 2.0;
    Constraint constraint3(coefficients, rhs, ConstraintSign::EQ, "-x0 + x1 == 2");
    builder.addConstraint(constraint3);

    VectorXd expectedSolution(n);
    expectedSolution(0) = -1.5;
    expectedSolution(1) = 0.5;
    double expectedObjectiveValue = 0.625;

    Model model = builder.build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());
    BOOST_CHECK_EQUAL(SolutionStatus::fixed_on_constraints, solution.getStatus());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    BOOST_CHECK_EQUAL(n, solution.getActiveConstraints().size());
}

BOOST_AUTO_TEST_CASE(constrainedFunction_3D) {

    double tolerance = 0.001;
    SolverSettings settings;
    // settings.enableDebugInfo();
    int n = 3;
    VectorXd x(n);
    x(0) = 0;
    x(1) = 0;
    x(2) = -20;

    ModelBuilder builder = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction3D)
        .setDerivativeEps(1e-6);

    VectorXd coefficients(n);
    double rhs;

    // x1 >= 0.0
    coefficients(0) = 0;
    coefficients(1) = 1;
    coefficients(2) = 0;
    rhs = 0.0;
    Constraint constraint1(coefficients, rhs, ConstraintSign::GEQ, "x1 >= 0.0");
    builder.addConstraint(constraint1);

    // x2 <= 0.0
    coefficients(0) = 0;
    coefficients(1) = 0;
    coefficients(2) = 1;
    rhs = 0.0;
    Constraint constraint2(coefficients, rhs, ConstraintSign::LEQ, "x2 <= 0.0");
    builder.addConstraint(constraint2);

    // x0 + x1 <= 0.0
    coefficients(0) = 1;
    coefficients(1) = 1;
    coefficients(2) = 0;
    rhs = 0.0;
    Constraint constraint3(coefficients, rhs, ConstraintSign::LEQ, "x0 + x1 <= 0.0");
    builder.addConstraint(constraint3);

    VectorXd expectedSolution(n);
    expectedSolution(0) = -1.0;
    expectedSolution(1) = 0.0;
    expectedSolution(2) = 0.0;
    double expectedObjectiveValue = 0.0;

    Model model = builder.build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    BOOST_CHECK_SMALL(solverSolution(2) - expectedSolution(2), tolerance);
}

BOOST_AUTO_TEST_CASE(constrainedFunction_3D_parallel) {

    double tolerance = 0.001;
    SolverSettings settings;
    // settings.enableDebugInfo();
    int n = 3;
    int numberOfThreads = 4;
    VectorXd x(n);
    x(0) = 0;
    x(1) = 0;
    x(2) = -20;

    ModelBuilder builder = Model::getBuilder()
        .setProblemSize(n)
        .setObjectiveFunction(constrainedFunction3D)
        .setDerivativeEps(1e-6)
        .setNumberOfThreads(numberOfThreads);

    VectorXd coefficients(n);
    double rhs;

    // x1 >= 0.0
    coefficients(0) = 0;
    coefficients(1) = 1;
    coefficients(2) = 0;
    rhs = 0.0;
    Constraint constraint1(coefficients, rhs, ConstraintSign::GEQ, "x1 >= 0.0");
    builder.addConstraint(constraint1);

    // x2 <= 0.0
    coefficients(0) = 0;
    coefficients(1) = 0;
    coefficients(2) = 1;
    rhs = 0.0;
    Constraint constraint2(coefficients, rhs, ConstraintSign::LEQ, "x2 <= 0.0");
    builder.addConstraint(constraint2);

    // x0 + x1 <= 0.0
    coefficients(0) = 1;
    coefficients(1) = 1;
    coefficients(2) = 0;
    rhs = 0.0;
    Constraint constraint3(coefficients, rhs, ConstraintSign::LEQ, "x0 + x1 <= 0.0");
    builder.addConstraint(constraint3);

    VectorXd expectedSolution(n);
    expectedSolution(0) = -1.0;
    expectedSolution(1) = 0.0;
    expectedSolution(2) = 0.0;
    double expectedObjectiveValue = 0.0;

    Model model = builder.build();

    Solver solver(&model, &settings);
    Solution solution = solver.solve(x);

    BOOST_CHECK_EQUAL(true, solution.isSolutionFound());

    VectorXd solverSolution = solution.getSolution();

    BOOST_CHECK_SMALL(solution.getObjectiveValue() - expectedObjectiveValue, tolerance);
    BOOST_CHECK_SMALL(solverSolution(0) - expectedSolution(0), tolerance);
    BOOST_CHECK_SMALL(solverSolution(1) - expectedSolution(1), tolerance);
    BOOST_CHECK_SMALL(solverSolution(2) - expectedSolution(2), tolerance);
}

BOOST_AUTO_TEST_SUITE_END() 