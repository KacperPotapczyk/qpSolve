#include <boost/test/unit_test.hpp>
#include <SolverSettings.h>

BOOST_AUTO_TEST_SUITE(SolverSettings)

BOOST_AUTO_TEST_CASE(DefaultValues) {
    
    qpSolve::SolverSettings solverSettings;

    BOOST_CHECK_EQUAL(10, solverSettings.getMaxIterations());
    BOOST_CHECK_EQUAL(1e-6, solverSettings.getDxEps());
    BOOST_CHECK_EQUAL(1e-6, solverSettings.getDfEps());
    BOOST_CHECK_EQUAL(1e-6, solverSettings.getGradientEps());
    BOOST_CHECK_EQUAL(false, solverSettings.isDebugInfo());
    BOOST_CHECK_EQUAL(false, solverSettings.isRestrictedStep());
    BOOST_CHECK_EQUAL(false, solverSettings.isSaveIntermediateSolutions());
    BOOST_CHECK_EQUAL(false, solverSettings.isHessianSymmetryCorrection());
    BOOST_CHECK_EQUAL(false, solverSettings.isSaveActiveConstraints());
}

BOOST_AUTO_TEST_CASE(ConstructorValues) {
    
    int maxIterations = 48;
    double dxEps = 1e-5;
    double dfEps = 1e-4;
    double gradientEps = 5e-3;
    bool debugInfo = true;
    bool saveIntermediateSolutions = false;
    bool saveActiveConstraints = true;
    int numberOfThreads = 2;

    qpSolve::SolverSettings solverSettings(
        maxIterations,
        dxEps,
        dfEps,
        gradientEps,
        debugInfo,
        saveIntermediateSolutions,
        saveActiveConstraints
    );

    BOOST_CHECK_EQUAL(maxIterations, solverSettings.getMaxIterations());
    BOOST_CHECK_EQUAL(dxEps, solverSettings.getDxEps());
    BOOST_CHECK_EQUAL(dfEps, solverSettings.getDfEps());
    BOOST_CHECK_EQUAL(gradientEps, solverSettings.getGradientEps());
    BOOST_CHECK_EQUAL(debugInfo, solverSettings.isDebugInfo());
    BOOST_CHECK_EQUAL(false, solverSettings.isRestrictedStep());
    BOOST_CHECK_EQUAL(saveIntermediateSolutions, solverSettings.isSaveIntermediateSolutions());
    BOOST_CHECK_EQUAL(false, solverSettings.isHessianSymmetryCorrection());
    BOOST_CHECK_EQUAL(saveActiveConstraints, solverSettings.isSaveActiveConstraints());
}

BOOST_AUTO_TEST_CASE(SetValues) {
    
    qpSolve::SolverSettings solverSettings;
    int maxIterations = 48;
    double dxEps = 1e-5;
    double dfEps = 1e-4;
    double gradientEps = 5e-3;
    double symmetryEps = 7e-4;
    double maxStepLength = 2.1;
    int numberOfThreads = 2;

    solverSettings.setMaxIterations(maxIterations);
    solverSettings.setDxEps(dxEps);
    solverSettings.setDfEps(dfEps);
    solverSettings.setGradientEps(gradientEps);
    solverSettings.enableDebugInfo();
    solverSettings.enableRestrictedStep(maxStepLength);
    solverSettings.enableSaveIntermediateSolutions();
    solverSettings.enableHessianSymmetryCorrection(symmetryEps);
    solverSettings.enableSaveActiveConstraints();

    BOOST_CHECK_EQUAL(maxIterations, solverSettings.getMaxIterations());
    BOOST_CHECK_EQUAL(dxEps, solverSettings.getDxEps());
    BOOST_CHECK_EQUAL(dfEps, solverSettings.getDfEps());
    BOOST_CHECK_EQUAL(gradientEps, solverSettings.getGradientEps());
    BOOST_CHECK_EQUAL(true, solverSettings.isDebugInfo());
    BOOST_CHECK_EQUAL(true, solverSettings.isRestrictedStep());
    BOOST_CHECK_EQUAL(maxStepLength, solverSettings.getMaxStepLength());
    BOOST_CHECK_EQUAL(true, solverSettings.isSaveIntermediateSolutions());
    BOOST_CHECK_EQUAL(true, solverSettings.isHessianSymmetryCorrection());
    BOOST_CHECK_EQUAL(symmetryEps, solverSettings.getHessianSymmetryCorrectionEps());
    BOOST_CHECK_EQUAL(true, solverSettings.isSaveActiveConstraints());
}

BOOST_AUTO_TEST_SUITE_END()