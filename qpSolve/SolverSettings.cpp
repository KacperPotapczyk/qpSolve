#include "SolverSettings.h"
#include <stdexcept>

namespace qpSolve {

    SolverSettings::SolverSettings() {
        maxIterations = 10;
        dxEps = 1e-6;
        dfEps = 1e-6;
        gradientEps = 1e-6;
        debugInfo = false;
        restrictedStep = false;
        saveIntermediateSolutions = false;
        hessianSymmetryCorrection = false;
        saveActiveConstraints = false;
    }
    
    SolverSettings::SolverSettings(
        int maxIterations, 
        double dxEps, 
        double dfEps, 
        double gradientEps,
        bool debugInfo,
        bool saveIntermediateSolutions,
        bool saveActiveConstraints
    ) {

        this->maxIterations = maxIterations;
        this->dxEps = dxEps;
        this->dfEps = dfEps;
        this->gradientEps = gradientEps;
        this->debugInfo = debugInfo;
        this->restrictedStep = false;
        this->saveIntermediateSolutions = saveIntermediateSolutions;
        this->hessianSymmetryCorrection = false;
        this->saveActiveConstraints = saveActiveConstraints;
    }

    void SolverSettings::setMaxIterations(const int &maxIterations) {
        if (maxIterations <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::setMaxIterations: maxIterations <= 0");
        }
        this->maxIterations = maxIterations;
    }

    void SolverSettings::setDxEps(const double &dxEps) {
        if (dxEps <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::setDxEps: dxEps <= 0");
        }
        this->dxEps = dxEps;
    }

    void SolverSettings::setDfEps(const double &dfEps) {
        if (dfEps <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::setDfEps: dfEps <= 0");
        }
        this->dfEps = dfEps;
    }

    void SolverSettings::setGradientEps(const double &gradientEps) {
        if (gradientEps <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::setGradientEps: gradientEps <= 0");
        }
        this->gradientEps = gradientEps;
    }

    void SolverSettings::enableRestrictedStep(const double &maxStepLength) {
        if (maxStepLength <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::enableRestrictedStep: maxStepLength <= 0");
        }
        this->restrictedStep = true;
        this->maxStepLength = maxStepLength;
    }

    void SolverSettings::enableHessianSymmetryCorrection(const double &hessianSymmetryCorrectionEps) {
        if (hessianSymmetryCorrectionEps <= 0) {
            throw std::invalid_argument("qpSolve::SolverSettings::enableHessianSymmetryCorrection: hessianSymmetryCorrectionEps <= 0");
        }
        this->hessianSymmetryCorrection = true;
        this->hessianSymmetryCorrectionEps = hessianSymmetryCorrectionEps;
    }
}
