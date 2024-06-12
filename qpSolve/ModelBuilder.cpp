#include "ModelBuilder.h"

namespace qpSolve {

    ModelBuilder::ModelBuilder() {
        this->problemSize = 0;
        this->derivativeEps = 1e-3;
        this->objectiveFunction = NULL;
        this->gradientFunction = NULL;
        this->hessianFunction = NULL;
        this->twiceDiffFunctionNumberOfThreads = 1;
    }

    ModelBuilder::ModelBuilder(int problemSize, ObjectiveFunctionPtr objectiveFunction) {
        this->problemSize = problemSize;
        this->derivativeEps = 1e-3;
        this->objectiveFunction = objectiveFunction;
        this->gradientFunction = NULL;
        this->hessianFunction = NULL;
        this->twiceDiffFunctionNumberOfThreads = 1;
    }

    ModelBuilder& ModelBuilder::setProblemSize(const int &problemSize) {
        this->problemSize = problemSize;
        return *this;
    }

    ModelBuilder& ModelBuilder::setDerivativeEps(const double &derivativeEps) {
        this->derivativeEps = derivativeEps;
        return *this;
    }

    ModelBuilder& ModelBuilder::setObjectiveFunction(const ObjectiveFunctionPtr objectiveFunction) {
        this->objectiveFunction = objectiveFunction;
        return *this;
    }

    ModelBuilder& ModelBuilder::setGradientFunction(const GradientFunctionPtr gradientFunction) {
        this->gradientFunction = gradientFunction;
        return *this;
    }

    ModelBuilder& ModelBuilder::setHessianFunction(const HessianFunctionPtr hessianFunction) {
        this->hessianFunction = hessianFunction;
        return *this;
    }

    ModelBuilder &ModelBuilder::addConstraint(const Constraint &constraint) {
        this->constraints.push_back(constraint);
        return *this;
    }

    ModelBuilder &ModelBuilder::addConstraints(const std::vector<Constraint> &constraintVector) {
        for (const Constraint & constraint : constraintVector) {
            this->constraints.push_back(constraint);
        }
        return *this;
    }

    ModelBuilder &ModelBuilder::setNumberOfThreads(const int &numberOfThreads) {
        this->twiceDiffFunctionNumberOfThreads = numberOfThreads;
        return *this;
    }

    Model ModelBuilder::build() {

        if (problemSize <= 0) {
            throw std::invalid_argument("qpSolve::ModelBuilder::build: problemSize <= 0");
        }
        if (objectiveFunction == NULL) {
            throw std::invalid_argument("qpSolve::ModelBuilder::build: objectiveFunction == NULL");
        }
        if (derivativeEps <= 0) {
            throw std::invalid_argument("qpSolve::ModelBuilder::build: derivativeEps <= 0");
        }

        TwiceDiffFunction twiceDiffFunction(
            problemSize,
            objectiveFunction,
            gradientFunction,
            hessianFunction,
            derivativeEps,
            twiceDiffFunctionNumberOfThreads
        );

        Model model(twiceDiffFunction);
        model.setConstraints(constraints);
        return model;
    }
}