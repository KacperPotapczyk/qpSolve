#ifndef QPSOLVE_MODEL_BUILDER_H_
#define QPSOLVE_MODEL_BUILDER_H_

#include "TwiceDiffFunction.h"
#include "Model.h"
#include "Constraint.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace qpSolve {

    class ModelBuilder {
    public:
        ModelBuilder();
        ModelBuilder(int problemSize, ObjectiveFunctionPtr objectiveFunction);

        ModelBuilder& setProblemSize(const int &problemSize);
        ModelBuilder& setDerivativeEps(const double &derivativeEps);
        ModelBuilder& setObjectiveFunction(const ObjectiveFunctionPtr objectiveFunction);
        ModelBuilder& setGradientFunction(const GradientFunctionPtr gradientFunction);
        ModelBuilder& setHessianFunction(const HessianFunctionPtr hessianFunction);
        ModelBuilder& addConstraint(const Constraint& constraint);
        ModelBuilder& addConstraints(const std::vector<Constraint> &constraintVector);
        ModelBuilder& setNumberOfThreads(const int &numberOfThreads);

        Model build();
    
    private:
        int problemSize;
        int twiceDiffFunctionNumberOfThreads;
        double derivativeEps;
        ObjectiveFunctionPtr objectiveFunction;
        GradientFunctionPtr gradientFunction;
        HessianFunctionPtr hessianFunction;
        std::vector<Constraint> constraints;
    };
}


#endif /* QPSOLVE_MODEL_BUILDER_H_ */