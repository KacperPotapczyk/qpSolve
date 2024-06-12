#ifndef QPSOLVE_MODEL_H_
#define QPSOLVE_MODEL_H_

#include "TwiceDiffFunction.h"
#include "Constraint.h"

namespace qpSolve {
    class ModelBuilder;

    class Model {
    public:
        Model(TwiceDiffFunction& objectiveFunction);

        static ModelBuilder getBuilder();

        int getProblemSize() const {return problemSize;}
        TwiceDiffFunction& getObjectiveFunction() {return objectiveFunction;}
        bool isConstrained() const {return !constraints.empty();}
        void addConstraint(const Constraint &constraint);
        void setConstraints(const std::vector<Constraint> &constraintVector);
        const std::vector<Constraint> getConstraints() const {return constraints;}
        std::vector<Constraint> getEqualityConstraints() const;
        std::vector<Constraint> getInequalityConstraints() const;
        int getNumberOfConstraint() const {return constraints.size();}

    private:
        int problemSize;
        TwiceDiffFunction objectiveFunction;
        std::vector<Constraint> constraints;
    };
}


#endif /* QPSOLVE_MODEL_H_ */