#include "Model.h"
#include "ModelBuilder.h"
#include "Constraint.h"

namespace qpSolve {

    Model::Model(TwiceDiffFunction& objectiveFunction) {
        this->problemSize = objectiveFunction.getSize();
        this->objectiveFunction = objectiveFunction;
    }

    ModelBuilder Model::getBuilder() {
        return ModelBuilder{};
    }

    void Model::addConstraint(const Constraint &constraint) {

        if (problemSize != constraint.getSize()) {
            throw std::invalid_argument("qpSolve::Model::addConstraint: problemSize != constraint.getSize()");
        }

        constraints.push_back(constraint);
    }

    void Model::setConstraints(const std::vector<Constraint> &constraintVector) {

        this->constraints.clear();
        this->constraints.reserve(constraintVector.size());

        for (const Constraint & constraint : constraintVector) {
            addConstraint(constraint);
        }
    }
    
    std::vector<Constraint> Model::getEqualityConstraints() const {

        std::vector<Constraint> result;

        for (const Constraint &constraint : constraints) {
            if (constraint.getSign() == ConstraintSign::EQ) {
                result.push_back(constraint);
            }
        }

        return result;
    }

    std::vector<Constraint> Model::getInequalityConstraints() const {
        
        std::vector<Constraint> result;

        for (const Constraint &constraint : constraints) {
            if (constraint.getSign() != ConstraintSign::EQ) {
                result.push_back(constraint);
            }
        }

        return result;
    }
}