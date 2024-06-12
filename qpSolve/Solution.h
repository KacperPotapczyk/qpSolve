#ifndef QPSOLVE_SOLUTION_H_
#define QPSOLVE_SOLUTION_H_

#include <Eigen/Dense>
#include "Constraint.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace qpSolve {

    enum SolutionStatus {
        not_solved,
        error,
        max_iteration_reached,
        objective_improvement_leq_eps,
		solution_improvement_leq_eps,
		gradient_leq_eps,
		saddle_point,
        invalid_constraint,
        fixed_on_constraints
    };

    class Solution {
    public:
        Solution();
        Solution(Solution const &other);

        void setError(std::string errorMessage);
        void setInvalidConstraint();
        void setOnConstraint(const int &iteration, const double &objectiveValue, const VectorXd &solution);
        void setObjectiveImprovementLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution);
        void setSolutionImprovementLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution);
        void setGradientLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution);
        void setMaxIterationReached(const int &iteration, const double &objectiveValue, const VectorXd &solution);

        bool isSolutionFound() const {return solutionFound;}
        SolutionStatus getStatus() const {return status;}
        double getObjectiveValue() const {return objectiveValue;}
        VectorXd getSolution() const {return solution;}
        int getIteration() const {return iteration;}
        std::string getErrorMessage() const {return errorMessage;}

        void addIntermediateSolution(const VectorXd &x) {intermediateSolutions.push_back(x);}
        std::vector<VectorXd> getIntermediateSolutions() const {return intermediateSolutions;}

        void addActiveConstraint(const Constraint &constraint) {activeConstraints.push_back(constraint);}
        std::vector<Constraint> getActiveConstraints() const {return activeConstraints;}

    private:
        bool solutionFound;
        SolutionStatus status;
        double objectiveValue;
        VectorXd solution;
        int iteration;
        std::string errorMessage;

        std::vector<VectorXd> intermediateSolutions;
        std::vector<Constraint> activeConstraints;

        void setSolution(const int &iteration, const double &objectiveValue, const VectorXd &solution);
    };
}


#endif /* QPSOLVE_SOLUTION_H_ */