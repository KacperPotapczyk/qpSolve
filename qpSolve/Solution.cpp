#include "Solution.h"

namespace qpSolve {

    Solution::Solution() {
        this->solutionFound = false;
        this->status = SolutionStatus::not_solved;
        this->iteration = 0;
        this->objectiveValue = 0.0;
        this->solution = VectorXd();
        this->errorMessage = "";
    }

    Solution::Solution(Solution const &other) {
        this->solutionFound = other.solutionFound;
        this->status = other.status;
        this->iteration = other.iteration;
        this->objectiveValue = other.objectiveValue;
        this->solution = other.solution;
        this->errorMessage = other.errorMessage;
    }

    void Solution::setError(std::string errorMessage) {
        this->status = SolutionStatus::error;
        this->errorMessage = errorMessage;
    }

    void Solution::setInvalidConstraint() {
        this->status = SolutionStatus::invalid_constraint;
    }

    void Solution::setOnConstraint(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->solutionFound = true;
        this->status = SolutionStatus::fixed_on_constraints;
        setSolution(iteration, objectiveValue, solution);
    }

    void Solution::setObjectiveImprovementLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->solutionFound = true;
        this->status = SolutionStatus::objective_improvement_leq_eps;
        setSolution(iteration, objectiveValue, solution);
    }

    void Solution::setSolutionImprovementLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->solutionFound = true;
        this->status = SolutionStatus::solution_improvement_leq_eps;
        setSolution(iteration, objectiveValue, solution);
    }

    void Solution::setGradientLeqEps(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->solutionFound = true;
        this->status = SolutionStatus::gradient_leq_eps;
        setSolution(iteration, objectiveValue, solution);
    }

    void Solution::setMaxIterationReached(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->solutionFound = false;
        this->status = SolutionStatus::max_iteration_reached;
        setSolution(iteration, objectiveValue, solution);
    }

    void Solution::setSolution(const int &iteration, const double &objectiveValue, const VectorXd &solution) {
        this->iteration = iteration;
        this->objectiveValue = objectiveValue;
        this->solution = solution;
    }
}