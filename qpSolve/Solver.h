#ifndef QPSOLVE_SOLVER_H_
#define QPSOLVE_SOLVER_H_

#include <Eigen/Dense>
#include "Model.h"
#include "SolverSettings.h"
#include "Solution.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace qpSolve {
    
    class Solver {
    public:
        Solver(Model *model, SolverSettings *solverSettings);
        Solution solve(const VectorXd &initialX);
    
    private:
        Model *model;
        SolverSettings *settings;

        Solution solveUnconstrained(const VectorXd &initialX);
        Solution solveConstrained(const VectorXd &initialX);

        bool initiateConstraints(
            const VectorXd &x,
            std::vector<Constraint> &constraints,
            MatrixXd &activeConstraints,
            VectorXd &activeConstraintsRhs
        );

        bool isSolutionFound(
            const int &iteration,
            const VectorXd &x,
            const VectorXd &dx,
            const double &f, 
            const double &fPrev, 
            const VectorXd &gradient,
            Solution &solution
        );

        void newtonStep(
            const VectorXd &x, 
            const double f,
            const VectorXd &gradient,
            const MatrixXd &hessian,
            VectorXd &dx
        );

        bool restrictedStepCorrection(VectorXd &dx);

        bool findConstraintsToChange(
            std::vector<Constraint> &constraints,
            const VectorXd &lagrangeMultipliers,
            int &newConstraintIndex,
            int &removeConstraintIndex,
            int &removedActiveConstraintIndex,
            const VectorXd &dx,
            const VectorXd &xPrev,
            double &alphaMin
        );

        void changeActiveConstraints(
            int &activeSetSize,
            const int &newConstraintIndex,
            const int &removedConstraintIndex,
            const int &removedActiveConstraintIndex,
            std::vector<Constraint> &constraints,
            MatrixXd &activeConstraints,
            VectorXd &activeConstraintsRhs
        );

        bool matrixSymmetryCorrection(MatrixXd &matrix);
    };
}

#endif /* QPSOLVE_SOLVER_H_ */