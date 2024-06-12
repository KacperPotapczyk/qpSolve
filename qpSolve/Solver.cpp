#include "Solver.h"
#include <iostream>
#include <boost/asio/thread_pool.hpp>

namespace qpSolve {
    
    Solver::Solver(Model *model, SolverSettings *solverSettings) {
        this->model = model;
        this->settings = solverSettings;
    }

	Solution Solver::solve(const VectorXd &initialX) {
		if (model->isConstrained()) {
			return solveConstrained(initialX);
		} else {
			return solveUnconstrained(initialX);
		}
	}

    Solution Solver::solveUnconstrained(const VectorXd &initialX) {

		Solution solution;
		int n = model->getProblemSize();

		if (n != initialX.size()) {
			solution.setError("model->getProblemSize() != initialX.size()");
            return solution;
        }

		VectorXd x(initialX);
	    VectorXd xPrev(initialX);
	    VectorXd dx(initialX);
	    double f, fPrev;
	    VectorXd gradient(n);
	    MatrixXd hessian(n, n);
	    bool solutionFound = false;
	    int iteration = 0;
	    double dxNorm;
		double maxStepLength;

        TwiceDiffFunction objective = model->getObjectiveFunction();
		objective.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);

		if (settings->isSaveIntermediateSolutions()) {
			solution.addIntermediateSolution(x);
		}

	    if (gradient.norm() <= settings->getGradientEps()) {
	    	solutionFound = true;
	    	solution.setGradientLeqEps(iteration, f, x);
	    }

	    while (!solutionFound && iteration <= settings->getMaxIterations()) {

			iteration++;
			xPrev = x;
			fPrev = f;

			if (settings->isHessianSymmetryCorrection()) {
				bool correctionRequired = matrixSymmetryCorrection(hessian);
				if (correctionRequired && settings->isDebugInfo()) {
					std::cout << "Hessian symmetry corrected" << std::endl;
				}
			}

			if (settings->isDebugInfo()) {
				std::cout << "Iteration: " << iteration << std::endl;
				std::cout << "x: " << x.transpose() << std::endl;
				std::cout << "f: " << f << std::endl;
				std::cout << "gradient: " << gradient.transpose() << std::endl;
				std::cout << "hessian: \n" << hessian << std::endl;
			}

            newtonStep(x, f, gradient, hessian, dx);

			x += dx;
			if (settings->isSaveIntermediateSolutions()) {
				solution.addIntermediateSolution(x);
			}

			objective.evaluateFunctionGradientAndHessian(x, f, gradient, hessian);
            
            solutionFound = isSolutionFound(
				iteration,
				x, dx,
				f, fPrev,
				gradient,
				solution
			);
        }
		
		if (settings->isDebugInfo()) {

			if (solutionFound) {
				std::cout << "Solution found" << std::endl;
				std::cout << "Iteration: " << iteration << std::endl;
				std::cout << "x: " << x.transpose() << std::endl;
				std::cout << "f: " << f << std::endl;
				std::cout << "gradient: " << gradient.transpose() << std::endl;
				std::cout << "hessian: \n" << hessian << std::endl;
			} else {
				std::cout << "Solution not found" << std::endl;
			}
		}

        return solution;
    }

    Solution Solver::solveConstrained(const VectorXd &initialX) {

		Solution solution;
		int n = model->getProblemSize();

		if (n != initialX.size()) {
			solution.setError("model->getProblemSize() != initialX.size()");
            return solution;
        }

		int iteration = 0;
	    bool solutionFound = false;
		bool baseSolutionFound = false;

		VectorXd x(initialX), xPrev(initialX), dx(initialX);
	    double f, fPrev;
	    VectorXd gradient(n), gradientPrim(n);
	    MatrixXd hessian(n, n);

		std::vector<Constraint> constraints;
		bool validConstraints = false;
		bool constraintsChanged = true;
		int activeSetSize = 0;
		int newConstraintIndex = -1;
		int removedConstraintIndex = -1;
		int removedActiveConstraintIndex = -1;
		double alphaMin;

		MatrixXd activeConstraints;
		VectorXd activeConstraintsRhs;
		VectorXd activeConstraintsRhsPrim;
		VectorXd lagrangeMultipliers, Q1gPrim;

		MatrixXd reducedHessian, R, Q, Q1, Z, P;
		VectorXd reducedGradient, Yb, u, y;

        TwiceDiffFunction objective = model->getObjectiveFunction();

		validConstraints = initiateConstraints(
			x,
			constraints,
			activeConstraints,
			activeConstraintsRhs
		);

		if (validConstraints) {

			activeSetSize = activeConstraints.cols();

			objective.evaluateFunctionGradientAndHessian(x, f, gradientPrim, hessian);
			gradient = gradientPrim - hessian * x;
			fPrev = f;

			if (gradientPrim.norm() <= settings->getGradientEps()) {
				solutionFound = true;
				solution.setGradientLeqEps(iteration, f, x);
			}
		} else {
			if (settings->isDebugInfo()) {
				std::cout << "Initial solution is invalid due to constraints" << std::endl;
			}
			solution.setInvalidConstraint();
		}

		if (settings->isSaveIntermediateSolutions()) {
			solution.addIntermediateSolution(x);
		}

		while (!solutionFound && iteration <= settings->getMaxIterations() && validConstraints) {
			
			iteration++;
			baseSolutionFound = false;

			if (settings->isHessianSymmetryCorrection()) {
				bool correctionRequired = matrixSymmetryCorrection(hessian);
				if (correctionRequired && settings->isDebugInfo()) {
					std::cout << "Hessian symmetry corrected" << std::endl;
				}
			}

			if (settings->isDebugInfo()) {
				std::cout << "Iteration: " << iteration << std::endl;
				std::cout << "x: " << x.transpose() << std::endl;
				std::cout << "f: " << f << std::endl;
				std::cout << "gradient: " << gradient.transpose() << std::endl;
				std::cout << "hessian: \n" << hessian << std::endl;
			}

			if (n < activeSetSize) {
				solution.setError("Model is overconstraned");
				break;

			} else if (activeSetSize == 0) {
				newtonStep(x, f, gradientPrim, hessian, dx);
				x += dx;

				if (settings->isDebugInfo()) {
					std::cout << "x: " << x.transpose() << std::endl;
					std::cout << "dx: " << dx.transpose() << std::endl;
				}

			} else {

				if (constraintsChanged) {
					// recalculate QR decomposition
					Eigen::ColPivHouseholderQR qr = activeConstraints.colPivHouseholderQr();
					R = qr.matrixR().topLeftCorner(activeSetSize, activeSetSize);
					Q = qr.matrixQ();
					Q1 = Q.leftCols(activeSetSize);
					Z = Q.rightCols(n - activeSetSize);
					P = qr.colsPermutation();
					activeConstraintsRhsPrim = P * activeConstraintsRhs;
					u = R.transpose().triangularView<Eigen::Lower>().solve(activeConstraintsRhsPrim);
					Yb = Q1*u;

					if (settings->isDebugInfo()) {
						std::cout << "R: \n" << R << std::endl;
						std::cout << "Q1: \n" << Q1 << std::endl;
						std::cout << "Z: \n" << Z << std::endl;
						std::cout << "P: \n" << P << std::endl;
						std::cout << "activeConstraintsRhsPrim: " << activeConstraintsRhsPrim.transpose() << std::endl;
						std::cout << "u: " << u.transpose() << std::endl;
						std::cout << "Yb: \n" << Yb << std::endl;
					}
				}

				if (activeSetSize == n) {
					// Linear solution due to constraints
					x = Yb;
					dx = x - xPrev;

					baseSolutionFound = true;
					solution.setOnConstraint(iteration, f, x);

					if (settings->isDebugInfo()) {
						std::cout << "Linear solution due to constraints" << std::endl;
						std::cout << "x: " << x.transpose() << std::endl;
						std::cout << "dx: " << dx.transpose() << std::endl;
					}

				} else {
					// Constrained solution
					reducedHessian = Z.transpose() * hessian * Z;
					reducedGradient = Z.transpose() * (gradient + hessian * Yb);

					if (settings->isHessianSymmetryCorrection()) {
						if (matrixSymmetryCorrection(reducedHessian) && settings->isDebugInfo()) {
							std::cout << "Reduced hessian symmetry corrected" << std::endl;
						}
					}

					y = reducedHessian.llt().solve(reducedGradient);

					if (settings->isDebugInfo()) {
						std::cout << "hessian: \n" << hessian << std::endl;
						std::cout << "gradient: " << gradient.transpose() << std::endl;
						std::cout << "reducedHessian: \n" << reducedHessian << std::endl;
						std::cout << "reducedGradient: " << reducedGradient.transpose() << std::endl;
						std::cout << "y: " << y.transpose() << std::endl;
					}

					x = Yb - Z*y;
					dx = x - xPrev;

					if (settings->isRestrictedStep()) {
						if (restrictedStepCorrection(dx)) {
							x = xPrev + dx;
						}
					}

					if (settings->isDebugInfo()) {
						std::cout << "x: " << x.transpose() << std::endl;
						std::cout << "dx: " << dx.transpose() << std::endl;
					}
				}
			}

			objective.evaluateFunctionGradientAndHessian(x, f, gradientPrim, hessian);
			gradient = gradientPrim - hessian * x;

			if (!baseSolutionFound) {
				baseSolutionFound = isSolutionFound(
					iteration,
					x, dx,
					f, fPrev,
					gradientPrim,
					solution
				);
			}

			// calculate lagrange multipliers
			if (activeSetSize > 0) {
				Q1gPrim = Q1.transpose() * gradientPrim;
				lagrangeMultipliers = P * R.triangularView<Eigen::Upper>().solve(Q1gPrim);
			}
			
			constraintsChanged = findConstraintsToChange(
				constraints,
				lagrangeMultipliers,
				newConstraintIndex,
				removedConstraintIndex,
				removedActiveConstraintIndex,
				dx, xPrev, alphaMin
			);

			// new solution due to new constraint being activated
			if (newConstraintIndex >= 0) {
				
				x = xPrev + alphaMin * dx;
				objective.evaluateFunctionGradientAndHessian(x, f, gradientPrim, hessian);
				gradient = gradientPrim - hessian * x;
			}

			changeActiveConstraints(
				activeSetSize,
				newConstraintIndex,
				removedConstraintIndex,
				removedActiveConstraintIndex,
				constraints,
				activeConstraints,
				activeConstraintsRhs
			);

			fPrev = f;
			xPrev = x;

			if (settings->isSaveIntermediateSolutions()) {
				solution.addIntermediateSolution(x);
			}

			if (!constraintsChanged && baseSolutionFound) {
				solutionFound = true;
			}

		}	// end while

		if (iteration > settings->getMaxIterations()) {
			solution.setMaxIterationReached(iteration, f, x);
		}

		if (settings->isSaveActiveConstraints()) {
			for (const Constraint &constraint : constraints) {
				if (constraint.isActive()) {
					solution.addActiveConstraint(constraint);
				}
			}
		}

        return solution;
    }

	bool Solver::initiateConstraints(
		const VectorXd &x,
		std::vector<Constraint> &constraints,
		MatrixXd &activeConstraints,
		VectorXd &activeConstraintsRhs
	) {

		int index = 0;
		double eps = settings->getDxEps();

		constraints.clear();
		constraints.reserve(model->getNumberOfConstraint());
		std::vector<int> activeConstraintsIndexes;
		activeConstraintsIndexes.reserve(model->getNumberOfConstraint());
		
		for (Constraint &equalityConstraint : model->getEqualityConstraints()) {

			if (abs(equalityConstraint.residual(x)) >= eps) {
				return false;
			}

			constraints.push_back(equalityConstraint);
			activeConstraintsIndexes.push_back(index);
			index++;
		}
	
		for (Constraint inequalityConstraint : model->getInequalityConstraints()) {

			if (inequalityConstraint.getSign() == ConstraintSign::LEQ) {
				inequalityConstraint.swapSign();
			}

			double residual = inequalityConstraint.residual(x);
			if (residual < -1.0*eps) {
				return false;
			}

			if (abs(residual) <= eps) {
				inequalityConstraint.activate();
				activeConstraintsIndexes.push_back(index);
			}

			constraints.push_back(inequalityConstraint);
			index++;
		}
		
		int numberOfActiveConstraints = activeConstraintsIndexes.size();
		activeConstraints.resize(x.size(), numberOfActiveConstraints);
		activeConstraintsRhs.resize(numberOfActiveConstraints);

		for (int i=0; i<numberOfActiveConstraints; i++) {
			activeConstraints.col(i) = constraints.at(activeConstraintsIndexes.at(i)).getCoefficients();
			activeConstraintsRhs(i) = constraints.at(activeConstraintsIndexes.at(i)).getRhs();
		}

		return true;
	}

	bool Solver::findConstraintsToChange(
		std::vector<Constraint> &constraints,
		const VectorXd &lagrangeMultipliers,
		int &newConstraintIndex,
		int &removedConstraintIndex,
		int &removedActiveConstraintIndex,
		const VectorXd &dx,
		const VectorXd &xPrev,
        double &alphaMin
	) {

		double lambdaMin = -1 * settings->getDxEps();
		double alpha;
		alphaMin = 1.0;
		newConstraintIndex = -1;
		removedConstraintIndex = -1;

		int activeConstraintsIndex = 0;
		int index = 0;
		for (Constraint &constraint : constraints) {

			if (constraint.isActive()) {
				constraint.setLagrangeMultiplier(lagrangeMultipliers(activeConstraintsIndex));

				if (constraint.getSign() != ConstraintSign::EQ) {

					if (lagrangeMultipliers(activeConstraintsIndex) < lambdaMin) {
						lambdaMin = lagrangeMultipliers(activeConstraintsIndex);
						removedConstraintIndex = index;
						removedActiveConstraintIndex = activeConstraintsIndex;
					}
				}
				activeConstraintsIndex++;
				index++;
			}
			else {
				double aTs = constraint.value(dx);

				if (abs(aTs) > settings->getDxEps()) {
					alpha = (constraint.getRhs() - constraint.value(xPrev)) / aTs;
				} else {
					alpha = 1;
				}

				if (alpha < alphaMin && aTs < 0) {
					alphaMin = alpha;
					newConstraintIndex = index;
				}
				index++;
			}
		}
		return removedConstraintIndex >= 0 || newConstraintIndex >= 0;
	}

	void Solver::changeActiveConstraints(
		int &activeSetSize,
		const int &newConstraintIndex,
		const int &removedConstraintIndex,
		const int &removedActiveConstraintIndex,
		std::vector<Constraint> &constraints,
		MatrixXd &activeConstraints,
		VectorXd &activeConstraintsRhs
	) {

		if (newConstraintIndex >= 0) {
			constraints.at(newConstraintIndex).activate();

			if (removedConstraintIndex >= 0) {
				constraints.at(removedConstraintIndex).deactivate();

				// substitute old constraint
				activeConstraints.col(removedConstraintIndex) = constraints.at(newConstraintIndex).getCoefficients();
				activeConstraintsRhs(removedConstraintIndex) = constraints.at(newConstraintIndex).getRhs();
			} else {
				// only add new
				activeConstraints.conservativeResize(Eigen::NoChange, activeSetSize+1);
				activeConstraints.rightCols(1) = constraints.at(newConstraintIndex).getCoefficients();

				activeConstraintsRhs.conservativeResize(activeSetSize+1);
				activeConstraintsRhs(activeSetSize) = constraints.at(newConstraintIndex).getRhs();
			}
		} else if (removedConstraintIndex >= 0) {
			constraints.at(removedConstraintIndex).deactivate();

			// resize activeConstraint matirx
			activeConstraints.conservativeResize(Eigen::NoChange, activeSetSize-1);
			activeConstraintsRhs.conservativeResize(activeSetSize-1);

			// rewrite all inequality constraints
			int activeConstraintsIndex = 0;
			for (const Constraint &constraint : constraints) {
				if (constraint.isActive()) {
					if (constraint.getSign() != ConstraintSign::EQ && activeConstraintsIndex >= removedActiveConstraintIndex) {
						activeConstraints.col(activeConstraintsIndex) = constraint.getCoefficients();
						activeConstraintsRhs(activeConstraintsIndex) = constraint.getRhs();
					}
					activeConstraintsIndex++;
				}
			}
		}

		activeSetSize = activeConstraints.cols();

		if (settings->isDebugInfo()) {
			std::cout << "activeConstraints: \n" << activeConstraints << std::endl;
			std::cout << "activeConstraintsRhs: \n" << activeConstraintsRhs.transpose() << std::endl;
		}
	}

    bool Solver::isSolutionFound(
		const int &iteration,
		const VectorXd &x,
		const VectorXd &dx,
		const double &f, 
		const double &fPrev, 
		const VectorXd &gradient,
		Solution &solution
	) {

        if (dx.norm() <= settings->getDxEps()) {
            solution.setSolutionImprovementLeqEps(iteration, f, x);
            return true;
        } else if (fabs(fPrev - f) <= settings->getDfEps()) {
            solution.setObjectiveImprovementLeqEps(iteration, f, x);
            return true;
        } else if (gradient.norm() <= settings->getGradientEps()) {
            solution.setGradientLeqEps(iteration, f, x);
			return true;
        }
		return false;
    }

    void Solver::newtonStep(
		const VectorXd &x, 
		const double f,
		const VectorXd &gradient,
		const MatrixXd &hessian,
		VectorXd &dx
	) {

        dx = hessian.llt().solve(-1.0 * gradient);

		if (settings->isRestrictedStep()) {
			restrictedStepCorrection(dx);
		}

        if (settings->isDebugInfo()) {
            std::cout << "dx: " << dx.transpose() << std::endl;
        }
    }

	bool Solver::restrictedStepCorrection(VectorXd &dx) {

		double maxStepLength = settings->getMaxStepLength();
		double dxNorm = dx.norm();
		if (dxNorm > maxStepLength) {
			if (settings->isDebugInfo()) {
				std::cout << "Step is too long: " << dxNorm << std::endl;
				std::cout << "Restricting to: " << maxStepLength << std::endl;
			}
			dx *= maxStepLength / dxNorm;
			return true;
		}
		return false;
	}

	bool Solver::matrixSymmetryCorrection(MatrixXd &matrix) {

		bool correctionRequired = false;
		double newValue;
		double eps = settings->getHessianSymmetryCorrectionEps();

		for (int i=0; i<matrix.rows()-1; i++) {
			for(int j=i+1; j<matrix.cols(); j++) {
				if (abs(matrix(i, j) - matrix(j, i)) > eps) {
					correctionRequired = true;
					newValue = (matrix(i, j) + matrix(j, i)) / 2.0;

					matrix(i, j) = newValue;
					matrix(j, i) = newValue;
				}
			}
		}

        return correctionRequired;
    }
}