#ifndef QPSOLVE_CONSTRAINT_H_
#define QPSOLVE_CONSTRAINT_H_ 

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace qpSolve {

	enum ConstraintSign {
		EQ,
		GEQ,
		LEQ
	};

    class Constraint {
    public:
        Constraint(const VectorXd &coefficients, const double &rhs, const ConstraintSign sign);
        Constraint(const VectorXd &coefficients, const double &rhs, const ConstraintSign sign, const std::string name);
        Constraint(int &size, int &index, double &rhs, ConstraintSign sign);
        Constraint(int &size, int &index, double &rhs, ConstraintSign sign, std::string name);

        double residual(const VectorXd &x) const;
        double value(const VectorXd &x) const;
        void swapSign();

        ConstraintSign getSign() const {return sign;}
        int getSize() const {return size;}
        std::string getName() const {return name;}
        VectorXd getCoefficients() const {return coefficients;}
        double getRhs() const {return rhs;}

        bool isActive() const {return active;}
        void activate() {active = true;}
        void deactivate() {active = false;}

        void setLagrangeMultiplier(const double &value) {lagrangeMultiplier = value;}
        double getLagrangeMultiplier() {return lagrangeMultiplier;}

        bool isSignSwapped() const {return signSwapped;}
        bool isBoundConstarint() const {return boundConstraint;}

    private:
        int size;
        VectorXd coefficients;
        double rhs;
        ConstraintSign sign;
        std::string name;
        bool active;
        double lagrangeMultiplier;
        bool signSwapped;
        int index;
        bool boundConstraint;
    };
}

#endif /* QPSOLVE_CONSTRAINT\_H_ */