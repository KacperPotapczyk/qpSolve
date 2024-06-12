#include "Constraint.h"

namespace qpSolve {
    
    Constraint::Constraint(const VectorXd &coefficients, const double &rhs, const ConstraintSign sign) : Constraint(coefficients, rhs, sign, "") {}

    Constraint::Constraint(const VectorXd &coefficients, const double &rhs, const ConstraintSign sign, const std::string name) {
        
        this->size = coefficients.size();
        this->coefficients = coefficients;
        this->rhs = rhs;
        this->sign = sign;
        this->name = name;
        if (sign == ConstraintSign::EQ) {
            this->active = true;
        } else {
            this->active = false;
        }
        this->lagrangeMultiplier = 0.0;
        this->signSwapped = false;
        this->index = -1;
        this->boundConstraint = false;
    }

    Constraint::Constraint(int &size, int &index, double &rhs, ConstraintSign sign) : Constraint(size, index, rhs, sign, "") {}

    Constraint::Constraint(int &size, int &index, double &rhs, ConstraintSign sign, std::string name) {

        if (size <= 0) {
            throw std::invalid_argument("qpSolve::Constraint::Constraint: size <= 0");
        }
        if (index >= size) {
            throw std::invalid_argument("qpSolve::Constraint::Constraint: index >= size");
        }
        if (sign == ConstraintSign::EQ) {
            throw std::invalid_argument("qpSolve::Constraint::Constraint: bound constraint must be GEQ or LEQ");
        }

        this->size = size;
        this->coefficients.resize(size);
        for (int i=0; i<size; i++) {
            if (i == index) {
                coefficients(i) = 1.0;
            } else {
                coefficients(i) = 0.0;
            }
        }
        this->rhs = rhs;
        this->sign = sign;
        this->name = name;
        this->active = false;
        this->lagrangeMultiplier = 0.0;
        this->signSwapped = false;
        this->index = index;
        this->boundConstraint = true;
    }
    
    double Constraint::residual(const VectorXd &x) const {
        return boundConstraint ? coefficients(index) * x(index) - rhs : coefficients.dot(x)-rhs;
    }

    double Constraint::value(const VectorXd &x) const {

        return boundConstraint ? coefficients(index) * x(index) : coefficients.dot(x);
    }

    void Constraint::swapSign() {
        if (sign != ConstraintSign::EQ) {
            signSwapped = !signSwapped;
            coefficients *= -1.0;
            rhs *= -1.0;
            if (sign == ConstraintSign::LEQ) {
                sign = ConstraintSign::GEQ;
            } else {
                sign = ConstraintSign::LEQ;
            }
        }
    }
}