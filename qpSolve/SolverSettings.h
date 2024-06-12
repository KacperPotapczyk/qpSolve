#ifndef QPSOLVE_SOLVERSETTINGS_H_
#define QPSOLVE_SOLVERSETTINGS_H_

namespace qpSolve {

    class SolverSettings {
    public:
        SolverSettings();
        SolverSettings(
            int maxIterations,
            double dxEps,
            double dfEps,
            double gradientEps,
            bool debugInfo,
            bool saveIntermediateSolutions,
            bool saveActiveConstraints
        );

        int getMaxIterations() const {return maxIterations;}
        void setMaxIterations(const int &maxIterations);

        double getDxEps() const {return dxEps;}
        void setDxEps(const double &dxEps);

        double getDfEps() const {return dfEps;}
        void setDfEps(const double &dfEps);

        double getGradientEps() const {return gradientEps;}
        void setGradientEps(const double &gradientEps);

        bool isDebugInfo() const {return debugInfo;}
        void enableDebugInfo() {debugInfo = true;}
        void disableDebugInfo() {debugInfo = false;}

        bool isRestrictedStep() const {return restrictedStep;}
        double getMaxStepLength() const {return maxStepLength;}
        void enableRestrictedStep(const double &maxStepLength);
        void disableRestrictedStep() {restrictedStep = false;}
        
        bool isSaveIntermediateSolutions() const {return saveIntermediateSolutions;}
        void enableSaveIntermediateSolutions() {saveIntermediateSolutions = true;}
        void disableSaveIntermediateSolutions() {saveIntermediateSolutions = false;}

        bool isHessianSymmetryCorrection() const {return hessianSymmetryCorrection;}
        double getHessianSymmetryCorrectionEps() const {return hessianSymmetryCorrectionEps;}
        void enableHessianSymmetryCorrection(const double &hessianSymmetryCorrectionEps);
        void disableHessianSymmetryCorrection() {hessianSymmetryCorrection = false;}

        bool isSaveActiveConstraints() const {return saveActiveConstraints;}
        void enableSaveActiveConstraints() {saveActiveConstraints = true;}
        void disableSaveActiveConstraints() {saveActiveConstraints = false;}

    private:
        int maxIterations;
        double dxEps;
        double dfEps;
        double gradientEps;
        bool debugInfo;
        bool restrictedStep;
        double maxStepLength;
        bool saveIntermediateSolutions;
        bool hessianSymmetryCorrection;
        double hessianSymmetryCorrectionEps;
        bool saveActiveConstraints;
    };
}


#endif /* QPSOLVE_SOLVERSETTINGS_H_ */