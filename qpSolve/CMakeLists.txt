add_library(qpSolve 
    TwiceDiffFunction.cpp
    Model.cpp
    ModelBuilder.cpp
    SolverSettings.cpp
    Solver.cpp
    Solution.cpp
    Constraint.cpp
)

target_include_directories(qpSolve INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

target_link_libraries(qpSolve Eigen3::Eigen compiler_flags)
