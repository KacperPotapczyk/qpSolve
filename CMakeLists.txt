cmake_minimum_required(VERSION 3.0.0)
project(qpSolve VERSION 0.1.0 LANGUAGES C CXX)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_11)

add_subdirectory(qpSolve)

option(UNIT_TESTS "Enable unit tests." ON)

if(UNIT_TESTS)

    add_executable(qpSolve_test
        tests/TestFunctions.cpp
        tests/TestMain.cpp
        tests/ContextLoads.cpp
        tests/TwiceDiffFunctionTest.cpp
        tests/ModelTest.cpp
        tests/SolverSettingsTest.cpp
        tests/SolverTest.cpp
        tests/ConstraintTest.cpp
    )

    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    find_package(Boost COMPONENTS unit_test_framework REQUIRED)

    file(GLOB_RECURSE INCLUDE_FILES qpSolve/*.h)
    target_sources(qpSolve_test
        PRIVATE
        ${INCLUDE_FILES}
    )

    target_link_libraries(qpSolve_test
        compiler_flags
        Eigen3::Eigen 
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        qpSolve
    )

    enable_testing()
    add_test(NAME ContextLoads COMMAND qpSolve_test --run_test=ContextLoads)
    add_test(NAME TwiceDiffFunction COMMAND qpSolve_test --run_test=TwiceDiffFunction)
    add_test(NAME Model COMMAND qpSolve_test --run_test=Model)
    add_test(NAME SolverSettings COMMAND qpSolve_test --run_test=SolverSettings)
    add_test(NAME QP COMMAND qpSolve_test --run_test=QP)
    add_test(NAME Constraints COMMAND qpSolve_test --run_test=Constraints)
    
endif(UNIT_TESTS)
