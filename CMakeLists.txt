cmake_minimum_required (VERSION 3.0)
project (LMSolver)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(${EIGEN3_INCLUDE_DIR})


include_directories(prototype/)
link_directories(prototype/)
add_subdirectory(prototype/)

add_executable(${PROJECT_NAME} LMSolver.cpp)

target_link_libraries(${PROJECT_NAME} prototype Eigen3::Eigen)

configure_file(measurements.txt measurements.txt COPYONLY)