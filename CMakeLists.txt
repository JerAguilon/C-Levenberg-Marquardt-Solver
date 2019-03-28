cmake_minimum_required (VERSION 3.0)
project (LMSolver)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(${EIGEN3_INCLUDE_DIR})

file(GLOB src_files
    "LMSolver.cpp"
)
message(${src_files})

include_directories(prototype/)

add_executable(${PROJECT_NAME} ${src_files})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)