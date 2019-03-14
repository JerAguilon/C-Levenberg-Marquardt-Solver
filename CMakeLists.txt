cmake_minimum_required (VERSION 3.0)
project (LMExample)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable (BasicExample BasicExample.cpp)
target_link_libraries (BasicExample Eigen3::Eigen)
