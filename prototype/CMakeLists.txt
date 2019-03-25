set(prototype_files
    MyGTSAMSolver.cpp
    MyGTSAMSolver.h
)

message(Prototype Files: ${prototype_files})

add_library(prototype ${prototype_files})