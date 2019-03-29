# Lightweight Levenberg Marquardt

This is a malloc-free Levenberg-Marquardt optimizer for nonlinear least squares regression.
This means that heap objects are never allocated during the optimization phase.
It comes with a demonstration fitting an arbitrary nonlinear function. This is a subproject
for [GTSAM](https://bitbucket.org/gtborg/gtsam/), a smoothing and mapping library as a 
part of my undergraduate research.

# How does it work?

As a consumer of the optimizer engine, you simply need to implement the `DataManipulator` class.
The class is used to fill out a jacobian and a residual matrix belonging to the optimizer.
The optimizer does not care what data it's fitting, just that the manipulator fills the
aforementioned matrices.

# Dependencies

* [Eigen 3.3](http://eigen.tuxfamily.org/index.php?title=3.3) 

# Running the demo

From the root of the project:

1. `mkdir build/`
2. `cd build`
3. `cmake ..`
4. `make && ./LMSolver`
