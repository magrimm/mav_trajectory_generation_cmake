/*
 * Copyright (c) 2016, Markus Achtelik, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Helen Oleynikova, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Rik Bähnemann, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Marija Popovic, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MAV_TRAJECTORY_GENERATION_POLYNOMIAL_OPTIMIZATION_NONLINEAR_H_
#define MAV_TRAJECTORY_GENERATION_POLYNOMIAL_OPTIMIZATION_NONLINEAR_H_

#include <memory>
#include <nlopt.hpp>
#include <sdf_tools/sdf.hpp>

#include "mav_trajectory_generation/polynomial_optimization_linear.h"

namespace mav_trajectory_generation {

// Implements parts of the continuous-time trajectory optimization described
// in [2]
// [2]: Continuous-Time Trajectory Optimization for Online UAV Replanning.
//      Helen Oleynikova, Michael Burri, Zachary Taylor, Juan Nieto, Roland
// Siegwart and Enric Galceran. IROS 2016

// Class holding all important parameters for nonlinear optimization.
struct NonlinearOptimizationParameters {
  NonlinearOptimizationParameters()
      // Default parameters should be reasonable enough to use without further
      // fine-tuning.
      : f_abs(-1),
        f_rel(0.05),
        x_rel(-1),
        x_abs(-1),
        initial_stepsize_rel(0.1),
        equality_constraint_tolerance(1.0e-3),
        inequality_constraint_tolerance(0.1),
        max_iterations(3000),
        time_penalty(500.0),
        algorithm(nlopt::LN_SBPLX),
        random_seed(0),
        use_soft_constraints(true),
        soft_constraint_weight(100.0),
        print_debug_info(false),
        objective(kOptimizeFreeConstraintsAndTime),
        weights(),
        state_bound_multiplicator(2.0),
        map_resolution(0.0),
        min_bound(Eigen::Vector3d::Zero()),
        max_bound(Eigen::Vector3d::Zero()),
        use_numeric_grad(false),
        set_bounds_with_constraints(false) {}

  // Stopping criteria, if objective function changes less than absolute value.
  // Disabled if negative.
  double f_abs;

  // Stopping criteria, if objective function changes less than relative value.
  // Disabled if negative.
  double f_rel;

  // Stopping criteria, if state changes less than relative value. 
  // Disabled if negative.
  double x_rel;

  // Stopping criteria, if state changes less than absolute value.
  // Disabled if negative.
  double x_abs;

  // Determines a fraction of the initial guess as initial step size.
  // Heuristic value if negative.
  double initial_stepsize_rel;

  // Absolute tolerance, within an equality constraint is considered as met.
  double equality_constraint_tolerance;

  // Absolute tolerance, within an inequality constraint is considered as met.
  double inequality_constraint_tolerance;

  // Maximum number of iterations. Disabled if negative.
  int max_iterations;

  // Penalty for the segment time.
  double time_penalty;

  // Optimization algorithm used by nlopt, see
  // http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
  nlopt::algorithm algorithm;

  // Random seed, if an optimization algorithm involving random numbers
  // is used (e.g. nlopt::GN_ISRES).
  // If set to a value < 0, a "random" (getTimeofday) value for the seed
  // is chosen.
  int random_seed;

  // Decide whether to use soft constraints.
  bool use_soft_constraints;

  // Weights the relative violation of a soft constraint.
  double soft_constraint_weight;

  bool print_debug_info;

  // Specifies which optimization should be run.
  // kOptimizeTime: Run optimization over segment times only. Only the segment
  // times are optimization parameters, and the remaining free parameters are
  // found by solving the linear optimization problem with the given segment
  // times in every iteration.
  // kOptimizeFreeConstraintsAndTime: Both segment times and free
  // derivatives become optimization variables. This case is
  // theoretically correct, but may result in more iterations.
  // kOptimizeFreeConstraintsAndCollision: The free derivatives are optimized
  // with cost for the derivatives and cost for the colliison potential
  enum OptimizationObjective {
    kOptimizeFreeConstraints,
    kOptimizeFreeConstraintsAndTime,
    kOptimizeTime,
    kOptimizeFreeConstraintsAndCollision,
    kOptimizeFreeConstraintsAndCollisionAndTime,
    kUnknown
  } objective;

  // Struct storing all the weights for the cost and gradients
  struct cost_weights {
    // Default constructor
    cost_weights() : w_d(0.1), w_c(10.0), w_t(1.0), w_sc(1.0) {}

    double w_d;  // Weight for derivative cost
    double w_c;  // Weight for collision cost
    double w_t;  // Weight for time cost
    double w_sc; // Weight for soft constraint cost
  } weights;

  // Multiplication factor for optimization parameters of the initial guess.
  // Used for lower and upper bound. Optimization parameter boundaries
  // regarding positions of the vertices are set directly from the map
  // boundaries.
  double state_bound_multiplicator;

  // Map resolution of the environment
  double map_resolution;

  // Upper and Lower boundaries of the map/environment
  Eigen::Vector3d min_bound;
  Eigen::Vector3d max_bound;

  // Use numerical gradients
  bool use_numeric_grad;

  // Set the bounds of the optimization parameters with the constraints given
  // (map-->position, v_max, a_max)
  bool set_bounds_with_constraints;
};

class OptimizationInfo {
 public:
  OptimizationInfo()
      : n_iterations(0),
        stopping_reason(nlopt::FAILURE),
        cost_trajectory(0),
        cost_collision(0),
        cost_time(0),
        cost_soft_constraints(0),
        optimization_time(0) {}
  void print(std::ostream& stream) const;
  int n_iterations;
  int stopping_reason;
  double cost_trajectory;
  double cost_collision;
  double cost_time;
  double cost_soft_constraints;
  double optimization_time;
  std::map<int, Extremum> maxima;
};

// Implements a nonlinear optimization of the unconstrained optimization
// of paths consisting of polynomial segments as described in [1]
// [1]: Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense
// Indoor Environments.
// Charles Richter, Adam Bry, and Nicholas Roy. In ISRR 2013
// _N specifies the number of coefficients for the underlying polynomials.
template <int _N = 10>
class PolynomialOptimizationNonLinear {
  static_assert(_N % 2 == 0, "The number of coefficients has to be even.");

 public:
  enum { N = _N };

  // Sets up the nonlinear optimization problem.
  // Input: dimension = Spatial dimension of the problem. Usually 1 or 3.
  // Input: parameters = Parameters for the optimization problem.
  PolynomialOptimizationNonLinear(
      size_t dimension, const NonlinearOptimizationParameters& parameters);

  // Sets up the optimization problem from a vector of Vertex objects and
  // a vector of times between the vertices.
  // Input: vertices = Vector containing the vertices defining the support
  // points and constraints of the path.
  // Input: segment_times = Vector containing an initial guess of the time
  // between two vertices. Thus, its size is size(vertices) - 1.
  // Input: derivative_to_optimize = Specifies the derivative of which the
  // cost is optimized.
  bool setupFromVertices(
      const Vertex::Vector& vertices, const std::vector<double>& segment_times,
      int derivative_to_optimize =
          PolynomialOptimization<N>::kHighestDerivativeToOptimize);

  // Adds a constraint for the maximum of magnitude to the optimization
  // problem.
  // Input: derivative_order = Order of the derivative, for which the
  // constraint should be checked. Usually velocity (=1) or acceleration (=2).
  // maximum_value = Maximum magnitude of the specified derivative.
  bool addMaximumMagnitudeConstraint(int derivative_order,
                                     double maximum_value);

  // Solves the linear optimization problem according to [1].
  // The solver is re-used for every dimension, which means:
  //  - segment times are equal for each dimension.
  //  - each dimension has the same type/set of constraints. Their values can of
  // course differ.
  bool solveLinear();

  // Runs the optimization until one of the stopping criteria in
  // NonlinearOptimizationParameters and the constraints are met.
  int optimize();

  // Get the resulting trajectory out -- prefer this as the main method
  // to get the results of the optimization, over getting the reference
  // to the linear optimizer.
  void getTrajectory(Trajectory* trajectory) const {
    poly_opt_.getTrajectory(trajectory);
  }

  // Get the trajectory of the initial solution given to the nonlinear solver
  void getInitialSolutionTrajectory(Trajectory* trajectory) const {
    CHECK_NOTNULL(trajectory);
    Segment::Vector segments;
    trajectory_initial_.getSegments(&segments);
    trajectory->setSegments(segments);
  }

  // Returns a const reference to the underlying linear optimization
  // object.
  const PolynomialOptimization<N>& getPolynomialOptimizationRef() const {
    return poly_opt_;
  }

  // Returns a non-const reference to the underlying linear optimization
  // object.
  PolynomialOptimization<N>& getPolynomialOptimizationRef() {
    return poly_opt_;
  }

  OptimizationInfo getOptimizationInfo() const { return optimization_info_; }

  // Set the signed distance field needed for collision potential optimization.
  void setSDF(const std::shared_ptr<sdf_tools::SignedDistanceField>& sdf) {
    sdf_ = sdf;
  };

  // Compute the initial solution for the optimization without position
  // constraints apart from the start and goal vertices.
  // 1) Get the linear solution with position constraints at all vertices
  // 2) Get the polynomial coefficients from this solution for each segment
  // 3) Remove the position constraints from the intermediate vertices (still
  // fully constrained start and goal)
  // 4) Re-setup the problem with the new constraints. The position
  // constraints are now part of d_p the free constraints which are to be
  // optimized. (Fixed derivatives d_F gets smaller and d_P gets bigger)
  // 5) Get your new mapping matrix L (p = L*[d_f d_P]^T = A^(-1)*M*[d_f d_P]^T)
  // 6) Calculate your reordered endpoint-derivatives. d_all = L^(-1) * p_k
  // where p_k are the old coefficients from the original linear solution and
  // L the new remapping matrix
  // 7) Set the new free endpoint-derivatives d_p back in the linear solver.
  bool computeInitialSolutionWithoutPositionConstraints();

 private:
  // Holds the data for constraint evaluation, since these methods are
  // static.
  struct ConstraintData {
    PolynomialOptimizationNonLinear<N>* this_object;
    int derivative;
    double value;
  };

  // Objective function for the time-only version.
  // Input: segment_times = Segment times in the current iteration.
  // Input: gradient = Gradient of the objective function w.r.t. changes of
  // parameters.
  // We CANNOT compute the gradient analytically here.
  // --> Thus, only gradient-free optimization methods are possible.
  // Input: Custom data pointer = In our case, it's an ConstraintData object.
  // Output: Cost = based on the parameters passed in.
  static double objectiveFunctionTime(const std::vector<double>& segment_times,
                                      std::vector<double>& gradient,
                                      void* data);

  // Objective function for the version optimizing segment times and free
  // derivatives.
  // Input: optimization_variables = Optimization variables times in the
  // current iteration.
  // The variables (time, derivatives) are stacked as follows: [segment_times
  // derivatives_dim_0 ... derivatives_dim_N]
  // Input: gradient = Gradient of the objective function wrt. changes of
  // parameters.
  // We CANNOT compute the gradient analytically here.
  // --> Thus, only gradient free optimization methods are possible.
  // Input: data = Custom data pointer. In our case, it's an ConstraintData
  // object.
  // Output: Cost based on the parameters passed in.
  static double objectiveFunctionTimeAndConstraints(
      const std::vector<double>& optimization_variables,
      std::vector<double>& gradient, void* data);

  // Objective function for optimizing only the free endpoint-derivatives.
  // Input: optimization_variables = Optimization variables in the current
  // iteration.
  // The variables (derivatives) are stacked as follows: [derivatives_dim_0
  // ...  derivatives_dim_N]
  // Input: gradient = Gradient of the objective function wrt. changes of
  // parameters.
  // We CAN compute the gradient analytically here.
  // --> Thus, gradient-free and gradient-based optimization methods are
  // possible.
  // Input: data = Custom data pointer. In our case, it's an ConstraintData
  // object.
  // Output: Cost and gradients (only for gradient-based optimization) based
  // on the parameters passed in.
  static double objectiveFunctionFreeConstraints(
          const std::vector<double>& x, std::vector<double>& gradient,
          void* data);

  // Objective function for optimizing the free endpoint-derivatives and the
  // collision potential.
  // Input: optimization_variables = Optimization variables in the current
  // iteration.
  // The variables (derivatives) are stacked as follows: [derivatives_dim_0
  // ...  derivatives_dim_N]
  // Input: gradient = Gradient of the objective function wrt. changes of
  // parameters.
  // We CAN compute the gradient analytically here.
  // --> Thus, gradient-free and gradient-based optimization methods are
  // possible.
  // Input: data = Custom data pointer. In our case, it's an ConstraintData
  // object.
  // Output: Cost and gradients (only for gradient-based optimization) based
  // on the parameters passed in.
  static double objectiveFunctionFreeConstraintsAndCollision(
          const std::vector<double>& x, std::vector<double>& gradient,
          void* data);

  // Calculate the cost and gradients of the squared difference of the
  // derivative to be optimized.
  static double getCostAndGradientDerivative(
          std::vector<Eigen::VectorXd>* gradients, void* data);

  // Calculate the cost and gradients of the collision potential.
  static double getCostAndGradientCollision(
          std::vector<Eigen::VectorXd>* gradients, void* data);

  // Calculate the cost and gradient of the collision potential at the
  // current position. (See paper [3]
  static double getCostAndGradientPotentialESDF(
          const Eigen::VectorXd& position, Eigen::VectorXd* gradient,
          void* data);

  // Calculate the cost of the collision potential at a given distance to the
  // obstacle (ie. current distance to obstacle)
  double getCostPotential(double collision_distance);

  // Evaluates the maximum magnitude constraint at the current value of
  // the optimization variables.
  // All input parameters are ignored, all information is contained in data.
  static double evaluateMaximumMagnitudeConstraint(
      const std::vector<double>& optimization_variables,
      std::vector<double>& gradient, void* data);

  // Does the actual optimization work for the time-only version.
  int optimizeTime();

  // Does the actual optimization work for the full optimization version.
  int optimizeTimeAndFreeConstraints();

  // Does the actual optimization work for optimizing only the Free Constraints.
  int optimizeFreeConstraints();

  // Does the actual optimization work for optimizing the Free Constraints
  // with an objective function including a derivative term and the collision
  // potential.
  int optimizeFreeConstraintsAndCollision();

  // Does the actual optimization work for optimizing the Free Constraints
  // and the Segment Timeswith an objective function including a derivative
  // term,  collision potential and time term.
  int optimizeFreeConstraintsAndCollisionAndTime();

  // Evaluates the maximum magnitude constraints as soft constraints and
  // returns a cost, depending on the violation of the constraints.
  // cost_i = min(maximum_cost, exp(abs_violation_i / max_allowed_i * weight))
  //  cost = sum(cost_i)
  // Input: inequality_constraints = Vector of ConstraintData shared_ptrs,
  // describing the constraints.
  // Input: weight = Multiplicative weight of the constraint violation.
  // Input: maximum_cost = Upper bound of the cost. Necessary, since exp of a
  // high violation can end up in inf.
  // Output: Sum of the costs per constraint.
  double evaluateMaximumMagnitudeAsSoftConstraint(
      const std::vector<std::shared_ptr<ConstraintData> >&
          inequality_constraints,
      double weight, double maximum_cost = 1.0e12) const;

  // Computes the total trajectory time.
  static double computeTotalTrajectoryTime(
      const std::vector<double>& segment_times);

  // nlopt optimization object.
  std::shared_ptr<nlopt::opt> nlopt_;

  // Underlying linear optimization object.
  PolynomialOptimization<N> poly_opt_;

  // Parameters for the nonlinear optimzation.
  NonlinearOptimizationParameters optimization_parameters_;

  // Holds the data for evaluating inequality constraints.
  std::vector<std::shared_ptr<ConstraintData> > inequality_constraints_;

  OptimizationInfo optimization_info_;

  // Number of polynomials, e.g 3 for a 3D path.
  size_t dimension_;

  // The vertices of the trajectory
  Vertex::Vector vertices_;

  // Derivative to optimize
  int derivative_to_optimize_;

  // L = A_inv * M
  Eigen::MatrixXd L_;

  // Matrix for mapping a vector of polynomial coefficients of a function to
  // the polynomial coefficients of its derivative
  // [0 1 0 0 0 ...]              f_k(t) = a0 + a1*t + a2*t^2 + a3*t^3 + ...
  // [0 0 2 0 0 ...]          df_k(t)/dt =      a1   + 2*a2*t + 3*a3*t^2 + ...
  // [0 0 0 3 0 ...]                    with T = [t^0 t^1 t^2 t^3 t^4 ...]
  // [0 0 0 0 4 ...]            -->     f_k(t) = T * p_k
  // [  ...   ...  ]            --> df_k(t)/dt = T * V * p_k
  Eigen::MatrixXd V_;
  Eigen::MatrixXd V_all_segments_;

  // Signed Distance Field needed for optimizing the collision potential
  std::shared_ptr<sdf_tools::SignedDistanceField> sdf_;

  // Do we solve with or without position constraints for the vertices
  // betwenn start and goal?
  bool solve_with_position_constraint_;

  // Linear solution / Initial guess
  Trajectory trajectory_initial_;
};

}  // namespace mav_trajectory_generation

namespace nlopt {
// Convenience function that turns nlopt's return values into something
// readable.
std::string returnValueToString(int return_value);
}  // namespace nlopt

#endif  // MAV_TRAJECTORY_GENERATION_POLYNOMIAL_OPTIMIZATION_NONLINEAR_H_

#include "mav_trajectory_generation/impl/polynomial_optimization_nonlinear_impl.h"