/*
* Copyright (c) 2015, Markus Achtelik, ASL, ETH Zurich, Switzerland
* You can contact the author at <markus dot achtelik at mavt dot ethz dot ch>
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

#ifndef MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_
#define MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_

#include <chrono>

#include "mav_trajectory_generation/polynomial_optimization_linear.h"
#include "mav_trajectory_generation/timing.h"

namespace mav_trajectory_generation {

inline void OptimizationInfo::print(std::ostream& stream) const {
  stream << "--- optimization info ---" << std::endl;
  stream << "  optimization time:     " << optimization_time << std::endl;
  stream << "  n_iterations:          " << n_iterations << std::endl;
  stream << "  stopping reason:       "
         << nlopt::returnValueToString(stopping_reason) << std::endl;
  stream << "  cost trajectory:       " << cost_trajectory << std::endl;
  stream << "  cost collision:        " << cost_collision << std::endl;
  stream << "  cost time:             " << cost_time << std::endl;
  stream << "  cost soft constraints: " << cost_soft_constraints << std::endl;
  stream << "  sum: " << cost_trajectory + cost_collision + cost_time +
          cost_soft_constraints << std::endl;
  stream << "  maxima: " << std::endl;
  for (const std::pair<int, Extremum>& m : maxima) {
    stream << "    " << positionDerivativeToString(m.first) << ": "
           << m.second.value << " in segment " << m.second.segment_idx
           << " and segment time " << m.second.time << std::endl;
  }
}

template <int _N>
PolynomialOptimizationNonLinear<_N>::PolynomialOptimizationNonLinear(
    size_t dimension, const NonlinearOptimizationParameters& parameters)
    : poly_opt_(dimension),
      dimension_(dimension),
      derivative_to_optimize_(derivative_order::kINVALID),
      optimization_parameters_(parameters) {}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::setupFromVertices(
    const Vertex::Vector& vertices, const std::vector<double>& segment_times,
    int derivative_to_optimize) {
  derivative_to_optimize_ = derivative_to_optimize;
  vertices_ = vertices;

  bool ret = poly_opt_.setupFromVertices(vertices, segment_times,
                                         derivative_to_optimize);

  size_t n_optimization_parameters;
  switch (optimization_parameters_.objective) {
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraints:
      n_optimization_parameters =
              poly_opt_.getNumberFreeConstraints() * poly_opt_.getDimension();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndTime:
      n_optimization_parameters =
              segment_times.size() +
              poly_opt_.getNumberFreeConstraints() * poly_opt_.getDimension();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeTime:
      n_optimization_parameters = segment_times.size();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollision:
      n_optimization_parameters =
              poly_opt_.getNumberFreeConstraints() * poly_opt_.getDimension();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollisionAndTime:
      n_optimization_parameters =
              segment_times.size() +
              poly_opt_.getNumberFreeConstraints() * poly_opt_.getDimension();
      break;
    default:
      LOG(ERROR) << "Unknown Optimization Objective. Abort.";
      break;
  }

  nlopt_.reset(new nlopt::opt(optimization_parameters_.algorithm,
                              n_optimization_parameters));
  nlopt_->set_ftol_rel(optimization_parameters_.f_rel);
  nlopt_->set_ftol_abs(optimization_parameters_.f_abs);
  nlopt_->set_xtol_rel(optimization_parameters_.x_rel);
  nlopt_->set_xtol_abs(optimization_parameters_.x_abs);
  nlopt_->set_maxeval(optimization_parameters_.max_iterations);
  nlopt_->set_maxtime(optimization_parameters_.max_time);

  if (optimization_parameters_.random_seed < 0)
    nlopt_srand_time();
  else
    nlopt_srand(optimization_parameters_.random_seed);

  return ret;
}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::solveLinear() {
  return poly_opt_.solveLinear();
}

template <int _N>
bool PolynomialOptimizationNonLinear<_N
>::computeInitialSolutionWithoutPositionConstraints() {
  // compute initial solution
  poly_opt_.solveLinear();

  // Save the trajectory from the initial guess/solution
  trajectory_initial_.clear();
  getTrajectory(&trajectory_initial_);

  // Get dimension
  const size_t dim = poly_opt_.getDimension();

  // 2) Get the coefficients from the segments
  mav_trajectory_generation::Segment::Vector segments;
  poly_opt_.getSegments(&segments);
  std::vector<Eigen::VectorXd> p(dim, Eigen::VectorXd(N * segments.size()));

  for (int i = 0; i < dim; ++i) {
    for (size_t j = 0; j < segments.size(); ++j) {
      p[i].segment<N>(j * N) = segments[j][i].getCoefficients(0);
    }
  }

  // 3) Remove all position constraints apart from start and goal
  Vertex::Vector vertices = vertices_;
  for (int k = 1; k < vertices.size() - 1 ; ++k) {
    vertices_[k].removeConstraint(
            mav_trajectory_generation::derivative_order::POSITION);
  }

  std::cout << "vertices_: " << vertices_ << std::endl;

  // 4) Setup poly_opt_ again with new set of constraints
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);
  setupFromVertices(vertices_, segment_times, derivative_to_optimize_);

  // Save initial segment time
  trajectory_time_initial_ = computeTotalTrajectoryTime(segment_times);

  // Parameters after removing constraints
  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = poly_opt_.getNumberFixedConstraints();

  // TODO: move to linear solver. add method setFreeConstraintsFromCoefficients
  // 5) Get your new mapping matrix L (p = L*[d_f d_P]^T = A^(-1)*M*[d_f d_P]^T)
  // Fixed constraints are the same except plus the position constraints we
  // removed. Add those removed position constraints to free derivatives.
  Eigen::MatrixXd M_pinv, A, A_inv;
  poly_opt_.getA(&A);
  poly_opt_.getMpinv(&M_pinv);

  // 6) Calculate your reordered endpoint-derivatives. d_all = L^(-1) * p_k
  // where p_k are the old coefficients from the original linear solution and
  // L the new remapping matrix
  // d_all has the same size before and after removing constraints
  Eigen::VectorXd d_all(n_fixed_constraints + n_free_constraints);
  std::vector<Eigen::VectorXd> d_p(dim, Eigen::VectorXd(n_free_constraints));
  for (int i = 0; i < dim; ++i) {
    d_all = M_pinv * A * p[i]; // Old coeff p, but new ordering M_pinv * A
    d_p[i] = d_all.tail(n_free_constraints);
  }

  // 7) Set free constraints of problem according to initial solution and
  // removed constraints
  poly_opt_.setFreeConstraints(d_p);

  return true;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimize() {
  optimization_info_ = OptimizationInfo();
  int result = nlopt::FAILURE;

  const std::chrono::high_resolution_clock::time_point t_start =
      std::chrono::high_resolution_clock::now();

  switch (optimization_parameters_.objective) {
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraints:
      result = optimizeFreeConstraints();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndTime:
      result = optimizeTimeAndFreeConstraints();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeTime:
      result = optimizeTime();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollision:
      result = optimizeFreeConstraintsAndCollision();
      break;
    case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollisionAndTime:
      result = optimizeFreeConstraintsAndCollisionAndTime();
      break;
    default:
      LOG(ERROR) << "Unknown Optimization Objective. Abort.";
      break;
  }

  const std::chrono::high_resolution_clock::time_point t_stop =
      std::chrono::high_resolution_clock::now();
  optimization_info_.optimization_time =
      std::chrono::duration_cast<std::chrono::duration<double> >(t_stop -
                                                                 t_start)
          .count();

  optimization_info_.stopping_reason = result;

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeTime() {
  std::vector<double> initial_step, segment_times, upper_bounds;

  // Retrieve the segment times
  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  // TODO: FIX PROPERLY
  // compute initial solution
  poly_opt_.solveLinear();
  // Save the trajectory from the initial guess/solution
  trajectory_initial_.clear();
  getTrajectory(&trajectory_initial_);
  // Save the trajectory from the initial guess/solution
  trajectory_initial_after_removing_pos_.clear();
  getTrajectory(&trajectory_initial_after_removing_pos_);

  initial_step.reserve(n_segments);
  for (double t : segment_times) {
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel * t);
  }

  // TODO: replace with multiplier
  for (double t : segment_times) {
    upper_bounds.push_back(t * 2.0);
  }

  // TODO: no need to calculate twice. Calculate in comptueIntitialSol...
  // Calculate L
  Eigen::MatrixXd M, A_inv;
  poly_opt_.getM(&M);
  poly_opt_.getAInverse(&A_inv);

  L_ = Eigen::MatrixXd(A_inv * M);

  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  calculatePolynomialDerivativeMappingMatrices();

  try {
    // Set a lower bound on the segment time per segment to avoid numerical
    // issues.
    constexpr double kOptimizationTimeLowerBound = 0.1;
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_lower_bounds(kOptimizationTimeLowerBound);
    nlopt_->set_min_objective(
        &PolynomialOptimizationNonLinear<N>::objectiveFunctionTime, this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    result = nlopt_->optimize(segment_times, final_cost);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeFreeConstraints() {
  std::vector<double> initial_step, initial_solution, lower_bounds,
          upper_bounds;

  // compute initial solution
  if (optimization_parameters_.solve_with_position_constraint) {
    poly_opt_.solveLinear();
    // TODO: find better way of doing this
    // Save the trajectory from the initial guess/solution
    trajectory_initial_.clear();
    getTrajectory(&trajectory_initial_);
  } else {
    computeInitialSolutionWithoutPositionConstraints();
  }

  // Save the trajectory from the initial guess/solution
  trajectory_initial_after_removing_pos_.clear();
  getTrajectory(&trajectory_initial_after_removing_pos_);

  // Set up variables
  std::vector<Eigen::VectorXd> free_constraints;
  poly_opt_.getFreeConstraints(&free_constraints);
  CHECK(free_constraints.size() > 0);
  CHECK(free_constraints.front().size() > 0);

  const size_t n_optmization_variables =
          free_constraints.size() * free_constraints.front().size();

  initial_solution.reserve(n_optmization_variables);
  initial_step.reserve(n_optmization_variables);
  lower_bounds.reserve(n_optmization_variables);
  upper_bounds.reserve(n_optmization_variables);

  // TODO: no need to calculate twice. Calculate in comptueIntitialSol...
  // Calculate L
  Eigen::MatrixXd M, A_inv;
  poly_opt_.getM(&M);
  poly_opt_.getAInverse(&A_inv);

  L_ = Eigen::MatrixXd(A_inv * M);

  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  calculatePolynomialDerivativeMappingMatrices();

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  initial_step.reserve(n_optmization_variables);
  for (double x : initial_solution) {
    const double abs_x = std::abs(x);
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel *
                           abs_x);
  }

  setFreeEndpointDerivativeHardConstraints(initial_solution, &lower_bounds,
                                           &upper_bounds);

  // TODO: REMOVE only debug
  lower_bounds_ = lower_bounds;
  upper_bounds_ = upper_bounds;

  try {
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_lower_bounds(lower_bounds);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_min_objective(&PolynomialOptimizationNonLinear<
                                      N>::objectiveFunctionFreeConstraints,
                              this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nlin_deriv");
    result = nlopt_->optimize(initial_solution, final_cost);
    timer_solve.Stop();
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeFreeConstraintsAndCollision() {
  // compute initial solution
  if (optimization_parameters_.solve_with_position_constraint) {
    poly_opt_.solveLinear();
    // TODO: find better way of doing this
    // Save the trajectory from the initial guess/solution
    trajectory_initial_.clear();
    getTrajectory(&trajectory_initial_);
  } else {
    computeInitialSolutionWithoutPositionConstraints();
  }

  // Save the trajectory from the initial guess/solution
  trajectory_initial_after_removing_pos_.clear();
  getTrajectory(&trajectory_initial_after_removing_pos_);

  // Get and check free constraints and get number of optimization variables
  std::vector<Eigen::VectorXd> free_constraints;
  poly_opt_.getFreeConstraints(&free_constraints);
  CHECK(free_constraints.size() > 0);
  CHECK(free_constraints.front().size() > 0);

  const size_t n_optmization_variables =
          free_constraints.size() * free_constraints.front().size();

  std::vector<double> initial_step, initial_solution, lower_bounds,
          upper_bounds;
  initial_solution.reserve(n_optmization_variables);
  initial_step.reserve(n_optmization_variables);
  lower_bounds.reserve(n_optmization_variables);
  upper_bounds.reserve(n_optmization_variables);

  // TODO: no need to calculate twice. Calculate in comptueIntitialSol...
  // Calculate L
  Eigen::MatrixXd M, A_inv;
  poly_opt_.getM(&M);
  poly_opt_.getAInverse(&A_inv);

  L_ = Eigen::MatrixXd(A_inv * M);

  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  calculatePolynomialDerivativeMappingMatrices();

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  // Get the lower and upper bounds constraints on the free endpoint derivatives
  setFreeEndpointDerivativeHardConstraints(initial_solution, &lower_bounds,
                                           &upper_bounds);

  // TODO: REMOVE only debug
  lower_bounds_ = lower_bounds;
  upper_bounds_ = upper_bounds;

  initial_step.reserve(n_optmization_variables);
  for (double x : initial_solution) {
    const double abs_x = std::abs(x);
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel *
                                   abs_x);
  }

  if (optimization_parameters_.print_debug_info) {
    std::cout << "NLOPT X BOUNDS: LOWER | UPPER || INITIAL SOL || INITIAL STEP"
              << std::endl;
    for (int j = 0; j < lower_bounds.size(); ++j) {
      std::cout << j << ": " << lower_bounds[j] << " | "
                << upper_bounds[j] << " || "
                << initial_solution[j] << " || "
                << initial_step[j] << std::endl;
    }
    std::cout << std::endl;
  }

  try {
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_lower_bounds(lower_bounds);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_min_objective(
            &PolynomialOptimizationNonLinear<
                    N>::objectiveFunctionFreeConstraintsAndCollision,
            this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nlin_deriv_coll");
    result = nlopt_->optimize(initial_solution, final_cost);
    timer_solve.Stop();
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeTimeAndFreeConstraints() {
  std::vector<double> initial_step, initial_solution, segment_times,
      lower_bounds, upper_bounds;

  // Get the segment times
  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  // compute initial solution
  poly_opt_.solveLinear();
  std::vector<Eigen::VectorXd> free_constraints;
  poly_opt_.getFreeConstraints(&free_constraints);
  CHECK(free_constraints.size() > 0);
  CHECK(free_constraints.front().size() > 0);

  // TODO: FIX PROPERLY
  // Save the trajectory from the initial guess/solution
  trajectory_initial_.clear();
  getTrajectory(&trajectory_initial_);
  // Save the trajectory from the initial guess/solution
  trajectory_initial_after_removing_pos_.clear();
  getTrajectory(&trajectory_initial_after_removing_pos_);

  const size_t n_optmization_variables =
      n_segments + free_constraints.size() * free_constraints.front().size();

  initial_solution.reserve(n_optmization_variables);
  initial_step.reserve(n_optmization_variables);
  lower_bounds.reserve(n_optmization_variables);
  upper_bounds.reserve(n_optmization_variables);

  // TODO: no need to calculate twice. Calculate in comptueIntitialSol...
  // Calculate L
  Eigen::MatrixXd M, A_inv;
  poly_opt_.getM(&M);
  poly_opt_.getAInverse(&A_inv);

  L_ = Eigen::MatrixXd(A_inv * M);

  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  calculatePolynomialDerivativeMappingMatrices();

  // copy all constraints into one vector:
  for (double t : segment_times) {
    initial_solution.push_back(t);
  }

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  initial_step.reserve(n_optmization_variables);
  for (double x : initial_solution) {
    const double abs_x = std::abs(x);
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel *
                           abs_x);
    lower_bounds.push_back(-abs_x * 2);
    upper_bounds.push_back(abs_x * 2);
  }

  for (size_t i = 0; i < n_segments; ++i) {
    lower_bounds[i] = 0.1;
  }

  // TODO: REMOVE only debug
  lower_bounds_ = lower_bounds;
  upper_bounds_ = upper_bounds;

  try {
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_lower_bounds(lower_bounds);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_min_objective(&PolynomialOptimizationNonLinear<
                                  N>::objectiveFunctionTimeAndConstraints,
                              this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nlin_deriv_time");
    result = nlopt_->optimize(initial_solution, final_cost);
    timer_solve.Stop();
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N
>::optimizeFreeConstraintsAndCollisionAndTime() {
  std::vector<double> initial_step, initial_solution, segment_times,
          lower_bounds, upper_bounds;

  // compute initial solution
  if (optimization_parameters_.solve_with_position_constraint) {
    poly_opt_.solveLinear();
    // TODO: find better way of doing this
    // Save the trajectory from the initial guess/solution
    trajectory_initial_.clear();
    getTrajectory(&trajectory_initial_);
  } else {
    computeInitialSolutionWithoutPositionConstraints();
  }

  // Save the trajectory from the initial guess/solution
  trajectory_initial_after_removing_pos_.clear();
  getTrajectory(&trajectory_initial_after_removing_pos_);

  // Get segment times
  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  // Get and check free constraints and get number of optimization variables
  std::vector<Eigen::VectorXd> free_constraints;
  poly_opt_.getFreeConstraints(&free_constraints);
  CHECK(free_constraints.size() > 0);
  CHECK(free_constraints.front().size() > 0);

  const size_t n_optmization_variables =
          n_segments + free_constraints.size() * free_constraints.front().size();

  initial_solution.reserve(n_optmization_variables);
  initial_step.reserve(n_optmization_variables);
  lower_bounds.reserve(n_optmization_variables);
  upper_bounds.reserve(n_optmization_variables);

  // TODO: no need to calculate twice. Calculate in comptueIntitialSol...
  // Calculate L
  Eigen::MatrixXd M, A_inv;
  poly_opt_.getM(&M);
  poly_opt_.getAInverse(&A_inv);

  L_ = Eigen::MatrixXd(A_inv * M);

  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  calculatePolynomialDerivativeMappingMatrices();

  // copy all constraints into one vector:
  for (double t : segment_times) {
    initial_solution.push_back(t);
  }

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  // Retrieve the free endpoint derivative initial solution
  std::vector<double> initial_solution_free(
          initial_solution.begin()+n_segments,initial_solution.end());
  // Setup for getting bounds on the free endpoint derivatives
  std::vector<double> lower_bounds_free, upper_bounds_free;
  const size_t n_optmization_variables_free =
          free_constraints.size() * free_constraints.front().size();
  lower_bounds_free.reserve(n_optmization_variables_free);
  upper_bounds_free.reserve(n_optmization_variables_free);

  // Get the lower and upper bounds constraints on the free endpoint derivatives
  setFreeEndpointDerivativeHardConstraints(
          initial_solution_free, &lower_bounds_free, &upper_bounds_free);

  // Set segment time constraints
  for (int l = 0; l < n_segments; ++l) {
    const double abs_x = std::abs(initial_solution[l]);
    lower_bounds.push_back(0.1);
    upper_bounds.push_back(HUGE_VAL);
  }
  // Append free endpoint derivative constraints
  lower_bounds.insert(std::end(lower_bounds), std::begin(lower_bounds_free),
                      std::end(lower_bounds_free));
  upper_bounds.insert(std::end(upper_bounds), std::begin(upper_bounds_free),
                      std::end(upper_bounds_free));

  // TODO: REMOVE only debug
  lower_bounds_ = lower_bounds;
  upper_bounds_ = upper_bounds;

  initial_step.reserve(n_optmization_variables);
  for (double x : initial_solution) {
    const double abs_x = std::abs(x);
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel *
                           abs_x);
  }

  std::cout << "NLOPT X BOUNDS: LOWER | UPPER || INITIAL SOL || INITIAL STEP"
            << std::endl;
  for (int j = 0; j < lower_bounds.size(); ++j) {
    std::cout << j << ": " << lower_bounds[j] << " | "
              << upper_bounds[j] << " || "
              << initial_solution[j] << " || "
              << initial_step[j] << std::endl;
  }
  std::cout << std::endl;

  try {
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_lower_bounds(lower_bounds);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_min_objective(
            &PolynomialOptimizationNonLinear<
                    N>::objectiveFunctionFreeConstraintsAndCollisionAndTime,
            this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nlin_deriv_coll_time");
    result = nlopt_->optimize(initial_solution, final_cost);
    timer_solve.Stop();
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::addMaximumMagnitudeConstraint(
    int derivative, double maximum_value) {
  CHECK_GE(derivative, 0);
  CHECK_GE(maximum_value, 0.0);

  std::shared_ptr<ConstraintData> constraint_data(new ConstraintData);
  constraint_data->derivative = derivative;
  constraint_data->value = maximum_value;
  constraint_data->this_object = this;

  // Store the shared_ptrs such that their data will be destroyed later.
  inequality_constraints_.push_back(constraint_data);

  if (!optimization_parameters_.use_soft_constraints) {
    try {
      nlopt_->add_inequality_constraint(
          &PolynomialOptimizationNonLinear<
              N>::evaluateMaximumMagnitudeConstraint,
          constraint_data.get(),
          optimization_parameters_.inequality_constraint_tolerance);
    } catch (std::exception& e) {
      LOG(ERROR) << "ERROR while setting inequality constraint " << e.what()
                 << std::endl;
      return false;
    }
  }
  return true;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionTime(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  CHECK(gradient.empty())
      << "computing gradient not possible, choose a gradient free method";
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
      static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  CHECK_EQ(segment_times.size(),
           optimization_data->poly_opt_.getNumberSegments());

  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.solveLinear();

  const double cost_trajectory = optimization_data->poly_opt_.computeCost();
  const double total_time = computeTotalTrajectoryTime(segment_times);
  const double cost_time = total_time * total_time *
          optimization_data->optimization_parameters_.time_penalty;

  bool is_collision = false;
  double cost_collision = 0.0;
  const double w_c = optimization_data->optimization_parameters_.weights.w_c;
  if (w_c > 0.0) {
    cost_collision = w_c * optimization_data->getCostAndGradientCollision(
            NULL, optimization_data, &is_collision);
  }

  double cost_constraints = 0.0;
  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
        optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
            optimization_data->inequality_constraints_,
            optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  collision: " << cost_collision << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_time + cost_constraints
              << std::endl;
    std::cout << "  sum iter0: " << optimization_data->total_cost_iter0_
              << std::endl;
    std::cout << "  total time: " << total_time << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_collision = cost_collision;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
      cost_constraints;

  // Save the total cost at iteration 0
  if (optimization_data->is_iter0_) {
    optimization_data->total_cost_iter0_ =
            cost_trajectory + cost_collision + cost_time + cost_constraints;
    optimization_data->is_iter0_ = false;
  }

  return cost_trajectory + cost_collision + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionTimeAndConstraints(
    const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK(gradient.empty())
      << "computing gradient not possible, choose a gradient free method";
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
      static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_segments = optimization_data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints =
      optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_segments + n_free_constraints * dim);

  // Retrieve segment times
  std::vector<double> segment_times;
  segment_times.reserve(n_segments);
  for (size_t i = 0; i < n_segments; ++i) segment_times.push_back(x[i]);

  // Retrieve free endpoint-derivative constraints
  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  for (size_t d = 0; d < dim; ++d) {
    const size_t idx_start = n_segments + d * n_free_constraints;

    Eigen::VectorXd& free_constraints_dim = free_constraints[d];
    free_constraints_dim.resize(n_free_constraints, Eigen::NoChange);
    for (size_t i = 0; i < n_free_constraints; ++i) {
      free_constraints_dim[i] = x[idx_start + i];
    }
  }

  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.setFreeConstraints(free_constraints);

  const double cost_trajectory = optimization_data->poly_opt_.computeCost();
  double cost_time = 0;
  double cost_constraints = 0;

  const double total_time = computeTotalTrajectoryTime(segment_times);
  cost_time = total_time * total_time *
              optimization_data->optimization_parameters_.time_penalty;

  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
            optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
                    optimization_data->inequality_constraints_,
                    optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_time + cost_constraints
              << std::endl;
    std::cout << "  total time: " << total_time << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
          cost_constraints;

  return cost_trajectory + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionFreeConstraints(
        const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_free_constraints =
          optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_free_constraints * dim);

  // Retrieve free endpoint-derivative constraints
  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  for (size_t d = 0; d < dim; ++d) {
    const size_t idx_start = d * n_free_constraints;

    Eigen::VectorXd& free_constraints_dim = free_constraints[d];
    free_constraints_dim.resize(n_free_constraints, Eigen::NoChange);
    for (size_t i = 0; i < n_free_constraints; ++i) {
      free_constraints_dim[i] = x[idx_start + i];
    }
  }

  optimization_data->poly_opt_.setFreeConstraints(free_constraints);

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "LOWER BOUNDS -- FREE CONSTRAINTS -- UPPER BOUNDS" << std::endl;
    for (size_t d = 0; d < dim; ++d) {
      for (int i = 0; i < free_constraints[0].size(); ++i) {
        const size_t idx_start = d * n_free_constraints;
        std::cout << d << " " << i << ": "
                  << optimization_data->lower_bounds_[idx_start+i] << " | "
                  << free_constraints[d][i] << " | "
                  << optimization_data->upper_bounds_[idx_start+i] << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::vector<Eigen::VectorXd> grad_d;
  grad_d.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  double J_d = 0.0;
  if (!gradient.empty()) {
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
  } else {
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
  }

  // TODO: get and multiply with weights
  const double cost_trajectory = J_d;
  double cost_constraints = 0.0;

  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
            optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
                    optimization_data->inequality_constraints_,
                    optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_constraints << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_soft_constraints =
          cost_constraints;

  if (!gradient.empty()) {
    gradient.clear();
    gradient.resize(3*n_free_constraints);

    for (int i = 0; i < n_free_constraints; ++i) {
      gradient[0 * n_free_constraints + i] = grad_d[0][i];
      gradient[1 * n_free_constraints + i] = grad_d[1][i];
      gradient[2 * n_free_constraints + i] = grad_d[2][i];
    }
  }

  return cost_trajectory + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::
objectiveFunctionFreeConstraintsAndCollision(
        const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_free_constraints =
          optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_free_constraints * dim);

  // Retrieve the free endpoint-derivative constraints
  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  for (size_t d = 0; d < dim; ++d) {
    const size_t idx_start = d * n_free_constraints;

    Eigen::VectorXd& free_constraints_dim = free_constraints[d];
    free_constraints_dim.resize(n_free_constraints, Eigen::NoChange);
    for (size_t i = 0; i < n_free_constraints; ++i) {
      free_constraints_dim[i] = x[idx_start + i];
    }
  }

  optimization_data->poly_opt_.setFreeConstraints(free_constraints);

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "LOWER BOUNDS -- FREE CONSTRAINTS -- UPPER BOUNDS" << std::endl;
    for (size_t d = 0; d < dim; ++d) {
      for (int i = 0; i < free_constraints[0].size(); ++i) {
        const size_t idx_start = d * n_free_constraints;
        std::cout << d << " " << i << ": "
                  << optimization_data->lower_bounds_[idx_start+i] << " | "
                  << free_constraints[d][i] << " | "
                  << optimization_data->upper_bounds_[idx_start+i] << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  bool is_collision;
  std::vector<Eigen::VectorXd> grad_d, grad_c, grad_sc;
  grad_d.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  grad_c.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  grad_sc.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  double J_d = 0.0;
  double J_c = 0.0;
  double J_sc = 0.0;
  if (!gradient.empty()) {
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            &grad_c, optimization_data, &is_collision);
    if (optimization_data->optimization_parameters_.use_soft_constraints) {
      J_sc = optimization_data->getCostAndGradientSoftConstraints(
              &grad_sc, optimization_data);
    }
  } else {
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            NULL, optimization_data, &is_collision);
    if (optimization_data->optimization_parameters_.use_soft_constraints) {
      J_sc = optimization_data->getCostAndGradientSoftConstraints(
              NULL, optimization_data);
    }
  }

  // Numerical gradients for collision cost
  if (!gradient.empty()) {
    if (optimization_data->optimization_parameters_.use_numeric_grad) {
      std::vector<Eigen::VectorXd> grad_c_numeric(dim, Eigen::VectorXd::Zero
              (n_free_constraints));

      optimization_data->getNumericalGradientsCollision(&grad_c_numeric,
                                                        optimization_data);

      std::cout << "grad_c | grad_c_numeric | diff | grad_sc: " << std::endl;
      for (int k = 0; k < dim; ++k) {
        for (int n = 0; n < n_free_constraints; ++n) {
          std::cout << k << " " << n << ": " << grad_c[k][n] << " | "
                    << grad_c_numeric[k][n] << " | "
                    << grad_c[k][n] - grad_c_numeric[k][n] << " | "
                    << grad_sc[k][n] << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      for (int k = 0; k < dim; ++k) {
        grad_c[k] = grad_c_numeric[k];
      }
    }
  }

  // Weighting terms for different costs
  const double w_d = optimization_data->optimization_parameters_.weights.w_d;
  const double w_c = optimization_data->optimization_parameters_.weights.w_c;
  const double w_sc = optimization_data->optimization_parameters_.weights.w_sc;

  // Get the weighted cost
  const double cost_trajectory = w_d * J_d;
  const double cost_collision = w_c * J_c;
  const double cost_constraints = w_sc * J_sc;

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  collision: " << cost_collision << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_collision +
            cost_constraints << std::endl;
    std::cout << "  sum iter0: " << optimization_data->total_cost_iter0_
              << std::endl;
  }

  // Save the trajectory of this iteration
  Trajectory trajectory_i;
  optimization_data->getTrajectory(&trajectory_i);
  optimization_data->all_trajectories_.push_back(trajectory_i);

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_collision = cost_collision;
  optimization_data->optimization_info_.cost_soft_constraints =
          cost_constraints;

  // Save the total cost at iteration 0
  if (optimization_data->is_iter0_) {
    optimization_data->total_cost_iter0_ =
            cost_trajectory + cost_collision + cost_constraints;
    optimization_data->is_iter0_ = false;
  }

  if (!gradient.empty()) {
    gradient.clear();
    gradient.resize(3*n_free_constraints);

    for (int i = 0; i < n_free_constraints; ++i) {
      for (int k = 0; k < dim; ++k) {
        gradient[k * n_free_constraints + i] =
                w_d * grad_d[k][i] + w_c * grad_c[k][i] + w_sc * grad_sc[k][i];
      }
    }
  }

  return cost_trajectory + cost_collision + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N
>::objectiveFunctionFreeConstraintsAndCollisionAndTime(
        const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_segments = optimization_data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints =
          optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_segments + n_free_constraints * dim);

  // Retrieve optimized segment times
  std::vector<double> segment_times;
  segment_times.reserve(n_segments);
  for (size_t i = 0; i < n_segments; ++i) segment_times.push_back(x[i]);

  // Retrieve optimized free endpoint-derivatives
  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  for (size_t d = 0; d < dim; ++d) {
    const size_t idx_start = n_segments + d * n_free_constraints;

    Eigen::VectorXd& free_constraints_dim = free_constraints[d];
    free_constraints_dim.resize(n_free_constraints, Eigen::NoChange);
    for (size_t i = 0; i < n_free_constraints; ++i) {
      free_constraints_dim[i] = x[idx_start + i];
    }
  }

  // Set segment times and free constraints back in trajectory
  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.setFreeConstraints(free_constraints);

  // Update L since new segment times
  Eigen::MatrixXd M, A_inv;
  optimization_data->poly_opt_.getM(&M);
  optimization_data->poly_opt_.getAInverse(&A_inv);
  optimization_data->L_ = Eigen::MatrixXd(A_inv * M);

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "LOWER BOUNDS -- FREE CONSTRAINTS -- UPPER BOUNDS" << std::endl;
    for (int i = 0; i < segment_times.size(); ++i) {
      std::cout << "tm_" << i << ": "
                << optimization_data->lower_bounds_[i] << " | "
                << segment_times[i] << " | "
                << optimization_data->upper_bounds_[i] << std::endl;
    }
    std::cout << std::endl;
    for (size_t d = 0; d < dim; ++d) {
      for (int i = 0; i < free_constraints[0].size(); ++i) {
        const size_t idx_start = n_segments + d * n_free_constraints;
        std::cout << d << " " << i << ": "
                  << optimization_data->lower_bounds_[idx_start+i] << " | "
                  << free_constraints[d][i] << " | "
                  << optimization_data->upper_bounds_[idx_start+i] << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  bool is_collision;
  std::vector<Eigen::VectorXd> grad_d, grad_c, grad_sc;
  std::vector<double> grad_t(n_segments);
  grad_d.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  grad_c.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  grad_sc.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  double J_d = 0.0;
  double J_c = 0.0;
  double J_t = 0.0;
  double J_sc = 0.0;
  if (!gradient.empty()) {
    timing::Timer opti_deriv_timer("opti/deriv");
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
    opti_deriv_timer.Stop();
    timing::Timer opti_coll_timer("opti/coll");
    J_c = optimization_data->getCostAndGradientCollision(
            &grad_c, optimization_data, &is_collision);
    opti_coll_timer.Stop();

    if (optimization_data->optimization_parameters_.use_soft_constraints) {
      if (optimization_data->optimization_parameters_.is_simple_numgrad_constraints) {
        timing::Timer opti_constraints_simple_timer("opti/constraints_simple");
        J_sc = optimization_data->getCostAndGradientSoftConstraintsSimple(
                &grad_sc, optimization_data);
        opti_constraints_simple_timer.Stop();
      } else {
        timing::Timer opti_constraints_timer("opti/constraints");
        J_sc = optimization_data->getCostAndGradientSoftConstraints(
                &grad_sc, optimization_data);
        opti_constraints_timer.Stop();
      }
    }

    if (optimization_data->optimization_parameters_.is_simple_numgrad_time) {
      timing::Timer opti_time_simple_timer("opti/time_simple");
      J_t = optimization_data->getCostAndGradientTimeSimple(
              &grad_t, optimization_data, J_d, J_c, J_sc);
      opti_time_simple_timer.Stop();
    } else {
      timing::Timer opti_time_timer("opti/time");
      J_t = optimization_data->getCostAndGradientTime(&grad_t,
                                                      optimization_data);
      opti_time_timer.Stop();
    }
  } else {
    timing::Timer opti_deriv_timer("opti_gradfree/deriv");
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
    opti_deriv_timer.Stop();
    timing::Timer opti_coll_timer("opti_gradfree/coll");
    J_c = optimization_data->getCostAndGradientCollision(
            NULL, optimization_data, &is_collision);
    opti_coll_timer.Stop();
    timing::Timer opti_time_timer("opti_gradfree/time");
    J_t = optimization_data->getCostAndGradientTime(NULL, optimization_data);
    opti_time_timer.Stop();
    if (optimization_data->optimization_parameters_.use_soft_constraints) {
      if (optimization_data->optimization_parameters_.is_simple_numgrad_constraints) {
        timing::Timer opti_constraints_simple_timer("opti/constraints_simple");
        J_sc = optimization_data->getCostAndGradientSoftConstraintsSimple(
                NULL, optimization_data);
        opti_constraints_simple_timer.Stop();
      } else {
        timing::Timer opti_constraints_timer("opti_gradfree/constraints");
        J_sc = optimization_data->getCostAndGradientSoftConstraints(
                NULL, optimization_data);
        opti_constraints_timer.Stop();
      }
    }
  }

  // Numerical gradients for collision cost
  if (!gradient.empty()) {
    if (optimization_data->optimization_parameters_.use_numeric_grad) {
      std::vector<Eigen::VectorXd> grad_c_numeric(dim, Eigen::VectorXd::Zero
              (n_free_constraints));

      optimization_data->getNumericalGradientsCollision(&grad_c_numeric,
                                                        optimization_data);

      std::cout << "grad_c | grad_c_numeric | diff | grad_sc: "
                << std::endl;
      for (int k = 0; k < dim; ++k) {
        for (int n = 0; n < n_free_constraints; ++n) {
          std::cout << k << " " << n << ": " << grad_c[k][n] << " | "
                    << grad_c_numeric[k][n] << " | "
                    << grad_c[k][n] - grad_c_numeric[k][n] << " | "
                    << grad_sc[k][n] << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "grad_t: " << std::endl;
      for (int i = 0; i < n_segments; ++i) {
        std::cout << grad_t[i] << " | ";
      }
      std::cout << std::endl << std::endl;

      for (int k = 0; k < dim; ++k) {
        grad_c[k] = grad_c_numeric[k];
      }
    }
  }

  // Weighting terms for different costs
  const double w_d = optimization_data->optimization_parameters_.weights.w_d;
  const double w_c = optimization_data->optimization_parameters_.weights.w_c;
  const double w_sc = optimization_data->optimization_parameters_.weights.w_sc;
  const double w_t = optimization_data->optimization_parameters_.weights.w_t;

  // Get the weighted cost
  const double cost_trajectory = w_d * J_d;
  double cost_collision = w_c * J_c;
  const double cost_time = w_t * J_t;
  const double cost_constraints = w_sc * J_sc;

  // If in collision and total cost is smaller than initial total cost
  const double total_cost =
          cost_trajectory + cost_collision + cost_time + cost_constraints;

  if (optimization_data->optimization_parameters_.is_collision_safe) {
    if (optimization_data->optimization_parameters_.is_coll_raise_first_iter) {
      if (is_collision && (total_cost < optimization_data->total_cost_iter0_)) {
        cost_collision = optimization_data->total_cost_iter0_ -
                (total_cost - cost_collision) +
                optimization_data->optimization_parameters_.add_coll_raise;
//        LOG(INFO) << "COLLISION: Raise total cost to inital total cost.";
      }
    } else {
      const double total_cost_last_iter =
              optimization_data->optimization_info_.cost_trajectory +
              optimization_data->optimization_info_.cost_collision +
              optimization_data->optimization_info_.cost_time +
              optimization_data->optimization_info_.cost_soft_constraints;
      if (is_collision && (total_cost < total_cost_last_iter)) {
        cost_collision = total_cost_last_iter - (total_cost - cost_collision) +
                optimization_data->optimization_parameters_.add_coll_raise;
//        LOG(INFO) << "COLLISION: Raise total cost to initial total cost.";
      }
    }
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  collision: " << cost_collision << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_collision + cost_time +
                              cost_constraints << std::endl;
    std::cout << "  sum iter0: " << optimization_data->total_cost_iter0_
              << std::endl;

    std::cout << "SEGMENT TIMES" << std::endl;
    for (int j = 0; j < segment_times.size(); ++j) {
      std::cout << j << ": " << segment_times[j] << std::endl;
    }

    const double total_time =
            optimization_data->computeTotalTrajectoryTime(segment_times);
    std::cout << "INITIAL TIME: "
              << optimization_data->trajectory_time_initial_ << std::endl;
    std::cout << "TOTAL TIME: " << total_time << std::endl;
  }

  // Save the trajectory of this iteration
  Trajectory trajectory_i;
  optimization_data->getTrajectory(&trajectory_i);
  optimization_data->all_trajectories_.push_back(trajectory_i);

  // Update the optimization information
  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_collision = cost_collision;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
          cost_constraints;

  // Save the total cost at iteration 0
  if (optimization_data->is_iter0_) {
    optimization_data->total_cost_iter0_ =
            cost_trajectory + cost_collision + cost_time + cost_constraints;
    optimization_data->is_iter0_ = false;
  }

  if (!gradient.empty()) {
    gradient.clear();
    gradient.resize(n_segments + 3*n_free_constraints);

    if (optimization_data->optimization_parameters_.print_debug_info) {
      std::cout << std::endl << "GRADIENTS TIME: " << std::endl;
      for (int j = 0; j < n_segments; ++j) {
        std::cout << j << ": " << grad_t[j] << std::endl;
      }

      std::cout << "GRADIENTS D | C | SC: " << std::endl;
      for (int k = 0; k < dim; ++k) {
        const int start_idx = n_segments + (k * n_free_constraints);
        for (int i = 0; i < n_free_constraints; ++i) {
          std::cout << start_idx+i << ": "
                    << w_d * grad_d[k][i] << " | "
                    << w_c * grad_c[k][i] << " | "
                    << w_sc * grad_sc[k][i] << " | " << std::endl;
        }
      }
    }

    for (int j = 0; j < n_segments; ++j) {
      gradient[j] = grad_t[j];
    }

    for (int k = 0; k < dim; ++k) {
      const int start_idx = n_segments + (k * n_free_constraints);
      for (int i = 0; i < n_free_constraints; ++i) {
        gradient[start_idx + i] =
                w_d * grad_d[k][i] + w_c * grad_c[k][i] + w_sc * grad_sc[k][i];
      }
    }
  }

  return cost_trajectory + cost_collision + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientDerivative(
        std::vector<Eigen::VectorXd>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  // Compare the two approaches: getCost() and the full matrix.
  const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = data->poly_opt_.getNumberFixedConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  double J_d = 0.0;
  std::vector<Eigen::VectorXd> grad_d(
          dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Retrieve R
  Eigen::MatrixXd R;
  data->poly_opt_.getR(&R);

  // Set up mappings to R_FF R_FP R_PP etc. R_FP' = R_PF if that saves
  // time eventually.
  // All of these are the same per axis.
  // R_ff * d_f is actually constant so can cache this term.
  const Eigen::Block<Eigen::MatrixXd> R_ff =
          R.block(0, 0, n_fixed_constraints, n_fixed_constraints);
  const Eigen::Block<Eigen::MatrixXd> R_pf =
          R.block(n_fixed_constraints, 0, n_free_constraints,
                  n_fixed_constraints);
  const Eigen::Block<Eigen::MatrixXd> R_pp =
          R.block(n_fixed_constraints, n_fixed_constraints, n_free_constraints,
                  n_free_constraints);

  // Get d_p and d_f vector for all axes.
  std::vector<Eigen::VectorXd> d_p_vec;
  std::vector<Eigen::VectorXd> d_f_vec;
  data->poly_opt_.getFreeConstraints(&d_p_vec);
  data->poly_opt_.getFixedConstraints(&d_f_vec);

  Eigen::MatrixXd J_d_temp;
  // Compute costs over all axes.
  for (int k = 0; k < dim; ++k) {
    // Get a copy of d_p and d_f for this axis.
    const Eigen::VectorXd& d_p = d_p_vec[k];
    const Eigen::VectorXd& d_f = d_f_vec[k];

    // Now do the other thing.
    J_d_temp = d_f.transpose() * R_ff * d_f +
               d_f.transpose() * R_pf.transpose() * d_p +
               d_p.transpose() * R_pf * d_f + d_p.transpose() * R_pp * d_p;
    J_d += J_d_temp(0, 0);

    // And get the gradient.
    // Should really separate these out by k.
    grad_d[k] =
            2 * d_f.transpose() * R_pf.transpose() + 2 * d_p.transpose() * R_pp;
  }

  if (gradients != NULL) {
    gradients->clear();
    gradients->resize(dim);

    for (int k = 0; k < dim; ++k) {
      (*gradients)[k] = grad_d[k];
    }
  }

  return J_d;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientCollision(
        std::vector<Eigen::VectorXd>* gradients, void* opt_data,
        bool* is_collision) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  const size_t n_segments = data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = data->poly_opt_.getNumberFixedConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  double J_c = 0.0;
  std::vector<Eigen::VectorXd> grad_c(dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Get d_p and d_f vector for all axes.
  std::vector<Eigen::VectorXd> d_p_vec, d_f_vec;
  data->poly_opt_.getFreeConstraints(&d_p_vec);
  data->poly_opt_.getFixedConstraints(&d_f_vec);

  // 1) Get coefficients
  std::vector<Eigen::VectorXd> p_all_segments(dim, Eigen::VectorXd(N * n_segments));
  for (int k = 0; k < dim; ++k) {
    Eigen::VectorXd d_all_segments(n_fixed_constraints + n_free_constraints);
    d_all_segments.head(n_fixed_constraints) = d_f_vec[k];
    d_all_segments.tail(n_free_constraints) = d_p_vec[k];

    // The coefficients for each axis k with size (N * num_segments) x 1
    p_all_segments[k] = data->L_ * d_all_segments;
  }

  // Retrieve segment times
  std::vector<double> segment_times;
  data->poly_opt_.getSegmentTimes(&segment_times);

  // Get the correct L block to calculate derivatives.
  Eigen::Block<Eigen::MatrixXd> L_pp =
          data->L_.block(0, n_fixed_constraints,
                         data->L_.rows(), n_free_constraints);

  *is_collision = false;
  const double dt = data->optimization_parameters_.coll_check_time_increment;
  const double dist_sum_limit = data->optimization_parameters_.map_resolution;

  Eigen::VectorXd prev_pos(dim);
  prev_pos.setZero();
  // sum --> numerical integral
  double time_sum = -1;
  double dist_sum = 0.0;
  double t = 0.0;
  for (int i = 0; i < n_segments; ++i) {
    for (t = 0.0; t < segment_times[i]; t += dt) {

      // 2) Calculate the T vector (see paper equation (8)) for each segment
      Eigen::VectorXd T; // Is supposed to be a column-vector
      T.resize(N);
      T.setZero();
      for (int n = 0; n < N; ++n) {
        T[n] = pow(t, n);
      }

      // Create T for all segments
      Eigen::VectorXd T_all_seg(n_segments * N);
      T_all_seg.setZero();
      T_all_seg.segment(i * N, N) = T;

      // 3) Calculate position and velocity (see paper equation (9) and (11))
      Eigen::VectorXd pos(dim), vel(dim);
      pos.setZero();
      vel.setZero();
      for (int k = 0; k < dim; ++k) {
        // Coeff for this segment
        Eigen::Block<Eigen::VectorXd> p_k =
                p_all_segments[k].block(i * N, 0, N, 1);

        // TODO: pos bound checking
        pos(k) = (T.transpose() * p_k)(0);
        vel(k) = (T.transpose() * data->V_ * p_k)(0);
      }

      // Numerical integration --> Skip first entry
      if (time_sum < 0) {
        time_sum = 0.0;
        prev_pos = pos;
        continue;
      }

      time_sum += dt;
      dist_sum += (pos - prev_pos).norm();
      prev_pos = pos;

      if (dist_sum < dist_sum_limit) { continue; }

      // Cost and gradient of potential map from esdf
      bool is_pos_collision;
      Eigen::VectorXd grad_c_d_f(dim); // dc/dd_f
      double c = 0.0;
      if (gradients != NULL) {
        c = getCostAndGradientPotentialESDF(pos, &grad_c_d_f, data,
                                            &is_pos_collision);
      } else {
        c = getCostAndGradientPotentialESDF(pos, NULL, data, &is_pos_collision);
      }

      if (is_pos_collision) { *is_collision = true; }

      // Cost per segment and time sample
      const double J_c_i_t = c * vel.norm() * time_sum;
      J_c += J_c_i_t;

      // TODO: Only DEBUG
      // Numerical gradients
      if (data->optimization_parameters_.use_numeric_grad) {
        const double increment_dist =
                data->optimization_parameters_.map_resolution;
        Eigen::VectorXd increment(dim);
        Eigen::VectorXd grad_c_k_num(dim);
        grad_c_k_num.setZero();
        for (int k = 0; k < dim; ++k) {
          increment.setZero();
          increment[k] = increment_dist;

          bool is_collision_left, is_collision_right;
          const double cost_left = getCostAndGradientPotentialESDF(
                  pos - increment, NULL, data, &is_collision_left);
          const double cost_right = getCostAndGradientPotentialESDF(
                  pos + increment, NULL, data, &is_collision_right);
          grad_c_k_num[k] += (cost_right - cost_left) / (2.0 * increment_dist);
        }

        if (gradients != NULL) {
          if (data->optimization_parameters_.print_debug_info) {
            const Eigen::VectorXd diff = grad_c_d_f - grad_c_k_num;
            if ((diff.array() != 0.0).any()) {
              std::cout << "diff grad num coll potential: " << diff[0] << " | "
                        << diff[1] << " | " << diff[2] << std::endl;
            }
          }
        }
      }

      if (gradients != NULL) {
        // Norm has to be non-zero
        if (vel.norm() > 1e-6) {
          // Calculate gradient per axis
          for (int k = 0; k < dim; ++k) {
            // See paper equation (14)
            const Eigen::VectorXd grad_c_k =
                    (vel.norm() * time_sum * grad_c_d_f(k) * T_all_seg.transpose() * L_pp +
                            time_sum * c * vel(k) / vel.norm() * T_all_seg.transpose() *
                                    data->V_all_segments_ * L_pp).transpose();

            grad_c[k] += grad_c_k;
          }
        } else {
          LOG(INFO) << "VELOCITY NORM < 1e-6! --> DROP GRADIENT.";
        }
      }

      // Clear numeric integrals
      dist_sum = 0.0;
      time_sum = 0.0;
      prev_pos = pos;
    }

    // Make sure the dt is correct for the next segment:
    time_sum += -dt + (segment_times[i] - t);
  }

  if (gradients != NULL) {
    gradients->clear();
    gradients->resize(dim);

    for (int k = 0; k < dim; ++k) {
      (*gradients)[k] = grad_c[k];
    }
  }

  return J_c;
}


template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientPotentialESDF(
        const Eigen::VectorXd& position, Eigen::VectorXd* gradient,
        void* opt_data, bool* is_collision) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  const Eigen::Vector3d min_bound = data->optimization_parameters_.min_bound;
  const Eigen::Vector3d max_bound = data->optimization_parameters_.max_bound;

  // TODO: How to make sure trajectory stays within bounds?
  bool is_valid_state = true;
  const double increment_dist = data->optimization_parameters_.map_resolution;
  if (position[0] < min_bound[0]+increment_dist ||
          position[0] > max_bound[0]-increment_dist ||
          position[1] < min_bound[1]+increment_dist ||
          position[1] > max_bound[1]-increment_dist ||
          position[2] < min_bound[2]+increment_dist ||
          position[2] > max_bound[2]-increment_dist) {
    is_valid_state = false;
  }

  // Get distance from collision at current position
  double distance = 0.0;
  if (is_valid_state && data->optimization_parameters_.use_continous_distance) {
    distance = data->getDistanceSDF(position, data->sdf_);
  } else {
    distance = data->sdf_->Get(position[0], position[1], position[2]);
  }

  // Get potential cost from distance to collision
  const double J_c_esdf = data->getCostPotential(distance, is_collision);

  if (gradient != NULL) {
    // Numerical gradients
    Eigen::VectorXd grad_c_potential(data->dimension_);
    grad_c_potential.setZero();
    Eigen::VectorXd increment(data->dimension_);
    for (int k = 0; k < data->dimension_; ++k) {
      increment.setZero();
      increment[k] = increment_dist;

      // Get distance and potential cost from collision at current position
      double left_dist, right_dist;
      if (is_valid_state &&
              data->optimization_parameters_.use_continous_distance) {
        left_dist = data->getDistanceSDF(position-increment, data->sdf_);
        right_dist = data->getDistanceSDF(position+increment, data->sdf_);
      } else { // Discretized uniform grid
        left_dist = data->sdf_->Get3d(position-increment);
        right_dist = data->sdf_->Get3d(position+increment);
      }

      bool is_collision_left, is_collision_right;
      const double left_cost = data->getCostPotential(left_dist,
                                                      &is_collision_left);
      const double right_cost = data->getCostPotential(right_dist,
                                                       &is_collision_right);

      grad_c_potential[k] = (right_cost - left_cost) / (2.0 * increment_dist);
    }

    (*gradient) = grad_c_potential;

    // TODO: ONLY DEBUG
    if (!is_valid_state) {
      std::cout << "position: " << position[0] << " | "
                << position[1] << " | " << position[2]
                << " || distance: " << distance
                << " | J_c_esdf: " << J_c_esdf << std::endl;

      std::cout << "grad_c_potential: ";
      for (int i = 0; i < grad_c_potential.size(); ++i) {
        std::cout << grad_c_potential[i] << " | ";
      }
      std::cout << std::endl;
    }
  }

  return J_c_esdf;
}

template <int _N>
std::vector<std::pair<float, bool>> PolynomialOptimizationNonLinear<_N>::
getNeighborsSDF(const std::vector<int64_t>& idx,
                const std::shared_ptr<sdf_tools::SignedDistanceField>& sdf) {
  std::vector<std::pair<float, bool>> q;

  // Get distances of 8 corner neighbors
  q.push_back(sdf->GetSafe(idx[0]-1, idx[1]-1, idx[2]-1)); // q000
  q.push_back(sdf->GetSafe(idx[0]-1, idx[1]-1, idx[2]+1)); // q001
  q.push_back(sdf->GetSafe(idx[0]-1, idx[1]+1, idx[2]-1)); // q010
  q.push_back(sdf->GetSafe(idx[0]-1, idx[1]+1, idx[2]+1)); // q011
  q.push_back(sdf->GetSafe(idx[0]+1, idx[1]-1, idx[2]-1)); // q100
  q.push_back(sdf->GetSafe(idx[0]+1, idx[1]-1, idx[2]+1)); // q101
  q.push_back(sdf->GetSafe(idx[0]+1, idx[1]+1, idx[2]-1)); // q110
  q.push_back(sdf->GetSafe(idx[0]+1, idx[1]+1, idx[2]+1)); // q111

  return q;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getDistanceSDF(
        const Eigen::Vector3d& position,
        const std::shared_ptr<sdf_tools::SignedDistanceField>& sdf) {

  // Convert location to grid index
  const std::vector<int64_t> idx = sdf->LocationToGridIndex3d(position);
  // Retrieve all neighbouring grid cells
  const std::vector<std::pair<float, bool>> q_xyz = getNeighborsSDF(idx, sdf);

  // Check if all neighbors are valid nodes (inside bounds)
  bool is_valid = true;
  for (const auto& q : q_xyz) {
    if (!q.second){
      is_valid = false;
      break;
    }
  }

  double distance = 0.0;
  if (!is_valid) {
    distance = sdf->Get(position[0], position[1], position[2]);
    std::cout << "NOT VALID DIST: " << distance << std::endl;
  } else {
    // Get positions x0, x1, y0, y1, z0, z1
    std::vector<double> x0y0z0 =
            sdf->GridIndexToLocation(idx[0]-1, idx[1]-1, idx[2]-1);
    std::vector<double> x1y1z1 =
            sdf->GridIndexToLocation(idx[0]+1, idx[1]+1, idx[2]+1);

    distance = triLerp(
            position[0], position[1], position[2],
            q_xyz[0].first, q_xyz[1].first, q_xyz[2].first, q_xyz[3].first,
            q_xyz[4].first, q_xyz[5].first, q_xyz[6].first, q_xyz[7].first,
            x0y0z0[0], x1y1z1[0], x0y0z0[1], x1y1z1[1], x0y0z0[2], x1y1z1[2]);
  }

  return distance;
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::getNumericalGradientsCollision(
        std::vector<Eigen::VectorXd>* gradients_num, void* opt_data) {
  CHECK_NOTNULL(opt_data);
  CHECK_NOTNULL(gradients_num); // Num gradients only needed for grad-based opti

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  const size_t n_free_constraints =
          data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  // Get the current free constraints
  std::vector<Eigen::VectorXd> free_constraints;
  data->poly_opt_.getFreeConstraints(&free_constraints);

  gradients_num->clear();
  gradients_num->resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

  std::vector<Eigen::VectorXd> free_constraints_left, free_constraints_right;
  free_constraints_left.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  free_constraints_right.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  const double increment_dist = data->optimization_parameters_.map_resolution;

  std::vector<Eigen::VectorXd> increment(dim, Eigen::VectorXd::Zero
          (n_free_constraints));
  for (int k = 0; k < dim; ++k) {
    increment.clear();
    increment.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

    for (int n = 0; n < n_free_constraints; ++n) {
      increment[k].setZero();
      increment[k][n] = increment_dist;

      for (int k2 = 0; k2 < dim; ++k2) {
        free_constraints_left[k2] = free_constraints[k2] - increment[k2];
      }
      data->poly_opt_.setFreeConstraints(free_constraints_left);
      bool is_collision_left, is_collision_right;
      const double cost_left = data->getCostAndGradientCollision(
              NULL, data, &is_collision_left);

      for (int k2 = 0; k2 < dim; ++k2) {
        free_constraints_right[k2] = free_constraints[k2] + increment[k2];
      }
      data->poly_opt_.setFreeConstraints(free_constraints_right);
      const double cost_right = data->getCostAndGradientCollision(
              NULL, data, &is_collision_right);

      const double grad_k_n = (cost_right - cost_left) / (2.0 * increment_dist);
      gradients_num->at(k)[n] = grad_k_n;
    }
  }

  // Set again the original constraints from before calculating the numerical
  // constraints
  data->poly_opt_.setFreeConstraints(free_constraints);
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::getNumericalGradDerivatives(
        std::vector<Eigen::VectorXd>* gradients_num, void* opt_data) {
  CHECK_NOTNULL(opt_data);
  CHECK_NOTNULL(gradients_num); // Num gradients only needed for grad-based opti

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  const size_t n_free_constraints =
          data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  // Get the current free constraints
  std::vector<Eigen::VectorXd> free_constraints;
  data->poly_opt_.getFreeConstraints(&free_constraints);

  gradients_num->clear();
  gradients_num->resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

  std::vector<Eigen::VectorXd> free_constraints_left, free_constraints_right;
  free_constraints_left.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  free_constraints_right.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  const double increment_dist = data->optimization_parameters_.map_resolution;

  std::vector<Eigen::VectorXd> increment(
          dim, Eigen::VectorXd::Zero(n_free_constraints));
  for (int k = 0; k < dim; ++k) {

    increment.clear();
    increment.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    for (int n = 0; n < n_free_constraints; ++n) {

      increment[k].setZero();
      increment[k][n] = increment_dist;

      for (int k2 = 0; k2 < dim; ++k2) {
        free_constraints_left[k2] = free_constraints[k2] - increment[k2];
      }
      data->poly_opt_.setFreeConstraints(free_constraints_left);
      const double cost_left = data->getCostAndGradientDerivative(NULL, data);

      for (int k2 = 0; k2 < dim; ++k2) {
        free_constraints_right[k2] = free_constraints[k2] + increment[k2];
      }
      data->poly_opt_.setFreeConstraints(free_constraints_right);
      const double cost_right = data->getCostAndGradientDerivative(NULL, data);

      const double grad_k_n = (cost_right - cost_left) / (2.0 * increment_dist);
      gradients_num->at(k)[n] = grad_k_n;
    }
  }

  // Set again the original constraints from before calculating the numerical
  // constraints
  data->poly_opt_.setFreeConstraints(free_constraints);
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientSoftConstraints(
        std::vector<Eigen::VectorXd>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  if (gradients != NULL) {
    const size_t n_free_constraints =
            data->poly_opt_.getNumberFreeConstraints();
    const size_t dim = data->poly_opt_.getDimension();

    gradients->clear();
    gradients->resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

    // Get the current free constraints
    std::vector<Eigen::VectorXd> free_constraints;
    data->poly_opt_.getFreeConstraints(&free_constraints);

    std::vector<Eigen::VectorXd> free_constraints_left, free_constraints_right;
    free_constraints_left.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    free_constraints_right.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    const double increment_dist = data->optimization_parameters_.map_resolution;

    std::vector<Eigen::VectorXd> increment(
            dim, Eigen::VectorXd::Zero(n_free_constraints));
    for (int k = 0; k < dim; ++k) {
      increment.clear();
      increment.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

      for (int n = 0; n < n_free_constraints; ++n) {
        increment[k].setZero();
        increment[k][n] = increment_dist;

        for (int k2 = 0; k2 < dim; ++k2) {
          free_constraints_left[k2] = free_constraints[k2] - increment[k2];
        }
        data->poly_opt_.setFreeConstraints(free_constraints_left);
        const double cost_left =
                data->evaluateMaximumMagnitudeAsSoftConstraint(
                data->inequality_constraints_,
                data->optimization_parameters_.soft_constraint_weight);

        for (int k2 = 0; k2 < dim; ++k2) {
          free_constraints_right[k2] = free_constraints[k2] + increment[k2];
        }
        data->poly_opt_.setFreeConstraints(free_constraints_right);
        const double cost_right =
                data->evaluateMaximumMagnitudeAsSoftConstraint(
                data->inequality_constraints_,
                data->optimization_parameters_.soft_constraint_weight);

        const double grad_k_n = (cost_right - cost_left) / (2.0*increment_dist);
        gradients->at(k)[n] = grad_k_n;
      }
    }

    // Set again the original constraints from before calculating the numerical
    // constraints
    data->poly_opt_.setFreeConstraints(free_constraints);
  }

  double J_sc = data->evaluateMaximumMagnitudeAsSoftConstraint(
          data->inequality_constraints_,
          data->optimization_parameters_.soft_constraint_weight);
  return J_sc;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::
getCostAndGradientSoftConstraintsSimple(
        std::vector<Eigen::VectorXd>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  double J_sc = data->evaluateMaximumMagnitudeAsSoftConstraint(
          data->inequality_constraints_,
          data->optimization_parameters_.soft_constraint_weight);

  if (gradients != NULL) {
    const size_t n_free_constraints =
            data->poly_opt_.getNumberFreeConstraints();
    const size_t dim = data->poly_opt_.getDimension();

    gradients->clear();
    gradients->resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

    // Get the current free constraints
    std::vector<Eigen::VectorXd> free_constraints;
    data->poly_opt_.getFreeConstraints(&free_constraints);

    std::vector<Eigen::VectorXd> free_constraints_right;
    free_constraints_right.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    const double increment_dist = data->optimization_parameters_.map_resolution;

    std::vector<Eigen::VectorXd> increment(
            dim, Eigen::VectorXd::Zero(n_free_constraints));
    for (int k = 0; k < dim; ++k) {
      increment.clear();
      increment.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

      for (int n = 0; n < n_free_constraints; ++n) {
        increment[k].setZero();
        increment[k][n] = increment_dist;

        for (int k2 = 0; k2 < dim; ++k2) {
          free_constraints_right[k2] = free_constraints[k2] + increment[k2];
        }
        data->poly_opt_.setFreeConstraints(free_constraints_right);
        const double cost_right =
                data->evaluateMaximumMagnitudeAsSoftConstraint(
                data->inequality_constraints_,
                data->optimization_parameters_.soft_constraint_weight);

        const double grad_k_n = (cost_right - J_sc) / (increment_dist);
        gradients->at(k)[n] = grad_k_n;
      }
    }

    // Set again the original constraints from before calculating the numerical
    // constraints
    data->poly_opt_.setFreeConstraints(free_constraints);
  }

  return J_sc;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientTime(
        std::vector<double>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  // Weighting terms for different costs
  const double w_d = data->optimization_parameters_.weights.w_d;
  const double w_c = data->optimization_parameters_.weights.w_c;
  const double w_t = data->optimization_parameters_.weights.w_t;
  const double w_sc = data->optimization_parameters_.weights.w_sc;

  // Retrieve the current segment times
  std::vector<double> segment_times;
  data->poly_opt_.getSegmentTimes(&segment_times);

  if (gradients != NULL) {
    const size_t n_segments = data->poly_opt_.getNumberSegments();
    const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
    const size_t dim = data->poly_opt_.getDimension();

    gradients->clear();
    gradients->resize(n_segments);

    // Initialize changed segment times for numerical derivative
    std::vector<double> segment_times_smaller(n_segments);
    std::vector<double> segment_times_bigger(n_segments);
    const double increment_time = data->optimization_parameters_.increment_time;
    for (int n = 0; n < n_segments; ++n) {
      // Calculate cost with lower segment time
      segment_times_smaller = segment_times;
      // TODO: check if segment times are bigger than 0.1; else ?
      segment_times_smaller[n] = segment_times_smaller[n] <= 0.1 ?
                                0.1 : segment_times_smaller[n] - increment_time;

      // Update the segment times. This changes the polynomial coefficients.
      data->poly_opt_.updateSegmentTimes(segment_times_smaller);

      // Calculate cost and gradient with new segment time
      bool is_collision_smaller;
      const double J_d_smaller = data->getCostAndGradientDerivative(NULL, data);
      const double J_c_smaller = data->getCostAndGradientCollision(
              NULL, data, &is_collision_smaller);
      double J_sc_smaller = 0.0;
      if (data->optimization_parameters_.use_soft_constraints) {
        J_sc_smaller = data->getCostAndGradientSoftConstraints(NULL, data);
      }

      // Now the same with an increased segment time
      // Calculate cost with higher segment time
      segment_times_bigger = segment_times;
      // TODO: check if segment times are bigger than 0.1; else ?
      segment_times_bigger[n] = segment_times_bigger[n] <= 0.1 ?
                                0.1 : segment_times_bigger[n] + increment_time;

      // Update the segment times. This changes the polynomial coefficients.
      data->poly_opt_.updateSegmentTimes(segment_times_bigger);

      // Calculate cost and gradient with new segment time
      bool is_collision_bigger;
      const double J_d_bigger = data->getCostAndGradientDerivative(NULL, data);
      const double J_c_bigger = data->getCostAndGradientCollision(
              NULL, data, &is_collision_bigger);
      double J_sc_bigger = 0.0;
      if (data->optimization_parameters_.use_soft_constraints) {
        J_sc_bigger = data->getCostAndGradientSoftConstraints(NULL, data);
      }

      const double dJd_dt = (J_d_bigger - J_d_smaller) / (2.0 * increment_time);
      const double dJc_dt = (J_c_bigger - J_c_smaller) / (2.0 * increment_time);
      const double dJsc_dt = (J_sc_bigger - J_sc_smaller) /(2.0*increment_time);
      const double dJt_dt = 1.0; // J_t = t --> dJt_dt = 1.0 for all tm

      // Calculate the gradient
      gradients->at(n) = w_d*dJd_dt + w_c*dJc_dt + w_sc*dJsc_dt + w_t*dJt_dt;
    }

    // Set again the original segment times from before calculating the
    // numerical gradient
    data->poly_opt_.updateSegmentTimes(segment_times);
  }

  // Compute cost without gradient (only time)
  double total_time = computeTotalTrajectoryTime(segment_times);
  double J_t = total_time;  // TODO: squared? if yes, adjust dJt_dt = 2t

  return J_t;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientTimeSimple(
        std::vector<double>* gradients, void* opt_data,
        double J_d, double J_c, double J_sc) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  // Weighting terms for different costs
  const double w_d = data->optimization_parameters_.weights.w_d;
  const double w_c = data->optimization_parameters_.weights.w_c;
  const double w_t = data->optimization_parameters_.weights.w_t;
  const double w_sc = data->optimization_parameters_.weights.w_sc;

  // Retrieve the current segment times
  std::vector<double> segment_times;
  data->poly_opt_.getSegmentTimes(&segment_times);

  if (gradients != NULL) {
    const size_t n_segments = data->poly_opt_.getNumberSegments();
    const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
    const size_t dim = data->poly_opt_.getDimension();

    gradients->clear();
    gradients->resize(n_segments);

    // Initialize changed segment times for numerical derivative
    std::vector<double> segment_times_bigger(n_segments);
    const double increment_time = data->optimization_parameters_.increment_time;
    for (int n = 0; n < n_segments; ++n) {

      // Now the same with an increased segment time
      // Calculate cost with higher segment time
      segment_times_bigger = segment_times;
      // TODO: check if segment times are bigger than 0.1; else ?
      segment_times_bigger[n] = segment_times_bigger[n] <= 0.1 ?
                                0.1 : segment_times_bigger[n] + increment_time;

      // Update the segment times. This changes the polynomial coefficients.
      data->poly_opt_.updateSegmentTimes(segment_times_bigger);

      // Calculate cost and gradient with new segment time
      bool is_collision_bigger;
      const double J_d_bigger = data->getCostAndGradientDerivative(NULL, data);
      const double J_c_bigger = data->getCostAndGradientCollision(
              NULL, data, &is_collision_bigger);
      double J_sc_bigger = 0.0;
      if (data->optimization_parameters_.use_soft_constraints) {
        J_sc_bigger = data->getCostAndGradientSoftConstraints(NULL, data);
      }

      const double dJd_dt = (J_d_bigger-J_d) / (increment_time);
      const double dJc_dt = (J_c_bigger-J_c) / (increment_time);
      const double dJsc_dt = (J_sc_bigger-J_sc) /(increment_time);
      const double dJt_dt = 1.0; // J_t = t --> dJt_dt = 1.0 for all tm

      // Calculate the gradient
      gradients->at(n) = w_d*dJd_dt + w_c*dJc_dt + w_sc*dJsc_dt + w_t*dJt_dt;
    }

    // Set again the original segment times from before calculating the
    // numerical gradient
    data->poly_opt_.updateSegmentTimes(segment_times);
  }

  // Compute cost without gradient (only time)
  double total_time = computeTotalTrajectoryTime(segment_times);
  double J_t = total_time;  // TODO: squared? if yes, adjust dJt_dt = 2t

  return J_t;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostPotential(
        double collision_distance, bool* is_collision) {

  // Get parameter
  const double epsilon = optimization_parameters_.epsilon;
  const double robot_radius = optimization_parameters_.robot_radius;
  const double collision_potential_multiplier =
          optimization_parameters_.coll_pot_multiplier;

  *is_collision = false;
  double cost = 0.0;
  collision_distance -= robot_radius;
  if (collision_distance <= 0.0) {
//    cost = -collision_distance + 0.5 * epsilon;
    cost = collision_potential_multiplier*(-collision_distance) + 0.5 * epsilon;
    *is_collision = true;
  } else if (collision_distance <= epsilon) {
    double epsilon_distance = collision_distance - epsilon;
    cost = 0.5 * 1.0 / epsilon * epsilon_distance * epsilon_distance;
  } else { // TODO: WHAT IF DIST IS INF/not in map
    cost = 0.0;
  }

  return cost;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::evaluateMaximumMagnitudeConstraint(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  ConstraintData* constraint_data =
      static_cast<ConstraintData*>(data);  // wheee ...
  PolynomialOptimizationNonLinear<N>* optimization_data =
      constraint_data->this_object;

  Extremum max;
  // for now, let's assume that the optimization has been done
  switch (constraint_data->derivative) {
    case derivative_order::POSITION:
      max = optimization_data->poly_opt_
                .template computeMaximumOfMagnitude<derivative_order::POSITION>(
                    nullptr);
      break;
    case derivative_order::VELOCITY:
      max = optimization_data->poly_opt_
                .template computeMaximumOfMagnitude<derivative_order::VELOCITY>(
                    nullptr);
      break;
    case derivative_order::ACCELERATION:
      max = optimization_data->poly_opt_.template computeMaximumOfMagnitude<
          derivative_order::ACCELERATION>(nullptr);
      break;
    case derivative_order::JERK:
      max = optimization_data->poly_opt_
                .template computeMaximumOfMagnitude<derivative_order::JERK>(
                    nullptr);
      break;
    case derivative_order::SNAP:
      max = optimization_data->poly_opt_
                .template computeMaximumOfMagnitude<derivative_order::SNAP>(
                    nullptr);
      break;
    default:
      LOG(WARNING) << "[Nonlinear inequality constraint evaluation]: no "
                      "implementation for derivative: "
                   << constraint_data->derivative;
      return 0;
  }

  optimization_data->optimization_info_.maxima[constraint_data->derivative] =
      max;

  return max.value - constraint_data->value;
}

template <int _N>
double
PolynomialOptimizationNonLinear<_N>::evaluateMaximumMagnitudeAsSoftConstraint(
    const std::vector<std::shared_ptr<ConstraintData> >& inequality_constraints,
    double weight, double maximum_cost) const {
  std::vector<double> dummy;
  double cost = 0;

//  if (optimization_parameters_.print_debug_info) {
//    std::cout << "  soft_constraints: " << std::endl;
//  }

  for (std::shared_ptr<const ConstraintData> constraint :
       inequality_constraints_) {
    // need to call the c-style callback function here, thus the ugly cast to
    // void*.
    double abs_violation = evaluateMaximumMagnitudeConstraint(
        dummy, dummy, (void*)constraint.get());

    double relative_violation = abs_violation / constraint->value;
    const double current_cost =
        std::min(maximum_cost, exp(relative_violation * weight));
    cost += current_cost;
//    if (optimization_parameters_.print_debug_info) {
//      std::cout << "    derivative " << constraint->derivative
//                << " abs violation: " << abs_violation
//                << " : relative violation: " << relative_violation
//                << " cost: " << current_cost << std::endl;
//    }
  }
  return cost;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::computeTotalTrajectoryTime(
    const std::vector<double>& segment_times) {
  double total_time = 0;
  for (double t : segment_times) total_time += t;
  return total_time;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::lerp(
        double x, double x1, double x2, double q00, double q01) {
  return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::biLerp(
        double x, double y, double q11, double q12, double q21, double q22,
        double x1, double x2, double y1, double y2) {
  double r1 = lerp(x, x1, x2, q11, q21);
  double r2 = lerp(x, x1, x2, q12, q22);

  return lerp(y, y1, y2, r1, r2);
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::triLerp(
        double x, double y, double z, double q000, double q001, double q010,
        double q011, double q100, double q101, double q110, double q111,
        double x1, double x2, double y1, double y2, double z1, double z2) {
  double x00 = lerp(x, x1, x2, q000, q100);
  double x10 = lerp(x, x1, x2, q010, q110);
  double x01 = lerp(x, x1, x2, q001, q101);
  double x11 = lerp(x, x1, x2, q011, q111);
  double r0 = lerp(y, y1, y2, x00, x01);
  double r1 = lerp(y, y1, y2, x10, x11);

  return lerp(z, z1, z2, r0, r1);
}


template <int _N>
void PolynomialOptimizationNonLinear<_N>::
calculatePolynomialDerivativeMappingMatrices() {
  // Calculate matrix for mapping vector of polynomial coefficients of a
  // function to the polynomial coefficients of its derivative.
  // [0 1 0 0 0 ...]              f_k(t) = a0 + a1*t + a2*t^2 + a3*t^3 + ...
  // [0 0 2 0 0 ...]          df_k(t)/dt =      a1   + 2*a2*t + 3*a3*t^2 + ...
  // [0 0 0 3 0 ...]                    with T = [t^0 t^1 t^2 t^3 t^4 ...]
  // [0 0 0 0 4 ...]            -->     f_k(t) = T * p_k
  // [  ...   ...  ]            --> df_k(t)/dt = T * V * p_k
  size_t n_segments = poly_opt_.getNumberSegments();

  V_all_segments_.resize(n_segments * N, n_segments * N);
  V_.resize(N, N);

  V_all_segments_.setZero();
  V_.setZero();
  for (int i = 0; i < V_all_segments_.diagonal(1).size(); ++i) {
    V_all_segments_.diagonal(1)(i) = (i + 1) % N;
  }
  V_ = V_all_segments_.block(0, 0, N, N);

  Acc_.resize(N, N);
  Acc_.setZero();
  for (int i = 0; i < Acc_.diagonal(2).size(); ++i) {
    Acc_.diagonal(2)(i) = (i + 1)*(i + 2);
  }

  Jerk_.resize(N, N);
  Jerk_.setZero();
  for (int i = 0; i < Jerk_.diagonal(3).size(); ++i) {
    Jerk_.diagonal(3)(i) = (i + 1)*(i + 2)*(i + 3);
  }

  Snap_.resize(N, N);
  Snap_.setZero();
  for (int i = 0; i < Snap_.diagonal(4).size(); ++i) {
    Snap_.diagonal(4)(i) = (i + 1)*(i + 2)*(i + 3)*(i + 4);
  }

  Snap_all_segments_.resize(n_segments * N, n_segments * N);
  Snap_all_segments_.setZero();
  for (int n = 0; n < n_segments; ++n) {
    for (int i = 0; i < Snap_.diagonal(4).size(); ++i) {
      const int idx = n * N + i;
      Snap_all_segments_.diagonal(4)(idx) = (i + 1)*(i + 2)*(i + 3)*(i + 4);
    }
  }
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::
setFreeEndpointDerivativeHardConstraints(
        const std::vector<double>& initial_solution,
        std::vector<double>* lower_bounds, std::vector<double>* upper_bounds) {

  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();

  LOG(INFO) << "USE HARD CONSTRAINTS FOR ENDPOINT DERIVATIVE BOUNDARIES";

  // Set all values to -inf/inf and reset only bounded opti param with values
  for (double x : initial_solution) {
    lower_bounds->push_back(-HUGE_VAL);
    upper_bounds->push_back(HUGE_VAL);
  }

  // Add hard constraints with lower and upper bounds for opti parameters
  const size_t n_segments = poly_opt_.getNumberSegments();
  for (int k = 0; k < dimension_; ++k) {
    for (int n = 0; n < n_segments - 1; ++n) {
      unsigned int start_idx = 0;
      if (optimization_parameters_.solve_with_position_constraint) {
        start_idx = k*n_free_constraints + n*derivative_to_optimize_;
      } else {
        // Add position constraints given through the map boundaries
        start_idx = k*n_free_constraints + n*(derivative_to_optimize_ + 1);

        lower_bounds->at(start_idx) = optimization_parameters_.min_bound[k];
        upper_bounds->at(start_idx) = optimization_parameters_.max_bound[k];
      }

      // Add higher order derivative constraints (v_max and a_max)
      for (const auto& constraint_data : inequality_constraints_) {
        unsigned int deriv_idx = 0;
        if (optimization_parameters_.solve_with_position_constraint) {
          deriv_idx = constraint_data->derivative - 1;
        } else {
          deriv_idx = constraint_data->derivative;
        }

        lower_bounds->at(start_idx + deriv_idx) = -std::abs(
                constraint_data->value);
        upper_bounds->at(start_idx + deriv_idx) = std::abs(
                constraint_data->value);
      }
    }
  }
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::printMatlabSampledTrajectory(
        const std::string& file) const {

  const int n_fixed = poly_opt_.getNumberFixedConstraints();
  const int n_free = poly_opt_.getNumberFreeConstraints();

  // Allocate some size of p vector.
  double dt = 0.01; // TODO: parameterize

  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);

  int num_segments = poly_opt_.getNumberSegments();
  int total_samples = 0;
  for (int i = 0; i < num_segments; ++i) {
    total_samples += static_cast<int>(std::ceil(segment_times[i] / dt)) + 1;
  }

  std::vector<Eigen::VectorXd> d_p_vec;
  std::vector<Eigen::VectorXd> d_f_vec;
  poly_opt_.getFreeConstraints(&d_p_vec);
  poly_opt_.getFixedConstraints(&d_f_vec);

  std::vector<Eigen::VectorXd> p_vec(dimension_,
                                     Eigen::VectorXd::Zero(N * num_segments));
  for (int k = 0; k < dimension_; ++k) {
    Eigen::VectorXd d_all(n_fixed + n_free);
    d_all.head(n_fixed) = d_f_vec[k];
    d_all.tail(n_free) = d_p_vec[k];

    // Get the coefficients out.
    // L is shorthand for A_inv * M.
    p_vec[k] = L_ * d_all;
  }

  // Layout: [t, x, y, z, vx, vy, vz, jx, jy, jz, sx, sy, sz, tm]
  Eigen::MatrixXd output(total_samples, 5*dimension_ + 2);
  output.setZero();
  int j = 0;

  double t = 0.0;
  double current_segment_time = 0.0;
  for (int i = 0; i < num_segments; ++i) {
    // Select a time.
    for (t = 0.0; t < segment_times[i]; t += dt) {
      // T is the vector for just THIS SEGMENT.
      Eigen::VectorXd T_seg; // Is supposed to be a column-vector
      T_seg.resize(N);
      T_seg.setZero();
      for (int n = 0; n < N; ++n) {
        T_seg[n] = pow(t, n);
      }

      // Calculate the position per axis. Also calculate velocity so we don't
      // have to get p_k_i out again.
      Eigen::VectorXd position(dimension_);
      Eigen::VectorXd velocity(dimension_);
      Eigen::VectorXd acceleration(dimension_);
      Eigen::VectorXd jerk(dimension_);
      Eigen::VectorXd snap(dimension_);
      position.setZero();
      velocity.setZero();
      acceleration.setZero();
      jerk.setZero();
      snap.setZero();
      for (int k = 0; k < dimension_; ++k) {
        // Get the coefficients just for this segment.
        Eigen::Block<Eigen::VectorXd> p_k_i = p_vec[k].block(i * N, 0, N, 1);
        position(k) = (T_seg.transpose() * p_k_i)(0);
        velocity(k) = (T_seg.transpose() * V_ * p_k_i)(0);
        acceleration(k) = (T_seg.transpose() * Acc_ * p_k_i)(0);
        jerk(k) = (T_seg.transpose() * Jerk_ * p_k_i)(0);
        snap(k) = (T_seg.transpose() * Snap_ * p_k_i)(0);
      }
      if (j < output.rows()) {
        output(j, 0) = t + current_segment_time;
        output.row(j).segment(1, dimension_) = position.transpose();
        output.row(j).segment(1+dimension_, dimension_) = velocity.transpose();
        output.row(j).segment(1+2*dimension_, dimension_) =
                acceleration.transpose();
        output.row(j).segment(1+3*dimension_, dimension_) =
                jerk.transpose();
        output.row(j).segment(1+4*dimension_, dimension_) =
                snap.transpose();
        j++;
      }
    }
    current_segment_time += segment_times[i];
    output(i, 1+5*dimension_) = current_segment_time;
  }

  std::fstream fs;
  fs.open(file, std::fstream::out);
  fs << output;
  fs.close();
}

}  // namespace mav_trajectory_generation

namespace nlopt {

inline std::string returnValueToString(int return_value) {
  switch (return_value) {
    case nlopt::SUCCESS:
      return std::string("SUCCESS");
    case nlopt::FAILURE:
      return std::string("FAILURE");
    case nlopt::INVALID_ARGS:
      return std::string("INVALID_ARGS");
    case nlopt::OUT_OF_MEMORY:
      return std::string("OUT_OF_MEMORY");
    case nlopt::ROUNDOFF_LIMITED:
      return std::string("ROUNDOFF_LIMITED");
    case nlopt::FORCED_STOP:
      return std::string("FORCED_STOP");
    case nlopt::STOPVAL_REACHED:
      return std::string("STOPVAL_REACHED");
    case nlopt::FTOL_REACHED:
      return std::string("FTOL_REACHED");
    case nlopt::XTOL_REACHED:
      return std::string("XTOL_REACHED");
    case nlopt::MAXEVAL_REACHED:
      return std::string("MAXEVAL_REACHED");
    case nlopt::MAXTIME_REACHED:
      return std::string("MAXTIME_REACHED");
    default:
      return std::string("ERROR CODE UNKNOWN");
  }
}
}  // namespace nlopt

#endif  // MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_
