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
      derivative_to_optimize_(derivative_order::INVALID),
      optimization_parameters_(parameters),
      solve_with_position_constraint_(false) {}

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


  // Parameters before removing constraints
  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = poly_opt_.getNumberFixedConstraints();
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

  // TODO: needed? find runtime error
//  // Add inequality constraints again after reset nlopt_ --> Hard constraint
//  if (!optimization_parameters_.use_soft_constraints) {
//    for (const auto& constraint_data : inequality_constraints_) {
//      try {
//        nlopt_->add_inequality_constraint(
//                &PolynomialOptimizationNonLinear<
//                        N>::evaluateMaximumMagnitudeConstraint,
//                constraint_data.get(),
//                optimization_parameters_.inequality_constraint_tolerance);
//      } catch (std::exception& e) {
//        LOG(ERROR) << "ERROR while setting inequality constraint " << e.what()
//                   << std::endl;
//        return false;
//      }
//    }
//  }

  // Parameters after removing constraints // TODO: test if needed and true
  const size_t n_free_constraints_after = poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints_after = poly_opt_.getNumberFixedConstraints();
  // TODO: move to linear solver. add method setFreeConstraintsFromCoefficients
  // 5) Get your new mapping matrix L (p = L*[d_f d_P]^T = A^(-1)*M*[d_f d_P]^T)
  // Fixed constraints are the same except plus the position constraints we
  // removed. Add those removed position constraints to free derivatives.
  Eigen::MatrixXd M, M_pinv, A, A_inv;
  poly_opt_.getA(&A);
  poly_opt_.getMpinv(&M_pinv);

  const int n_constraints_per_vertex = N / 2; // Num constraints per vertex
  // 6) Calculate your reordered endpoint-derivatives. d_all = L^(-1) * p_k
  // where p_k are the old coefficients from the original linear solution and
  // L the new remapping matrix
  // d_all has the same size before and after removing constraints
  Eigen::VectorXd d_all(n_fixed_constraints + n_free_constraints);
  std::vector<Eigen::VectorXd> d_p(dim, Eigen::VectorXd(n_free_constraints_after));
  for (int i = 0; i < dim; ++i) {
    d_all = M_pinv * A * p[i]; // Old coeff p, but new ordering M_pinv * A
    d_p[i] = d_all.tail(n_free_constraints_after);
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

  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  initial_step.reserve(n_segments);
  for (double t : segment_times) {
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel * t);
  }

  for (double t : segment_times) {
    upper_bounds.push_back(t * 2.0);
  }

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
  poly_opt_.solveLinear();
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
    timing::Timer timer_solve("optimize_nonlinear_full_total_time");
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
  solve_with_position_constraint_ = false;
  if (solve_with_position_constraint_) {
    poly_opt_.solveLinear();
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

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  double multiplier = optimization_parameters_.state_bound_multiplicator;
  if (optimization_parameters_.set_bounds_with_constraints) {
    LOG(INFO) << "USE HARD CONSTRAINTS FOR ENDPOINT DERIVATIVE BOUNDARIES";

//    lower_bounds = std::vector<double>(n_optmization_variables, -HUGE_VAL);
//    upper_bounds = std::vector<double>(n_optmization_variables, HUGE_VAL);

    for (double x : initial_solution) {
      const double abs_x = std::abs(x);
      lower_bounds.push_back(-abs_x * multiplier);
      upper_bounds.push_back(abs_x * multiplier);
    }

    // Add hard constraints with lower and upper bounds for opti parameters
    for (int k = 0; k < dimension_; ++k) {
      for (int n = 0; n < n_segments - 1; ++n) {
        // Add position constraints given through the map boundaries
        const unsigned int start_idx = k * free_constraints.front().size() +
                                       n * (derivative_to_optimize_ + 1);
        lower_bounds[start_idx] = optimization_parameters_.min_bound[k];
        upper_bounds[start_idx] = optimization_parameters_.max_bound[k];

        // Add higher order derivative constraints (v_max and a_max)
        for (const auto& constraint_data : inequality_constraints_) {
          const unsigned int deriv_idx = constraint_data->derivative;
          lower_bounds[start_idx + deriv_idx] = -std::abs(
                  constraint_data->value);
          upper_bounds[start_idx + deriv_idx] = std::abs(
                  constraint_data->value);
        }
      }
    }
  } else {
    LOG(INFO) << "USE MULTIPLIER FOR ALL BOUNDS";

    for (double x : initial_solution) {
      const double abs_x = std::abs(x);
      lower_bounds.push_back(-abs_x * multiplier);
      upper_bounds.push_back(abs_x * multiplier);
    }
  }

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
                    N>::objectiveFunctionFreeConstraintsAndCollision,
            this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nlin_free_constraints_and_collision");
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

  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  // compute initial solution
  poly_opt_.solveLinear();
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
    timing::Timer timer_solve("optimize_nonlinear_full_total_time");
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
  solve_with_position_constraint_ = false;
  if (solve_with_position_constraint_) {
    poly_opt_.solveLinear();
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
  // [0 1 0 0 0 ...]              f_k(t) = a0 + a1*t + a2*t^2 + a3*t^3 + ...
  // [0 0 2 0 0 ...]          df_k(t)/dt =      a1   + 2*a2*t + 3*a3*t^2 + ...
  // [0 0 0 3 0 ...]                    with T = [t^0 t^1 t^2 t^3 t^4 ...]
  // [0 0 0 0 4 ...]            -->     f_k(t) = T * p_k
  // [  ...   ...  ]            --> df_k(t)/dt = T * V * p_k
//  size_t n_segments = poly_opt_.getNumberSegments();

  V_all_segments_.resize(n_segments * N, n_segments * N);
  V_.resize(N, N);

  V_all_segments_.setZero();
  V_.setZero();
  for (int i = 0; i < V_all_segments_.diagonal(1).size(); ++i) {
    V_all_segments_.diagonal(1)(i) = (i + 1) % N;
  }
  V_ = V_all_segments_.block(0, 0, N, N);

  // copy all constraints into one vector:
  for (double t : segment_times) {
    initial_solution.push_back(t);
  }

  for (const Eigen::VectorXd& c : free_constraints) {
    for (int i = 0; i < c.size(); ++i) {
      initial_solution.push_back(c[i]);
    }
  }

  double multiplier = optimization_parameters_.state_bound_multiplicator;
  if (optimization_parameters_.set_bounds_with_constraints) {
    LOG(INFO) << "USE HARD CONSTRAINTS FOR ENDPOINT DERIVATIVE BOUNDARIES";

//    lower_bounds = std::vector<double>(n_optmization_variables, -HUGE_VAL);
//    upper_bounds = std::vector<double>(n_optmization_variables, HUGE_VAL);

    for (double x : initial_solution) {
      const double abs_x = std::abs(x);
      lower_bounds.push_back(-abs_x * multiplier);
      upper_bounds.push_back(abs_x * multiplier);
    }

    // Add hard constraints with lower and upper bounds for opti parameters
    for (int k = 0; k < dimension_; ++k) {
      for (int n = 0; n < n_segments - 1; ++n) {
        // Add position constraints given through the map boundaries
        const unsigned int start_idx =
                n_segments + k * free_constraints.front().size() +
                        n * (derivative_to_optimize_ + 1);
        lower_bounds[start_idx] = optimization_parameters_.min_bound[k];
        upper_bounds[start_idx] = optimization_parameters_.max_bound[k];

        // Add higher order derivative constraints (v_max and a_max)
        for (const auto& constraint_data : inequality_constraints_) {
          const unsigned int deriv_idx = constraint_data->derivative;
          lower_bounds[start_idx + deriv_idx] = -std::abs(
                  constraint_data->value);
          upper_bounds[start_idx + deriv_idx] = std::abs(
                  constraint_data->value);
        }
      }
    }
  } else {
    LOG(INFO) << "USE MULTIPLIER FOR ALL BOUNDS";

    for (double x : initial_solution) {
      const double abs_x = std::abs(x);
      lower_bounds.push_back(-abs_x * multiplier);
      upper_bounds.push_back(abs_x * multiplier);
    }
  }

  for (size_t i = 0; i < n_segments; ++i) {
    lower_bounds[i] = 0.1; // TODO: needed? if yes parameterize
  }

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
    timing::Timer timer_solve("optimize_nonlinear_full_total_time");
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
  double cost_trajectory = optimization_data->poly_opt_.computeCost();
  double cost_time = 0;
  double cost_constraints = 0;
  const double total_time = computeTotalTrajectoryTime(segment_times);
  cost_time = total_time * total_time *
              optimization_data->optimization_parameters_.time_penalty;

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
  }

  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
        optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
            optimization_data->inequality_constraints_,
            optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
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

  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  std::vector<double> segment_times;
  segment_times.reserve(n_segments);

  for (size_t i = 0; i < n_segments; ++i) segment_times.push_back(x[i]);

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

  double cost_trajectory = optimization_data->poly_opt_.computeCost();
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

  std::vector<Eigen::VectorXd> grad_d;
  double J_d = 0.0;
  if (!gradient.empty()) {
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
  } else {
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
  }

  // TODO: get rid after testing
  double cost_trajectory2 = optimization_data->poly_opt_.computeCost();
  double cost_trajectory = J_d;
  double cost_time = 0.0;
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
    std::cout << "  computeCost(): " << cost_trajectory2 << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_time + cost_constraints
              << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_time = cost_time;
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

  return cost_trajectory + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionFreeConstraintsAndCollision(
        const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_free_constraints =
          optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_free_constraints * dim);

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

  std::vector<Eigen::VectorXd> grad_d, grad_c, grad_sc;
  double J_d = 0.0;
  double J_c = 0.0;
  double J_sc = 0.0;
  if (!gradient.empty()) {
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            &grad_c, optimization_data);
    if (optimization_data->optimization_parameters_.use_soft_constraints) {
      J_sc = optimization_data->getCostAndGradientSoftConstraints(
              &grad_sc, optimization_data);
    } else { // If not used, resize and set everything to zero
      grad_sc.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    }
  } else {
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            NULL, optimization_data);
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
  double w_d = optimization_data->optimization_parameters_.weights.w_d;
  double w_c = optimization_data->optimization_parameters_.weights.w_c;
  double w_sc = optimization_data->optimization_parameters_.weights.w_sc;

  // Get the weighted cost
  double cost_trajectory = w_d * J_d;
  double cost_collision = w_c * J_c;
  double cost_constraints = w_sc * J_sc;

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  collision: " << cost_collision << std::endl;
    std::cout << "  constraints: " << cost_constraints << std::endl;
    std::cout << "  sum: " << cost_trajectory + cost_collision +
            cost_constraints << std::endl;
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

  // TODO: Clean not needed terms (ie cost_time) everywhere
  return cost_trajectory + cost_collision + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N
>::objectiveFunctionFreeConstraintsAndCollisionAndTime(
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

  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  std::vector<double> segment_times;
  segment_times.reserve(n_segments);

  for (size_t i = 0; i < n_segments; ++i) segment_times.push_back(x[i]);

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

//  std::cout << "4 FREE CONSTRAINTS" << std::endl;
//  for (int i = 0; i < free_constraints[0].size(); ++i) {
//    std::cout << i << ": " << free_constraints[0][i] << " | "
//              << free_constraints[1][i] << " | "
//              << free_constraints[2][i]
//              << std::endl;
//  }
//  std::cout << std::endl;

  // TODO: calculate grad_t
  std::vector<Eigen::VectorXd> grad_d, grad_c, grad_t;
  double J_d = 0.0;
  double J_c = 0.0;
  double J_t = 0.0;
  if (!gradient.empty()) {
    J_d = optimization_data->getCostAndGradientDerivative(
            &grad_d, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            &grad_c, optimization_data);
  } else {
    J_d = optimization_data->getCostAndGradientDerivative(
            NULL, optimization_data);
    J_c = optimization_data->getCostAndGradientCollision(
            NULL, optimization_data);
  }
  const double total_time = computeTotalTrajectoryTime(segment_times);
  J_t = total_time * total_time *
          optimization_data->optimization_parameters_.time_penalty;

  // TODO: add numerical calculation option here

  // TODO: Include soft constraint cost here?
  double J_sc = 0.0;
  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    J_sc = optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
            optimization_data->inequality_constraints_,
            optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  // Weighting terms for different costs
  double w_d = optimization_data->optimization_parameters_.weights.w_d;
  double w_c = optimization_data->optimization_parameters_.weights.w_c;
  double w_sc = optimization_data->optimization_parameters_.weights.w_sc;
  double w_t = optimization_data->optimization_parameters_.weights.w_t;

  // Get the weighted cost
  double cost_trajectory = w_d * J_d;
  double cost_collision = w_c * J_c;
  double cost_time = w_t * J_t;
  double cost_constraints = w_sc * J_sc;

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
  }

  // Save the trajectory of this iteration
  Trajectory trajectory_i;
  optimization_data->getTrajectory(&trajectory_i);
  optimization_data->all_trajectories_.push_back(trajectory_i);

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_collision = cost_collision;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
          cost_constraints;

  if (!gradient.empty()) {
    gradient.clear();
    gradient.resize(3*n_free_constraints);

    for (int i = 0; i < n_free_constraints; ++i) {
      gradient[0 * n_free_constraints + i] =
              w_d * grad_d[0][i] + w_c * grad_c[0][i] + w_t * grad_t[0][i];
      gradient[1 * n_free_constraints + i] =
              w_d * grad_d[1][i] + w_c * grad_c[1][i] + w_t * grad_t[1][i];
      gradient[2 * n_free_constraints + i] =
              w_d * grad_d[2][i] + w_c * grad_c[2][i] + w_t * grad_t[2][i];
    }
  }

  // TODO: Clean not needed terms (ie cost_time) everywhere
  return cost_trajectory + cost_collision + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientDerivative(
        std::vector<Eigen::VectorXd>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  Eigen::MatrixXd R;
  data->poly_opt_.getR(&R);

  // Compare the two approaches:
  // getCost() and the full matrix.
//  const size_t n_segments = data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = data->poly_opt_.getNumberFixedConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  double J_d = 0.0;
  std::vector<Eigen::VectorXd> grad_d(dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Set up mappings to R_FF R_FP R_PP etc. R_FP' = R_PF if that saves
  // time eventually.
  // All of these are the same per axis.
  // Not sure if there's a boost from the sparse solver? Our problems are tiny
  // so I guess not.

  // R_ff * d_f is actually constant so can cache this term.
  Eigen::Block<Eigen::MatrixXd> R_ff =
          R.block(0, 0, n_fixed_constraints, n_fixed_constraints);

  Eigen::Block<Eigen::MatrixXd> R_pf =
          R.block(n_fixed_constraints, 0, n_free_constraints,
                   n_fixed_constraints);

  Eigen::Block<Eigen::MatrixXd> R_pp =
          R.block(n_fixed_constraints, n_fixed_constraints, n_free_constraints,
                   n_free_constraints);

  // Get d_p and d_f vector for all axes.
  std::vector<Eigen::VectorXd> d_p_vec;
  std::vector<Eigen::VectorXd> d_f_vec;

  // Figure out if we should have polyopt keep track of d_ps
  // or us keep track of d_ps over iterations.
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
        std::vector<Eigen::VectorXd>* gradients, void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  // Compare the two approaches:
  // getCost() and the full matrix.
  const size_t n_segments = data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints = data->poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = data->poly_opt_.getNumberFixedConstraints();
  const size_t dim = data->poly_opt_.getDimension();

  double J_c = 0.0;
  std::vector<Eigen::VectorXd> grad_c(dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Get d_p and d_f vector for all axes.
  // Figure out if we should have polyopt keep track of d_ps
  // or us keep track of d_ps over iterations.
  std::vector<Eigen::VectorXd> d_p_vec, d_f_vec;
  data->poly_opt_.getFreeConstraints(&d_p_vec);
  data->poly_opt_.getFixedConstraints(&d_f_vec);

  // 1) Get coefficients
  std::vector<Eigen::VectorXd> p_all_segments(dim, Eigen::VectorXd(N * n_segments));
  for (int k = 0; k < dim; ++k) {
    Eigen::VectorXd d_all_segments(n_fixed_constraints + n_free_constraints);
    d_all_segments.head(n_fixed_constraints) = d_f_vec[k];

    switch (data->optimization_parameters_.objective) {
      case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollision:
        d_all_segments.tail(n_free_constraints) = d_p_vec[k];
        break;
      case NonlinearOptimizationParameters::OptimizationObjective::kOptimizeFreeConstraintsAndCollisionAndTime:
        d_all_segments.tail(n_free_constraints) =
                d_p_vec[k].tail(n_free_constraints);
        break;
      default:
        LOG(ERROR) << "Unknown Optimization Objective. Abort.";
        break;
    }

    // The coefficients for each axis k with size (N * num_segments) x 1
    p_all_segments[k] = data->L_ * d_all_segments;
  }

  std::vector<double> segment_times;
  data->poly_opt_.getSegmentTimes(&segment_times);

  // Get the correct L block to calculate derivatives.
  Eigen::Block<Eigen::MatrixXd> L_pp =
          data->L_.block(0, n_fixed_constraints,
                         data->L_.rows(), n_free_constraints);

  double dt = 0.1; // TODO: parameterize
  double dist_sum_limit = 0.05; // TODO: parameterize map resolution

  Eigen::VectorXd prev_pos(dim);
  prev_pos.setZero();
  // sum --> numerical integral
  double time_sum = -1;
  double dist_sum = 0;
  double t = 0.0;
  for (int i = 0; i < n_segments; ++i) {
    for (t = 0.0; t < segment_times[i]; t += dt) {

      // 2) Calculate the T vector (see paper equation (8)) for each segment
      Eigen::VectorXd T; // Is supposed to be a column-vector
      T.resize(N);
      for (int n = 0; n < N; ++n) {
        T[n] = pow(t, n);
      }

      // Create T for all segemtnts
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

      // Numerical integration
      if (time_sum < 0) {
        // Skip first entry
        time_sum = 0.0;
        prev_pos = pos;
        continue;
      }

      time_sum += dt;
      dist_sum += (pos - prev_pos).norm();
      prev_pos = pos;

      if (dist_sum < dist_sum_limit) { continue; }

      // Cost and gradient of potential map from esdf
      Eigen::VectorXd grad_c_d_f(dim); // dc/dd_f
      double c = 0.0;
      if (gradients != NULL) {
        c = getCostAndGradientPotentialESDF(pos, &grad_c_d_f, data);
      } else {
        c = getCostAndGradientPotentialESDF(pos, NULL, data);
      }

      // Cost per segment and time sample
      double J_c_i_t = c * vel.norm() * time_sum;
      J_c += J_c_i_t;

      if (gradients != NULL) {
        // Norm has to be non-zero
        if (vel.norm() > 1e-6) {
          // Calculate gradient per axis
          for (int k = 0; k < dim; ++k) {
            // See paper equation (14)
            Eigen::VectorXd grad_c_k =
                    (vel.norm() * time_sum * grad_c_d_f(k) * T_all_seg.transpose() * L_pp +
                            time_sum * c * vel(k) / vel.norm() * T_all_seg.transpose() *
                                    data->V_all_segments_ * L_pp).transpose();

            grad_c[k] += grad_c_k;
          }
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
        void* opt_data) {
  CHECK_NOTNULL(opt_data);

  PolynomialOptimizationNonLinear<N>* data =
          static_cast<PolynomialOptimizationNonLinear<N>*>(opt_data);

  // Get distance from collision at current position
  double distance = data->sdf_->Get(position[0], position[1], position[2]);

  // Get potential cost from distance to collision
  double J_c_esdf = data->getCostPotential(distance);

  if (gradient != NULL) {
    std::vector<double> grad_c_esdf = data->sdf_->GetGradient3d(position, true);

    // Numerical gradients
    std::vector<double> grad_c_potential(data->dimension_);
    double increment_dist = 0.05; // map resolution
    Eigen::VectorXd increment(data->dimension_);
    for (int k = 0; k < data->dimension_; ++k) {
      increment.setZero();
      increment[k] = increment_dist;

      // Get distance and potential cost from collision at current position
      double left_dist = data->sdf_->Get3d(position-increment);
      double left_cost = data->getCostPotential(left_dist);
      double right_dist = data->sdf_->Get3d(position+increment);
      double right_cost = data->getCostPotential(right_dist);

      grad_c_potential[k] += (right_cost - left_cost) / (2.0 * increment_dist);
    }
    // TODO: GET RID --> only debug (adjust opti param boundaries lower_ upper_)
    // TODO: BUGGGGGGG
    if (grad_c_potential.empty()) {
      std::cout << "GRAD EMPTY --> SET ZERO" << std::endl;
      grad_c_potential = std::vector<double>(3, 0.0);
    }

    (*gradient)[0] = grad_c_potential[0];
    (*gradient)[1] = grad_c_potential[1];
    (*gradient)[2] = grad_c_potential[2];
  }

  return J_c_esdf;
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
  double increment_dist = data->optimization_parameters_.map_resolution;

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
      double cost_left = data->getCostAndGradientCollision(NULL, data);

      for (int k2 = 0; k2 < dim; ++k2) {
        free_constraints_right[k2] = free_constraints[k2] + increment[k2];
      }
      data->poly_opt_.setFreeConstraints(free_constraints_right);
      double cost_right = data->getCostAndGradientCollision(NULL, data);

      double grad_k_n = (cost_right - cost_left) / (2.0 * increment_dist);
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
    double increment_dist = data->optimization_parameters_.map_resolution;

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
        double cost_left = data->evaluateMaximumMagnitudeAsSoftConstraint(
                data->inequality_constraints_,
                data->optimization_parameters_.soft_constraint_weight);

        for (int k2 = 0; k2 < dim; ++k2) {
          free_constraints_right[k2] = free_constraints[k2] + increment[k2];
        }
        data->poly_opt_.setFreeConstraints(free_constraints_right);
        double cost_right = data->evaluateMaximumMagnitudeAsSoftConstraint(
                data->inequality_constraints_,
                data->optimization_parameters_.soft_constraint_weight);

        double grad_k_n = (cost_right - cost_left) / (2.0 * increment_dist);
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
double PolynomialOptimizationNonLinear<_N>::getCostPotential(
        double collision_distance) {

  double cost = 0.0;

  // TODO: paramterize
  double epsilon = 0.5;
  double robot_size = 0.8;
  double robot_radius = std::sqrt(3) * robot_size/2.0;

  collision_distance -= robot_radius;
  if (collision_distance < 0.0) {
    cost = -collision_distance + 0.5 * epsilon;
  } else if (collision_distance <= epsilon) {
    double epsilon_distance = collision_distance - epsilon;
    cost = 0.5 * 1.0 / epsilon * epsilon_distance * epsilon_distance;
  } else {
    cost = 0.0;
  }

  return cost;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::evaluateMaximumMagnitudeConstraint(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  // TODO: How to handle soft-constraints in gradient-based case?
//  CHECK(gradient.empty())
//      << "computing gradient not possible, choose a gradient free method";
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

  if (optimization_parameters_.print_debug_info)
    std::cout << "  soft_constraints: " << std::endl;

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
    if (optimization_parameters_.print_debug_info) {
      std::cout << "    derivative " << constraint->derivative
                << " abs violation: " << abs_violation
                << " : relative violation: " << relative_violation
                << " cost: " << current_cost << std::endl;
    }
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
