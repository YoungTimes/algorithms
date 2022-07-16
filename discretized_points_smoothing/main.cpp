#include <vector>
#include <iostream>

#include "fem_pos_deviation_osqp_interface.h"

#include "fem_pos_deviation_sqp_osqp_interface.h"

// max_constraint_interval : 0.25
// longitudinal_boundary_bound : 2.0
// max_lateral_boundary_bound : 0.5
// min_lateral_boundary_bound : 0.1
// curb_shift : 0.2
// lateral_buffer : 0.2

// discrete_points {
//   smoothing_method: FEM_POS_DEVIATION_SMOOTHING
//   fem_pos_deviation_smoothing {
//     weight_fem_pos_deviation: 1e10
//     weight_ref_deviation: 1.0
//     weight_path_length: 1.0
//     apply_curvature_constraint: false
//     max_iter: 500
//     time_limit: 0.0
//     verbose: false
//     scaled_termination: true
//     warm_start: true
//   }
// }

bool SqpWithOsqp(  
    const std::vector<std::pair<double, double>>& raw_point2d,
    std::vector<double>* opt_x,
    std::vector<double>* opt_y) {

  if (opt_x == nullptr || opt_y == nullptr) {
    std::cout<< "opt_x or opt_y is nullptr";
    return false;
  }

  FemPosDeviationSqpOsqpInterface solver;

  const double weight_fem_pos_deviation = 1e10;
  const double weight_ref_deviation = 1.0;
  const double weight_path_length = 1.0;
  const double weight_curvature_constraint_slack_var = 1e8;

  const double curvature_constraint = 0.18;
  const int max_iter = 500;
  const int sqp_sub_max_iter = 100;
  double time_limit = 0.0;

  const double sqp_ftol = 1e-4;
  const double sqp_ctol = 1e-3;
  const int sqp_pen_max_iter = 10;


  solver.set_weight_fem_pos_deviation(weight_fem_pos_deviation);
  solver.set_weight_path_length(weight_path_length);
  solver.set_weight_ref_deviation(weight_ref_deviation);
  solver.set_weight_curvature_constraint_slack_var(weight_curvature_constraint_slack_var);

  solver.set_curvature_constraint(curvature_constraint);

  solver.set_sqp_sub_max_iter(sqp_sub_max_iter);
  solver.set_sqp_ftol(sqp_ftol);
  solver.set_sqp_pen_max_iter(sqp_pen_max_iter);
  solver.set_sqp_ctol(sqp_ctol);

  solver.set_max_iter(max_iter);
  solver.set_time_limit(time_limit);
  solver.set_verbose(false);
  solver.set_scaled_termination(true);
  solver.set_warm_start(true);

  solver.set_ref_points(raw_point2d);

  std::vector<double> bounds;
  for (size_t i = 0; i < raw_point2d.size(); i++) {
    bounds.emplace_back(4.0);
  }
  bounds.front() = 0.0;
  bounds.back() = 0.0;

  solver.set_bounds_around_refs(bounds);

  if (!solver.Solve()) {
    return false;
  }

  std::vector<std::pair<double, double>> opt_xy = solver.opt_xy();

  // TODO(Jinyun): unify output data container
  opt_x->resize(opt_xy.size());
  opt_y->resize(opt_xy.size());
  for (size_t i = 0; i < opt_xy.size(); ++i) {
    (*opt_x)[i] = opt_xy[i].first;
    (*opt_y)[i] = opt_xy[i].second;
  }
  return true;
}


bool QpWithOsqp(
    const std::vector<std::pair<double, double>>& raw_point2d,
    std::vector<double>* opt_x,
    std::vector<double>* opt_y) {
  if (opt_x == nullptr || opt_y == nullptr) {
    std::cout<< "opt_x or opt_y is nullptr";
    return false;
  }

  FemPosDeviationOsqpInterface solver;

  double weight_fem_pos_deviation = 1e10;
  double weight_ref_deviation = 1.0;
  double weight_path_length = 1.0;
  int max_iter = 500;
  double time_limit = 0.0;

  solver.set_weight_fem_pos_deviation(weight_fem_pos_deviation);
  solver.set_weight_path_length(weight_path_length);
  solver.set_weight_ref_deviation(weight_ref_deviation);

  solver.set_max_iter(max_iter);
  solver.set_time_limit(time_limit);
  solver.set_verbose(false);
  solver.set_scaled_termination(true);
  solver.set_warm_start(true);

  solver.set_ref_points(raw_point2d);


  std::vector<double> bounds;
  for (size_t i = 0; i < raw_point2d.size(); i++) {
    bounds.emplace_back(4.0);
  }
  bounds.front() = 0.0;
  bounds.back() = 0.0;

  solver.set_bounds_around_refs(bounds);

  if (!solver.Solve()) {
    return false;
  }

  *opt_x = solver.opt_x();
  *opt_y = solver.opt_y();
  return true;
}


int main(int argc, char **argv) {

  double a[][2] = {{-30.,          -6.        },
                {-20.,          -6.        },
                {-10.,          -6.        },
                {  0.,          -6.        },
                { 10.,          -6.        },
                { 20.,          -6.        },
                { 30.,          -6.        },
                { 30.,           0.        },
                { 32.,           0.        },
                { 33.9849418 ,   0.21132565},
                { 35.94413172,   0.66303429},
                { 37.86796721,   1.33861618},
                { 39.74684571,   2.22156159},
                { 41.57116467,   3.2953608 },
                { 43.33132154,   4.54350405},
                { 45.01771376,   5.94948163},
                { 46.62073877,   7.4967838 },
                { 48.13079403,   9.16890081},
                { 49.53827697,  10.94932295},
                { 50.83358506,  12.82154046},
                { 52.00711572,  14.76904363},
                { 53.04926642,  16.77532272},
                { 53.95043459,  18.82386798},
                { 54.70101768,  20.8981697 },
                { 55.29141313,  22.98171813},
                { 55.7120184 ,  25.05800353},
                { 55.95323094,  27.11051619},
                { 56.01141343,  29.12462997},
                { 62. ,         30.        },
                { 62. ,         40.        },
                { 62. ,         50.        },
                { 62. ,         60.        },
                { 62. ,         70.        }};

  std::vector<std::pair<double, double>> ref_pts;
  for (int i = 0; i < 33; i++) {
    ref_pts.emplace_back(a[i][0], a[i][1]);
  }

  std::vector<double> opt_x;
  std::vector<double> opt_y;
  if (SqpWithOsqp(ref_pts, &opt_x, &opt_y) == false) {
    std::cout << "fail to solve osqp" << std::endl;
    return -1;
  }

  std::cout << "solve success." << std::endl;

  for (size_t i = 0; i < opt_x.size(); i++) {
    std::cout << "[" << opt_x[i] << ", " << opt_y[i] << "],";
  }
  std::cout << std::endl;

  return 0;

}
