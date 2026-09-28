// Equivalence check for control_law.hpp against the Python/JAX kernels (see test/test_control_law.py).
// stdin, one case per line (55 numbers):
//   nr_state(9) planar(5) last_input(4) ref_nr(4) ref_row(5) ff_row(2) K(10, row-major) u_lo(2) u_hi(2)
//   clip_lo(4) clip_hi(4) mass t_lookahead lookahead_step integration_step
// stdout, one line per case: u(4)
#include "px4_rta_mm_gpr_cpp/control_law.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

int main() {
  std::cout.precision(17);
  while (true) {
    double v[55];
    for (double& x : v) {
      std::string token;  // strtod, not operator>>: the anti-windup bounds contain "inf"
      if (!(std::cin >> token)) return 0;
      x = std::strtod(token.c_str(), nullptr);
    }
    const double* p = v;
    std::array<double, 9> nr_state;
    for (auto& x : nr_state) x = *p++;
    Eigen::Matrix<double, 5, 1> planar, ref_row;
    Eigen::Vector4d last_input, ref_nr, clip_lo, clip_hi;
    Eigen::Vector2d ff_row, u_lo, u_hi;
    Eigen::Matrix<double, 2, 5> K;
    for (int i = 0; i < 5; ++i) planar[i] = *p++;
    for (int i = 0; i < 4; ++i) last_input[i] = *p++;
    for (int i = 0; i < 4; ++i) ref_nr[i] = *p++;
    for (int i = 0; i < 5; ++i) ref_row[i] = *p++;
    for (int i = 0; i < 2; ++i) ff_row[i] = *p++;
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 5; ++c) K(r, c) = *p++;
    for (int i = 0; i < 2; ++i) u_lo[i] = *p++;
    for (int i = 0; i < 2; ++i) u_hi[i] = *p++;
    for (int i = 0; i < 4; ++i) clip_lo[i] = *p++;
    for (int i = 0; i < 4; ++i) clip_hi[i] = *p++;
    rta::NRParams params;
    params.mass = *p++;
    params.t_lookahead = *p++;
    params.lookahead_step = *p++;
    params.integration_step = *p++;
    const Eigen::Vector4d u = rta::control_step(nr_state, planar, last_input, ref_nr, ref_row, ff_row, K, u_lo, u_hi,
                                                clip_lo, clip_hi, params);
    std::cout << u[0] << ' ' << u[1] << ' ' << u[2] << ' ' << u[3] << '\n';
  }
}
