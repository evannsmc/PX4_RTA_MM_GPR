// The 100 Hz control law of px4_rta_mm_gpr, ported from Python/JAX (no ROS dependencies, so it can be unit
// tested against the Python kernels):
//   * NR tracker   -- jax_nr/jax_newtonraphson_utilities.py: NR_tracker_original
//   * RTA feedback -- jax_mm_rta/mm_rta.py: u_applied
//   * fused step   -- control_kernels.py: ControlKernels.control_step (RTA thrust/roll + NR pitch/yaw, clipped)
//
// The NR tracker needs d(predicted output)/d(input), a 4x4 Jacobian that JAX gets by forward-mode autodiff.
// Here the same thing is done with a dual number carrying FOUR tangent directions (one per input): running the
// forward-Euler predictor once on Jet<4> values yields the prediction and the full Jacobian in a single pass,
// exact to rounding (no finite-difference step size to tune).
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace rta {

constexpr double GRAVITY = 9.806;

// ---------------------------------------------------------------------------------------------------------------
// Forward-mode dual number with N tangent directions: v + sum_i d[i] eps_i
template <int N>
struct Jet {
  double v{0.0};
  Eigen::Matrix<double, N, 1> d{Eigen::Matrix<double, N, 1>::Zero()};
  Jet() = default;
  Jet(double value) : v(value) {}  // NOLINT: constants promote implicitly
  Jet(double value, const Eigen::Matrix<double, N, 1>& grad) : v(value), d(grad) {}
};
template <int N> Jet<N> operator+(const Jet<N>& a, const Jet<N>& b) { return {a.v + b.v, a.d + b.d}; }
template <int N> Jet<N> operator-(const Jet<N>& a, const Jet<N>& b) { return {a.v - b.v, a.d - b.d}; }
template <int N> Jet<N> operator-(const Jet<N>& a) { return {-a.v, -a.d}; }
template <int N> Jet<N> operator*(const Jet<N>& a, const Jet<N>& b) { return {a.v * b.v, a.d * b.v + b.d * a.v}; }
template <int N> Jet<N> operator/(const Jet<N>& a, const Jet<N>& b) {
  return {a.v / b.v, (a.d * b.v - b.d * a.v) / (b.v * b.v)};
}
template <int N> Jet<N> sin(const Jet<N>& a) { return {std::sin(a.v), a.d * std::cos(a.v)}; }
template <int N> Jet<N> cos(const Jet<N>& a) { return {std::cos(a.v), -a.d * std::sin(a.v)}; }
template <int N> Jet<N> tan(const Jet<N>& a) {
  const double t = std::tan(a.v);
  return {t, a.d * (1.0 + t * t)};
}
inline double value(double x) { return x; }
template <int N> double value(const Jet<N>& x) { return x.v; }
using std::cos;
using std::sin;
using std::tan;

// ---------------------------------------------------------------------------------------------------------------
// Quadrotor dynamics (jax_nr.dynamics): state = (x y z vx vy vz roll pitch yaw), input = (thrust, p, q, r)
template <typename T>
std::array<T, 9> dynamics(const std::array<T, 9>& s, const std::array<T, 4>& u, double mass) {
  const T& roll = s[6];
  const T& pitch = s[7];
  const T& yaw = s[8];
  const T sr = sin(roll), cr = cos(roll), sp = sin(pitch), cp = cos(pitch), sy = sin(yaw), cy = cos(yaw);
  const T tp = tan(pitch);
  // Euler-angle rates from body rates: T(roll, pitch) @ (p, q, r)
  const T rolldot = u[1] + sr * tp * u[2] + cr * tp * u[3];
  const T pitchdot = cr * u[2] - sr * u[3];
  const T yawdot = sr / cp * u[2] + cr / cp * u[3];
  const T thrust_over_m = u[0] / T(mass);
  const T vxdot = -thrust_over_m * (sr * sy + cr * cy * sp);
  const T vydot = -thrust_over_m * (cr * sy * sp - cy * sr);
  const T vzdot = T(GRAVITY) - thrust_over_m * (cr * cp);
  return {s[3], s[4], s[5], vxdot, vydot, vzdot, rolldot, pitchdot, yawdot};
}

// Forward-Euler prediction of the output (x, y, z, yaw) after t_lookahead (jax_nr.predict_output)
template <typename T>
std::array<T, 4> predict_output(const std::array<double, 9>& state, const std::array<T, 4>& u, double t_lookahead,
                                double step, double mass) {
  std::array<T, 9> s;
  for (int i = 0; i < 9; ++i) s[i] = T(state[i]);
  const int n = static_cast<int>(t_lookahead / step);  // JAX: (T / step).astype(int)
  for (int k = 0; k < n; ++k) {
    const auto f = dynamics(s, u, mass);
    for (int i = 0; i < 9; ++i) s[i] = s[i] + f[i] * T(step);
  }
  return {s[0], s[1], s[2], s[8]};
}

// Integral control barrier function (jax_nr.execute_cbf / integral_cbf)
inline double execute_cbf(double current, double phi, double max_value, double min_value, double gamma) {
  const double zeta_max = gamma * (max_value - current) - phi;
  const double zeta_min = gamma * (min_value - current) - phi;
  return current >= 0.0 ? std::min(0.0, zeta_max) : std::max(0.0, zeta_min);
}

inline Eigen::Vector4d integral_cbf(const Eigen::Vector4d& last_input, const Eigen::Vector4d& phi) {
  constexpr double thrust_max = 27.0, thrust_min = 0.5, rates_max = 0.8, gamma = 1.0;
  return {execute_cbf(last_input[0], phi[0], thrust_max, thrust_min, gamma),
          execute_cbf(last_input[1], phi[1], rates_max, -rates_max, gamma),
          execute_cbf(last_input[2], phi[2], rates_max, -rates_max, gamma),
          execute_cbf(last_input[3], phi[3], rates_max, -rates_max, gamma)};
}

// Moore-Penrose pseudo-inverse with numpy/JAX's default cutoff: rcond = 10 * max(M, N) * eps (relative to smax)
inline Eigen::Matrix4d pinv(const Eigen::Matrix4d& A) {
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector4d s = svd.singularValues();
  const double cutoff = 10.0 * 4.0 * std::numeric_limits<double>::epsilon() * s.maxCoeff();
  Eigen::Vector4d s_inv;
  for (int i = 0; i < 4; ++i) s_inv[i] = s[i] > cutoff ? 1.0 / s[i] : 0.0;
  return svd.matrixV() * s_inv.asDiagonal() * svd.matrixU().transpose();
}

struct NRParams {
  double mass{2.0};
  double t_lookahead{0.8};
  double lookahead_step{0.1};
  double integration_step{0.01};
  Eigen::Vector4d alpha{20.0, 30.0, 30.0, 50.0};
};

// NR_tracker_original: one Newton-Raphson input update toward ref = (x, y, z, yaw)
inline Eigen::Vector4d nr_tracker(const std::array<double, 9>& state, const Eigen::Vector4d& last_input,
                                  const Eigen::Vector4d& ref, const NRParams& p) {
  using J4 = Jet<4>;
  std::array<J4, 4> u;
  for (int i = 0; i < 4; ++i) u[i] = J4(last_input[i], Eigen::Vector4d::Unit(i));  // seed d/du_i
  const auto pred = predict_output(state, u, p.t_lookahead, p.lookahead_step, p.mass);

  Eigen::Vector4d pred_v;
  Eigen::Matrix4d dgdu;
  for (int r = 0; r < 4; ++r) {
    pred_v[r] = pred[r].v;
    dgdu.row(r) = pred[r].d.transpose();
  }
  Eigen::Vector4d error = ref - pred_v;
  // Yaw error as in get_tracking_error: q_err = q(ref) * conj(q(pred)) -> 2 atan2(sin(d/2), cos(d/2))
  const double d = ref[3] - pred_v[3];
  error[3] = 2.0 * std::atan2(std::sin(d / 2.0), std::cos(d / 2.0));

  const Eigen::Vector4d nr = pinv(dgdu) * error;
  const Eigen::Vector4d v = integral_cbf(last_input, nr);
  const Eigen::Vector4d udot = nr + v;
  return last_input + p.alpha.cwiseProduct(udot * p.integration_step);
}

// RTA feedback around the plan (mm_rta.u_applied): u = u_ff - K (x - x_ref), clipped to the input limits
inline Eigen::Vector2d rta_feedback(const Eigen::Matrix<double, 5, 1>& x, const Eigen::Matrix<double, 5, 1>& x_ref,
                                    const Eigen::Vector2d& u_ff, const Eigen::Matrix<double, 2, 5>& K,
                                    const Eigen::Vector2d& u_lo, const Eigen::Vector2d& u_hi) {
  const Eigen::Vector2d u = u_ff - K * (x - x_ref);
  return u.cwiseMax(u_lo).cwiseMin(u_hi);
}

// ControlKernels.control_step: RTA (thrust, roll rate) + NR (pitch, yaw rates), then the anti-windup clip
inline Eigen::Vector4d control_step(const std::array<double, 9>& nr_state, const Eigen::Matrix<double, 5, 1>& planar,
                                    const Eigen::Vector4d& last_input, const Eigen::Vector4d& ref_nr,
                                    const Eigen::Matrix<double, 5, 1>& ref_row, const Eigen::Vector2d& ff_row,
                                    const Eigen::Matrix<double, 2, 5>& K, const Eigen::Vector2d& u_lo,
                                    const Eigen::Vector2d& u_hi, const Eigen::Vector4d& clip_lo,
                                    const Eigen::Vector4d& clip_hi, const NRParams& p) {
  const Eigen::Vector4d nr_u = nr_tracker(nr_state, last_input, ref_nr, p);
  const Eigen::Vector2d rta_u = rta_feedback(planar, ref_row, ff_row, K, u_lo, u_hi);
  Eigen::Vector4d u(rta_u[0], rta_u[1], nr_u[2], nr_u[3]);
  return u.cwiseMax(clip_lo).cwiseMin(clip_hi);
}

// Force -> normalised throttle (utilities/sim_utilities.py, hardware_utilities.py)
inline double throttle_from_force_sim(double force) {
  constexpr double thrust_constant = 8.54858e-06, motor_velocity_armed = 10.0, motor_input_scaling = 1000.0;
  const double motor_speed = std::sqrt(force / (4.0 * thrust_constant));
  return (motor_speed - motor_velocity_armed) / motor_input_scaling;
}
inline double throttle_from_force_hardware(double force) {
  constexpr double a = 0.00705385408507030, b = 0.0807474474438391, c = 0.0252575818743285;
  return a * force + b * std::sqrt(force) + c;
}

// Quaternion (w, x, y, z; FRD body -> NED) to roll, pitch, yaw -- scipy as_euler('xyz') (extrinsic x-y-z)
inline std::array<double, 3> euler_xyz(double w, double x, double y, double z) {
  const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  const double sinp = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
  const double pitch = std::asin(sinp);
  const double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  return {roll, pitch, yaw};
}

// utilities.adjust_yaw: accumulate the wrapped increment between samples
class YawUnwrapper {
 public:
  double operator()(double yaw) {
    if (!initialized_) {
      initialized_ = true;
      prev_ = unwrapped_ = yaw;
      return yaw;
    }
    double delta = std::fmod(yaw - prev_ + M_PI, 2.0 * M_PI);
    if (delta < 0) delta += 2.0 * M_PI;
    unwrapped_ += delta - M_PI;
    prev_ = yaw;
    return unwrapped_;
  }

 private:
  bool initialized_{false};
  double prev_{0.0}, unwrapped_{0.0};
};

}  // namespace rta
