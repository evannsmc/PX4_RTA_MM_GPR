// rta_fast_loop: the C++ half of the RTA-MM-GPR stack.
//
// Owns everything that must run on time: PX4 offboard heartbeat, arming/mode logic, odometry, the 100 Hz control
// law (NR tracker + RTA feedback, see control_law.hpp) and the runtime-assurance watchdog. The Python planner
// (px4_rta_mm_gpr --cpp-control) computes the certified plans, the GP/wind estimate and the LQR gains, and
// publishes them; it also logs every ControlTick this node publishes.
//
// Threading: a MultiThreadedExecutor with two mutually-exclusive callback groups -- subscriptions in one, the
// control and heartbeat timers in the other -- so deserialising a plan never delays a control tick. Shared data
// (state, plan, gains, config) is exchanged as immutable snapshots through std::atomic<std::shared_ptr<const T>>:
// a writer builds a new object and swaps the pointer, a reader loads it once per callback.
#include "px4_rta_mm_gpr_cpp/control_law.hpp"

#include <rclcpp/rclcpp.hpp>

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/rc_channels.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_rta_mm_gpr_msgs/msg/control_tick.hpp>
#include <px4_rta_mm_gpr_msgs/msg/planner_status.hpp>
#include <px4_rta_mm_gpr_msgs/msg/rta_gains.hpp>
#include <px4_rta_mm_gpr_msgs/msg/rta_plan.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using px4_msgs::msg::OffboardControlMode;
using px4_msgs::msg::TrajectorySetpoint;
using px4_msgs::msg::VehicleCommand;
using px4_msgs::msg::VehicleRatesSetpoint;
using px4_msgs::msg::VehicleStatus;
using px4_rta_mm_gpr_msgs::msg::ControlTick;
using px4_rta_mm_gpr_msgs::msg::PlannerStatus;
using px4_rta_mm_gpr_msgs::msg::RtaGains;
using px4_rta_mm_gpr_msgs::msg::RtaPlan;

namespace {

double epoch_now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct VehicleState {
  std::array<double, 9> nr{};                 // x y z vx vy vz roll pitch yaw
  Eigen::Matrix<double, 5, 1> planar;         // py pz vy vz roll
};

struct Plan {
  uint32_t seq{0};
  double t_start{0}, dt{0.01}, collection_time{0};
  int32_t violation_idx{-1};
  std::vector<double> ref;  // n_rows x 5
  std::vector<double> ff;   // (n_rows - 1) x 2
  int n_rows() const { return static_cast<int>(ref.size() / 5); }
  int n_ff() const { return static_cast<int>(ff.size() / 2); }
  int index_at(double t) const {  // RolloutPlan.index_at
    const int idx = static_cast<int>((t - t_start) / dt + 1e-6);  // + 1e-6 row: exact row times must not truncate
    return std::clamp(idx, 0, std::max(n_ff() - 1, 0));
  }
  Eigen::Matrix<double, 5, 1> ref_row(int i) const {
    i = std::clamp(i, 0, n_rows() - 1);
    return Eigen::Map<const Eigen::Matrix<double, 5, 1>>(&ref[5 * i]);
  }
  Eigen::Vector2d ff_row(int i) const { return {ff[2 * i], ff[2 * i + 1]}; }
};

struct Gains {
  Eigen::Matrix<double, 2, 5> K;
};

}  // namespace

class RtaFastLoop : public rclcpp::Node {
 public:
  RtaFastLoop() : Node("rta_fast_loop") {
    sub_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    loop_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = sub_group_;

    const auto px4_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().transient_local();
    offboard_mode_pub_ = create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", px4_qos);
    vehicle_command_pub_ = create_publisher<VehicleCommand>("/fmu/in/vehicle_command", px4_qos);
    trajectory_pub_ = create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", px4_qos);
    rates_pub_ = create_publisher<VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", px4_qos);
    tick_pub_ = create_publisher<ControlTick>("/rta/control_tick", rclcpp::QoS(100).reliable());

    // PX4's odometry directly (100 Hz, NED), not the mocap_px4_relays relay (25 Hz with gaps up to 150 ms)
    odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", px4_qos,
        [this](px4_msgs::msg::VehicleOdometry::ConstSharedPtr msg) { on_odometry(*msg); }, sub_opts);
    status_sub_ = create_subscription<VehicleStatus>(
        "/fmu/out/vehicle_status_v1", px4_qos,
        [this](VehicleStatus::ConstSharedPtr msg) {
          nav_state_.store(msg->nav_state);
          armed_.store(msg->arming_state == VehicleStatus::ARMING_STATE_ARMED);
        },
        sub_opts);
    rc_sub_ = create_subscription<px4_msgs::msg::RcChannels>(
        "/fmu/out/rc_channels", px4_qos,
        [this](px4_msgs::msg::RcChannels::ConstSharedPtr msg) {
          rc_offboard_.store(msg->channels[kModeChannel - 1] >= 0.75f);
        },
        sub_opts);
    planner_sub_ = create_subscription<PlannerStatus>(
        "/rta/planner_status", rclcpp::QoS(10).reliable(),
        [this](PlannerStatus::ConstSharedPtr msg) { on_planner_status(msg); }, sub_opts);
    plan_sub_ = create_subscription<RtaPlan>(
        "/rta/plan", rclcpp::QoS(10).reliable(), [this](RtaPlan::ConstSharedPtr msg) { on_plan(*msg); }, sub_opts);
    gains_sub_ = create_subscription<RtaGains>(
        "/rta/gains", rclcpp::QoS(10).reliable(),
        [this](RtaGains::ConstSharedPtr msg) {
          auto g = std::make_shared<Gains>();
          for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 5; ++c) g->K(r, c) = msg->k_feedback[5 * r + c];
          gains_.store(std::const_pointer_cast<const Gains>(g));
        },
        sub_opts);

    heartbeat_timer_ = create_wall_timer(100ms, [this] { heartbeat(); }, loop_group_);
    control_timer_ = create_wall_timer(10ms, [this] { control(); }, loop_group_);
    RCLCPP_INFO(get_logger(), "rta_fast_loop up; waiting for the planner (/rta/planner_status)");
  }

 private:
  static constexpr int kModeChannel = 5;

  // ---------------------------------------------------------------------------------------------- subscriptions
  void on_planner_status(const PlannerStatus::ConstSharedPtr& msg) {
    if (config_.load()) return;  // first status fixes the mission clock and every tunable
    // Derived values first, config_ LAST: the control thread only reads them after it has loaded a non-null
    // config_, and the (sequentially consistent) atomic store orders these writes before that load.
    if (msg->sim) rc_offboard_.store(true);  // no RC in sim: the offboard switch is implied on
    nr_params_.mass = msg->mass;
    nr_params_.t_lookahead = msg->t_lookahead;
    nr_params_.lookahead_step = msg->lookahead_step;
    nr_params_.integration_step = msg->integration_step;
    u_lo_ = {msg->thrust_min, -msg->roll_rate_max};
    u_hi_ = {msg->thrust_max, msg->roll_rate_max};
    const double inf = std::numeric_limits<double>::infinity();
    const double r = msg->nr_rate_limit;
    clip_lo_ = {-inf, -inf, -r, -r};
    clip_hi_ = {inf, inf, r, r};
    last_input_ = {msg->mass * rta::GRAVITY, 0.01, 0.01, 0.01};
    config_.store(msg);
    RCLCPP_INFO(get_logger(),
                "planner connected: mission t0=%.3f, mass %.2f kg, thrust [%.2f, %.2f] N, NR rate limit %.2f, "
                "backup '%s' (grace %.3f s)",
                msg->t0_epoch, msg->mass, msg->thrust_min, msg->thrust_max, msg->nr_rate_limit, msg->backup.c_str(),
                msg->backup_grace);
  }

  void on_odometry(const px4_msgs::msg::VehicleOdometry& msg) {
    auto s = std::make_shared<VehicleState>();
    const auto [roll, pitch, yaw_raw] = rta::euler_xyz(msg.q[0], msg.q[1], msg.q[2], msg.q[3]);
    const double yaw = unwrap_(yaw_raw);  // only this callback touches the unwrapper (subscription group)
    s->nr = {msg.position[0], msg.position[1], msg.position[2], msg.velocity[0], msg.velocity[1], msg.velocity[2],
             roll, pitch, yaw};
    // planar model state (py, pz, h, v, theta): h, v are BODY-frame velocities, [h; v] = R(theta)^T [vy; vz]
    const double c = std::cos(roll), sn = std::sin(roll);
    s->planar << msg.position[1], msg.position[2], c * msg.velocity[1] + sn * msg.velocity[2],
        -sn * msg.velocity[1] + c * msg.velocity[2], roll;
    state_.store(std::const_pointer_cast<const VehicleState>(s));
  }

  void on_plan(const RtaPlan& msg) {
    auto p = std::make_shared<Plan>();
    p->seq = msg.seq;
    p->t_start = msg.t_start;
    p->dt = msg.dt;
    p->collection_time = msg.collection_time;
    p->violation_idx = msg.violation_idx;
    p->ref = msg.rollout_ref;
    p->ff = msg.feedfwd_input;
    if (p->n_rows() < 1 || p->n_ff() < 1) return;
    plan_.store(std::const_pointer_cast<const Plan>(p));
  }

  // ---------------------------------------------------------------------------------------------- timers
  double mission_time(const PlannerStatus& cfg) const { return epoch_now() - cfg.t0_epoch; }

  void heartbeat() {
    const auto cfg = config_.load();
    if (!cfg) return;
    const double t = mission_time(*cfg);
    if (!rc_offboard_.load()) {
      heartbeat_counter_ = 0;
      return;
    }
    const bool body_rate = (t >= cfg->begin_actuator_control && t < cfg->land_time);
    OffboardControlMode m;
    m.timestamp = now_us();
    m.position = !body_rate;
    m.body_rate = body_rate;
    offboard_mode_pub_->publish(m);
    if (heartbeat_counter_ <= 10) {
      if (heartbeat_counter_ == 10) {
        publish_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);  // offboard
        publish_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
        RCLCPP_INFO(get_logger(), "offboard + arm commands sent");
      }
      ++heartbeat_counter_;
    }
  }

  void control() {
    const double tick_start = epoch_now();
    const double period = std::isnan(last_tick_) ? 0.0 : tick_start - last_tick_;
    last_tick_ = tick_start;

    const auto cfg = config_.load();
    const auto s = state_.load();
    if (!cfg || !s || done_) return;
    if (!(rc_offboard_.load() && nav_state_.load() == VehicleStatus::NAVIGATION_STATE_OFFBOARD)) return;
    if (backup_reason_) return;  // PX4 LAND owns the vehicle now

    const double t = mission_time(*cfg);
    if (t < cfg->begin_actuator_control) {
      publish_position(0.0, cfg->max_y, cfg->max_height);
      return;
    }
    if (t < cfg->land_time) {
      const auto plan = plan_.load();
      const auto gains = gains_.load();
      if (!plan || !gains) {  // first safety plan not here yet: hold the pre-RTA waypoint
        publish_position(0.0, cfg->max_y, cfg->max_height);
        return;
      }
      rta_step(*cfg, *s, *plan, *gains, t, period);
      return;
    }
    // End of mission: descend to the landing point, then land + disarm (as the Python node did)
    publish_position(0.0, 0.0, -0.83);
    if (std::abs(s->nr[0]) < 0.25 && std::abs(s->nr[1]) < 0.25 && std::abs(s->nr[2]) <= 0.90) {
      publish_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
      publish_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f);
      RCLCPP_INFO(get_logger(), "mission complete: land + disarm");
      done_ = true;
    }
  }

  void rta_step(const PlannerStatus& cfg, const VehicleState& s, const Plan& plan, const Gains& gains, double t,
                double period) {
    // Certification watchdog: continuous time without a certified plan -> backup
    const bool expired = t > plan.collection_time;
    if (expired) {
      if (!uncertified_since_) uncertified_since_ = t;
      if (cfg.backup == "land" && t - *uncertified_since_ > cfg.backup_grace) {
        char reason[200];
        std::snprintf(reason, sizeof(reason),
                      "no certified plan for %.3f s (latest plan #%u: violation row %d)", t - *uncertified_since_,
                      plan.seq, plan.violation_idx);
        backup_reason_ = reason;
        publish_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
        RCLCPP_WARN(get_logger(), "t=%.2f s: BACKUP -> LAND (%s)", t, reason);
        publish_tick(t, ControlTick::PHASE_BACKUP, period, 0.0, s, last_input_, 0.0, plan, -1, true);
        return;
      }
    } else {
      uncertified_since_.reset();
    }

    // NR reference: x = 0, yaw = 0; y/z from the plan at the NR prediction horizon (or the old ramp)
    Eigen::Vector4d ref_nr(0.0, 0.0, std::clamp(cfg.max_height + 0.1 * t, cfg.max_height, -0.55), 0.0);
    if (cfg.nr_ref_from_plan) {
      const auto ahead = plan.ref_row(plan.index_at(t + cfg.t_lookahead));
      ref_nr[1] = ahead[0];
      ref_nr[2] = ahead[1];
    }
    const int idx = plan.index_at(t);

    const auto c0 = std::chrono::steady_clock::now();
    const Eigen::Vector4d u = rta::control_step(s.nr, s.planar, last_input_, ref_nr, plan.ref_row(idx),
                                                plan.ff_row(idx), gains.K, u_lo_, u_hi_, clip_lo_, clip_hi_,
                                                nr_params_);
    const double comp = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    last_input_ = u;

    const double throttle =
        cfg.sim ? rta::throttle_from_force_sim(u[0]) : rta::throttle_from_force_hardware(u[0]);
    VehicleRatesSetpoint r;
    r.timestamp = now_us();
    r.roll = static_cast<float>(u[1]);
    r.pitch = static_cast<float>(u[2]);
    r.yaw = static_cast<float>(u[3]);
    r.thrust_body = {0.0f, 0.0f, static_cast<float>(-throttle)};
    rates_pub_->publish(r);

    publish_tick(t, ControlTick::PHASE_RTA, period, comp, s, u, throttle, plan, idx, expired);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "t=%.1f s plan #%u idx %d u=[%.2f %.2f %.2f %.2f] %.0f us",
                         t, plan.seq, idx, u[0], u[1], u[2], u[3], comp * 1e6);
  }

  // ---------------------------------------------------------------------------------------------- publishing
  uint64_t now_us() { return static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000); }

  void publish_position(double x, double y, double z) {
    TrajectorySetpoint m;
    m.timestamp = now_us();
    m.position = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
    m.yaw = 0.0f;
    trajectory_pub_->publish(m);
  }

  void publish_command(uint32_t command, float param1 = 0.0f, float param2 = 0.0f) {
    VehicleCommand m;
    m.timestamp = now_us();
    m.command = command;
    m.param1 = param1;
    m.param2 = param2;
    m.target_system = 1;
    m.target_component = 1;
    m.source_system = 1;
    m.source_component = 1;
    m.from_external = true;
    vehicle_command_pub_->publish(m);
  }

  void publish_tick(double t, uint8_t phase, double period, double comp, const VehicleState& s,
                    const Eigen::Vector4d& u, double throttle, const Plan& plan, int idx, bool expired) {
    ControlTick m;
    m.t = t;
    m.phase = phase;
    m.control_period = period;
    m.ctrl_comp_time = comp;
    for (int i = 0; i < 9; ++i) m.nr_state[i] = s.nr[i];
    for (int i = 0; i < 5; ++i) m.planar_state[i] = s.planar[i];
    for (int i = 0; i < 4; ++i) m.u[i] = u[i];
    m.throttle = throttle;
    m.plan_seq = plan.seq;
    m.traj_idx = idx;
    m.plan_expired = expired;
    m.backup_engaged = backup_reason_.has_value();
    m.backup_reason = backup_reason_.value_or("");
    tick_pub_->publish(m);
  }

  // ---------------------------------------------------------------------------------------------- members
  rclcpp::CallbackGroup::SharedPtr sub_group_, loop_group_;
  rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_mode_pub_;
  rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_pub_;
  rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<VehicleRatesSetpoint>::SharedPtr rates_pub_;
  rclcpp::Publisher<ControlTick>::SharedPtr tick_pub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<px4_msgs::msg::RcChannels>::SharedPtr rc_sub_;
  rclcpp::Subscription<PlannerStatus>::SharedPtr planner_sub_;
  rclcpp::Subscription<RtaPlan>::SharedPtr plan_sub_;
  rclcpp::Subscription<RtaGains>::SharedPtr gains_sub_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_, control_timer_;

  // shared between the two callback groups: immutable snapshots / atomics only
  std::atomic<std::shared_ptr<const PlannerStatus>> config_;
  std::atomic<std::shared_ptr<const VehicleState>> state_;
  std::atomic<std::shared_ptr<const Plan>> plan_;
  std::atomic<std::shared_ptr<const Gains>> gains_;
  std::atomic<uint8_t> nav_state_{0};
  std::atomic<bool> armed_{false};
  std::atomic<bool> rc_offboard_{false};

  // control-group only (config-derived values are written once, before the first control tick uses them)
  rta::NRParams nr_params_;
  Eigen::Vector2d u_lo_{0, 0}, u_hi_{0, 0};
  Eigen::Vector4d clip_lo_, clip_hi_, last_input_{0, 0, 0, 0};
  std::optional<double> uncertified_since_;
  std::optional<std::string> backup_reason_;
  double last_tick_{std::numeric_limits<double>::quiet_NaN()};
  int heartbeat_counter_{0};
  bool done_{false};

  // subscription-group only
  rta::YawUnwrapper unwrap_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RtaFastLoop>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
