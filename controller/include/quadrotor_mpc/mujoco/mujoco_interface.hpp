#pragma once
/**
 * MujocoInterface — Reusable ROS2 node for MuJoCo quadrotor SiL.
 *
 * Subscribes  /quadrotor/odom        (nav_msgs/Odometry)
 * Publishes   /quadrotor/trpy_cmd    (quadrotor_msgs/TRPYCommand)
 * Subscribes  /quadrotor/collision   (std_msgs/Bool)
 *
 * Thread-safe: state protected by mutex so the control loop (main thread)
 * can read while the ROS2 executor writes from callbacks.
 *
 * Reusable by NMPC, MPCC, DQ-MPCC — controller-agnostic.
 */

#include "quadrotor_mpc/common/types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <quadrotor_msgs/msg/trpy_command.hpp>

#include <Eigen/Dense>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>

namespace quadrotor_mpc {

/// Full 13-state snapshot from MuJoCo odometry + 3-vec accel from IMU
struct DroneState {
    Vec3  pos   = Vec3::Zero();       // world frame [m]
    Vec3  vel   = Vec3::Zero();       // world frame [m/s]
    Quat4 quat  = Quat4(1,0,0,0);    // [qw,qx,qy,qz]
    Vec3  omega = Vec3::Zero();       // body frame  [rad/s]
    Vec3  accel = Vec3::Zero();       // body frame  [m/s²]  (from /imu, includes gravity)

    /// Pack into 13-element vector [p, v, q, ω]
    Eigen::Matrix<double,13,1> to_vector() const {
        Eigen::Matrix<double,13,1> x;
        x << pos, vel, quat, omega;
        return x;
    }
};

/// Acro-mode command: thrust [N] + desired body rates [rad/s]
struct AcroCommand {
    double thrust = 0.0;
    Vec3   omega_cmd = Vec3::Zero();  // [wx, wy, wz]
};

class MujocoInterface : public rclcpp::Node {
public:
    // Derive all topic/service paths from quad_name namespace:
    //   /quad_name/{odom, trpy_cmd, collision, sim/reset}
    // quad_name="quadrotor" is the default namespace.
    explicit MujocoInterface(
        const std::string& node_name = "mujoco_controller",
        const std::string& quad_name = "quadrotor");

    // State access (thread-safe)
    DroneState get_state() const;
    bool is_connected() const;
    bool is_crashed() const;
    void clear_crash();

    // Command publishing
    void send_cmd(const AcroCommand& cmd);
    void send_cmd(double thrust, double wx, double wy, double wz);
    void send_zero();

    // PD position hold (background thread)
    struct PdGains {
        double kp_xy  = 4.0,  kd_xy  = 2.5;
        double kp_z   = 8.0,  kd_z   = 4.0;
        double kp_att = 6.0,  kp_yaw = 2.0;
    };

    void start_pd_hold(const Vec3& target, double mass, double g,
                       const PdGains& gains);
    void start_pd_hold(const Vec3& target, double mass, double g = 9.81);
    void stop_pd_hold();

    // Simulator control
    bool reset_sim(double timeout_sec = 5.0);

    // Convergence wait
    bool wait_for_connection(double timeout_sec = 10.0) const;
    bool wait_for_pd_convergence(const Vec3& target,
                                  double settle_dist = 0.30,
                                  double settle_time = 1.0,
                                  double timeout_sec = 15.0) const;

private:
    // Callbacks
    void odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imu_cb_(const sensor_msgs::msg::Imu::SharedPtr msg);
    void collision_cb_(const std_msgs::msg::Bool::SharedPtr msg);

    // State
    mutable std::mutex state_mtx_;
    DroneState state_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> crashed_{false};

    // PD hold
    std::atomic<bool> pd_active_{false};
    std::thread pd_thread_;
    void pd_loop_(Vec3 target, double mass, double g, PdGains gains);

    // ROS2
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_sub_;
    rclcpp::Publisher<quadrotor_msgs::msg::TRPYCommand>::SharedPtr cmd_pub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_cli_;
};

}  // namespace quadrotor_mpc
