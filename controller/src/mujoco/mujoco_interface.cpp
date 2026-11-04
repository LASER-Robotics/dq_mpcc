#include "quadrotor_mpc/mujoco/mujoco_interface.hpp"

#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std::chrono_literals;

namespace quadrotor_mpc {

//  Construction

// Helper: build all topic/service paths from quad_name namespace.
static void init_topics_(
    rclcpp::Node* self,
    const std::string& odom_topic, const std::string& cmd_topic,
    const std::string& collision_topic, const std::string& reset_service,
    rclcpp::Publisher<quadrotor_msgs::msg::TRPYCommand>::SharedPtr& cmd_pub,
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr& odom_sub,
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr& collision_sub,
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr& reset_cli,
    std::function<void(const nav_msgs::msg::Odometry::SharedPtr)> odom_cb,
    std::function<void(const std_msgs::msg::Bool::SharedPtr)> collision_cb)
{
    cmd_pub       = self->create_publisher<quadrotor_msgs::msg::TRPYCommand>(cmd_topic, 10);
    odom_sub      = self->create_subscription<nav_msgs::msg::Odometry>(odom_topic, 10, odom_cb);
    collision_sub = self->create_subscription<std_msgs::msg::Bool>(collision_topic, 10, collision_cb);
    reset_cli     = self->create_client<std_srvs::srv::Trigger>(reset_service);
    RCLCPP_INFO(self->get_logger(),
                "MujocoInterface ready  odom=%s  cmd=%s  collision=%s  reset=%s",
                odom_topic.c_str(), cmd_topic.c_str(),
                collision_topic.c_str(), reset_service.c_str());
}

MujocoInterface::MujocoInterface(
    const std::string& node_name,
    const std::string& quad_name)
    : Node(node_name)
{
    const std::string base = "/" + quad_name;
    init_topics_(this,
        base + "/odom", base + "/trpy_cmd",
        base + "/collision", base + "/sim/reset",
        cmd_pub_, odom_sub_, collision_sub_, reset_cli_,
        std::bind(&MujocoInterface::odom_cb_, this, std::placeholders::_1),
        std::bind(&MujocoInterface::collision_cb_, this, std::placeholders::_1));
    // IMU sub (accel body-frame, includes gravity)
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        base + "/imu", 10,
        std::bind(&MujocoInterface::imu_cb_, this, std::placeholders::_1));
}

//  Callbacks

void MujocoInterface::odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(state_mtx_);
    auto& p = msg->pose.pose.position;
    auto& q = msg->pose.pose.orientation;
    auto& v = msg->twist.twist.linear;

    state_.pos   << p.x, p.y, p.z;
    state_.vel   << v.x, v.y, v.z;
    state_.quat  << q.w, q.x, q.y, q.z;
    // ω is not read from odometry: it comes from the IMU gyro (imu_cb_), as on the vehicle.
    connected_.store(true);
}

void MujocoInterface::imu_cb_(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(state_mtx_);
    auto& a = msg->linear_acceleration;
    auto& w = msg->angular_velocity;
    state_.accel << a.x, a.y, a.z;
    state_.omega << w.x, w.y, w.z;   // ω feedback from the IMU gyro, as on the vehicle
}

void MujocoInterface::collision_cb_(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data) crashed_.store(true);
}

//  State access

DroneState MujocoInterface::get_state() const {
    std::lock_guard<std::mutex> lk(state_mtx_);
    return state_;
}

bool MujocoInterface::is_connected() const { return connected_.load(); }
bool MujocoInterface::is_crashed()   const { return crashed_.load(); }
void MujocoInterface::clear_crash()        { crashed_.store(false); }

//  Command publishing

void MujocoInterface::send_cmd(const AcroCommand& cmd) {
    send_cmd(cmd.thrust, cmd.omega_cmd.x(), cmd.omega_cmd.y(), cmd.omega_cmd.z());
}

void MujocoInterface::send_cmd(double thrust, double wx, double wy, double wz) {
    auto msg = quadrotor_msgs::msg::TRPYCommand();
    msg.thrust = thrust;
    msg.angular_velocity.x = wx;
    msg.angular_velocity.y = wy;
    msg.angular_velocity.z = wz;
    cmd_pub_->publish(msg);
}

void MujocoInterface::send_zero() { send_cmd(0.0, 0.0, 0.0, 0.0); }

//  Simulator reset

bool MujocoInterface::reset_sim(double timeout_sec) {
    using Clock = std::chrono::steady_clock;
    auto deadline = Clock::now() + std::chrono::duration<double>(timeout_sec);

    while (!reset_cli_->service_is_ready()) {
        if (Clock::now() > deadline) {
            RCLCPP_WARN(get_logger(), "Reset service not available (timeout)");
            return false;
        }
        std::this_thread::sleep_for(10ms);
    }

    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = reset_cli_->async_send_request(req);

    // Poll until done (compatible with external spin thread)
    while (future.wait_for(5ms) != std::future_status::ready) {
        if (Clock::now() > deadline) {
            RCLCPP_WARN(get_logger(), "Reset service timeout");
            return false;
        }
    }

    auto resp = future.get();
    RCLCPP_INFO(get_logger(), "Reset: success=%d  msg='%s'",
                resp->success, resp->message.c_str());
    return resp->success;
}

//  Wait helpers

bool MujocoInterface::wait_for_connection(double timeout_sec) const {
    RCLCPP_INFO(get_logger(), "Waiting for odom connection...");
    using Clock = std::chrono::steady_clock;
    auto deadline = Clock::now() + std::chrono::duration<double>(timeout_sec);
    while (!connected_.load()) {
        if (Clock::now() > deadline) {
            RCLCPP_ERROR(get_logger(), "No odom received — launch MuJoCo first");
            return false;
        }
        std::this_thread::sleep_for(100ms);
    }
    auto s = get_state();
    RCLCPP_INFO(get_logger(), "Connected! pos=[%.2f, %.2f, %.2f]",
                s.pos.x(), s.pos.y(), s.pos.z());
    return true;
}

bool MujocoInterface::wait_for_pd_convergence(
    const Vec3& target, double settle_dist,
    double settle_time, double timeout_sec) const
{
    RCLCPP_INFO(get_logger(),
        "Waiting PD convergence to [%.2f,%.2f,%.2f] (dist<%.2f m, %.1fs, timeout=%.0fs)",
        target.x(), target.y(), target.z(), settle_dist, settle_time, timeout_sec);

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    auto deadline = t0 + std::chrono::duration<double>(timeout_sec);
    std::chrono::time_point<Clock> t_inside;
    bool inside = false;

    while (Clock::now() < deadline) {
        double dist = (get_state().pos - target).norm();
        if (dist < settle_dist) {
            if (!inside) { t_inside = Clock::now(); inside = true; }
            double dwell = std::chrono::duration<double>(Clock::now() - t_inside).count();
            if (dwell >= settle_time) {
                RCLCPP_INFO(get_logger(), "PD converged (dist=%.3f m)", dist);
                return true;
            }
        } else {
            inside = false;
        }
        std::this_thread::sleep_for(50ms);
    }
    RCLCPP_WARN(get_logger(), "PD convergence timeout — continuing anyway");
    return false;
}

//  PD position hold (background thread)

void MujocoInterface::start_pd_hold(const Vec3& target, double mass, double g) {
    start_pd_hold(target, mass, g, PdGains{});
}

void MujocoInterface::start_pd_hold(
    const Vec3& target, double mass, double g, const PdGains& gains)
{
    stop_pd_hold();  // ensure previous thread is gone
    pd_active_.store(true);
    pd_thread_ = std::thread(&MujocoInterface::pd_loop_, this,
                             target, mass, g, gains);
}

void MujocoInterface::stop_pd_hold() {
    pd_active_.store(false);
    if (pd_thread_.joinable()) pd_thread_.join();
}

void MujocoInterface::pd_loop_(
    Vec3 target, double mass, double g, PdGains gains)
{
    RCLCPP_INFO(get_logger(), "[PD] Flying to [%.2f,%.2f,%.2f] and holding...",
                target.x(), target.y(), target.z());

    while (pd_active_.load()) {
        auto st = get_state();
        Vec3 err = target - st.pos;

        // Desired acceleration (world frame) — vectorised
        Vec3 a_des;
        a_des.x() = gains.kp_xy * err.x() - gains.kd_xy * st.vel.x();
        a_des.y() = gains.kp_xy * err.y() - gains.kd_xy * st.vel.y();
        a_des.z() = gains.kp_z  * err.z() - gains.kd_z  * st.vel.z() + g;

        // Rotation matrix from quaternion
        const double qw = st.quat(0), qx = st.quat(1),
                     qy = st.quat(2), qz = st.quat(3);
        Mat3 R;
        R << 1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy),
             2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx),
             2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy);

        double thrust = std::clamp(mass * a_des.dot(R.col(2)), 0.0, 60.0);

        double psi = std::atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz));
        double az  = std::max(a_des.z(), 0.1);
        double cp  = std::cos(psi), sp = std::sin(psi);

        double pitch_des = std::clamp((a_des.x()*cp + a_des.y()*sp) / az, -0.5, 0.5);
        double roll_des  = std::clamp((a_des.x()*sp - a_des.y()*cp) / az, -0.5, 0.5);

        double pitch_cur = std::asin(std::clamp(R(0,2), -1.0, 1.0));
        double roll_cur  = std::atan2(-R(1,2), R(2,2));
        double yaw_err   = std::fmod(0.0 - psi + M_PI, 2*M_PI) - M_PI;

        double wx = gains.kp_att * (roll_des  - roll_cur);
        double wy = gains.kp_att * (pitch_des - pitch_cur);
        double wz = gains.kp_yaw * yaw_err;

        send_cmd(thrust, wx, wy, wz);
        std::this_thread::sleep_for(10ms);
    }
    RCLCPP_INFO(get_logger(), "[PD] Stopped — main controller takes over");
}

}  // namespace quadrotor_mpc
