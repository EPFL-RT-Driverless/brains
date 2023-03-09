#include "brains_custom_interfaces/msg/car_controls.hpp"
#include "brains_custom_interfaces/msg/car_state.hpp"
#include "brains_custom_interfaces/msg/velocity_estimation.hpp"
#include "brains_custom_interfaces/msg/wss_data.hpp"
#include "brains_custom_interfaces/srv/enable_api_fsds.hpp"
#include "brains_custom_interfaces/srv/map_name_fsds.hpp"
#include "brains_custom_interfaces/srv/restart_fsds.hpp"
#include "brains_cpp/common.hpp"
#include "common/AirSimSettings.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define velocity_estimation_uncertainty_percentage "velocity_estimation_covariance"

template <typename T>
using PublisherMap = std::unordered_map<std::string, std::shared_ptr<rclcpp::Publisher<T>>>;
using TimerMap = std::unordered_map<std::string, std::shared_ptr<rclcpp::TimerBase>>;
using StatisticsMap = std::unordered_map<std::string, brains_fsds_bridge::Statistics>;

class MainNode : public brains_fsds_bridge::BaseClient {
private:
    std::string vehicle_name;
    std::random_device rd;
    std::mt19937 gen = std::mt19937(127);

    // publishers
    std::shared_ptr<rclcpp::Publisher<brains_custom_interfaces::msg::CarState>> car_state_pub;
    std::shared_ptr<rclcpp::Publisher<brains_custom_interfaces::msg::VelocityEstimation>> velocity_estimation_pub;
    std::shared_ptr<rclcpp::Publisher<brains_custom_interfaces::msg::WssData>> wss_pub;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>> gss_pub;
    PublisherMap<sensor_msgs::msg::Imu> imu_pubs;
    PublisherMap<sensor_msgs::msg::NavSatFix> gps_pubs;
    // timers
    std::shared_ptr<rclcpp::TimerBase> car_state_timer;
    std::shared_ptr<rclcpp::TimerBase> wss_timer;
    std::shared_ptr<rclcpp::TimerBase> gss_timer;
    TimerMap imu_timers;
    TimerMap gps_timers;
    // subscribers
    std::shared_ptr<rclcpp::Subscription<brains_custom_interfaces::msg::CarControls>> car_controls_sub;
    // services
    std::shared_ptr<rclcpp::Service<brains_custom_interfaces::srv::RestartFSDS>> restart_srv;
    std::shared_ptr<rclcpp::Service<brains_custom_interfaces::srv::MapNameFSDS>> map_name_srv;
    std::shared_ptr<rclcpp::Service<brains_custom_interfaces::srv::EnableApiFSDS>> enable_api_srv;
    // statistics
    brains_fsds_bridge::Statistics car_state_statistics;
    brains_fsds_bridge::Statistics car_controls_statistics;
    brains_fsds_bridge::Statistics wss_statistics;
    brains_fsds_bridge::Statistics gss_statistics;
    StatisticsMap imu_statistics;
    StatisticsMap gps_statistics;

    void print_statistics() override {
        std::stringstream dbg_msg;
        dbg_msg << "--------- brains_fsds_bridge statistics ---------" << std::endl;
        dbg_msg << car_state_statistics.summary() << std::endl;
        dbg_msg << car_controls_statistics.summary() << std::endl;
        dbg_msg << gss_statistics.summary() << std::endl;
        if (wss_timer) {
            dbg_msg << wss_statistics.summary() << std::endl;
        }
        for (auto const& s : gps_statistics) {
            dbg_msg << s.second.summary() << std::endl;
        }
        for (auto const& s : imu_statistics) {
            dbg_msg << s.second.summary() << std::endl;
        }
        dbg_msg << "------------------------------------------" << std::endl;
        RCLCPP_INFO(this->get_logger(), "%s", dbg_msg.str().c_str());
    }

    void reset_statistics() override {
        car_state_statistics.reset();
        car_controls_statistics.reset();
        if (wss_timer) {
            wss_statistics.reset();
        }
        for (auto& s : gps_statistics) {
            s.second.reset();
        }
        for (auto& s : imu_statistics) {
            s.second.reset();
        }
        gss_statistics.reset();
    }

    void car_state_callback() {
        // TODO: replace the following block with a templated method in BaseClient
        msr::airlib::Kinematics::State state;
        rpc_call_wrapper(
                [this, &state]() {
                    state = this->rpc_client->simGetGroundTruthKinematics(vehicle_name);
                },
                "car_state_callback",
                &car_state_statistics);

        double x = state.pose.orientation.x(), y = state.pose.orientation.y(), z = state.pose.orientation.z(), w = state.pose.orientation.w();
        brains_custom_interfaces::msg::CarState car_state_msg;
        car_state_msg.header.stamp = this->now();
        // car_state_msg.header.frame_id = "map"; // TODO check this
        car_state_msg.x = state.pose.position[0];
        car_state_msg.y = state.pose.position[1];
        car_state_msg.phi = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        car_state_msg.v_x = state.twist.linear.x() * cos(car_state_msg.phi) + state.twist.linear.y() * sin(car_state_msg.phi);
        car_state_msg.v_y = state.twist.linear.x() * -sin(car_state_msg.phi) + state.twist.linear.y() * cos(car_state_msg.phi);
        car_state_msg.r = state.twist.angular.z();
        car_state_pub->publish(car_state_msg);
        car_state_statistics.increment_msg_count();

        brains_custom_interfaces::msg::VelocityEstimation velocity_estimation_msg;
        velocity_estimation_msg.header.stamp = this->now();
        // velocity_estimation_msg.header.frame_id = "map"; // TODO check this
        velocity_estimation_msg.v_x = car_state_msg.v_x;
        velocity_estimation_msg.v_y = car_state_msg.v_y;
        velocity_estimation_msg.r = car_state_msg.r;

        double err_percentage = this->get_parameter(velocity_estimation_uncertainty_percentage).as_double();
        velocity_estimation_msg.v_x += err_percentage * velocity_estimation_msg.v_x * std::gamma_distribution<double>(0.0)(rd);
        velocity_estimation_msg.v_y += err_percentage * velocity_estimation_msg.v_y * std::gamma_distribution<double>(0.0)(rd);
        velocity_estimation_msg.r += err_percentage * velocity_estimation_msg.r * std::gamma_distribution<double>(0.0)(rd);
        velocity_estimation_pub->publish(velocity_estimation_msg);
    }

    void car_controls_callback(const brains_custom_interfaces::msg::CarControls::SharedPtr msg) {
        msr::airlib::CarApiBase::CarControls controls;
        msg->throttle = std::min(std::max(msg->throttle, -1.0), 1.0);
        if (msg->throttle >= 0.0) {
            controls.throttle = msg->throttle;
            controls.brake = 0.0;
        } else {
            controls.brake = msg->throttle;
            controls.throttle = 0.0;
        }
        controls.steering = -std::min(std::max(msg->steering, -1.0), 1.0);

        rpc_call_wrapper(
                [this, &controls]() {
                    this->rpc_client->setCarControls(controls, vehicle_name);
                },
                "car_state_callback");
    }

    void restart_callback(const brains_custom_interfaces::srv::RestartFSDS::Request::SharedPtr req,
                          brains_custom_interfaces::srv::RestartFSDS::Response::SharedPtr res) {
        rpc_call_wrapper(
                [this]() {
                    this->rpc_client->restart();
                    // sleep for 1 second (this->timeout)
                    rclcpp::sleep_for(std::chrono::seconds(1));
                    this->setup_airsim();
                },
                "restart_callback");
    }

    void map_name_callback(const brains_custom_interfaces::srv::MapNameFSDS::Request::SharedPtr req,
                           brains_custom_interfaces::srv::MapNameFSDS::Response::SharedPtr res) {
        rpc_call_wrapper(
                [this, res]() {
                    res->map_name = this->rpc_client->getMap();
                },
                "map_name_callback");
    }

    void enable_api_callback(const brains_custom_interfaces::srv::EnableApiFSDS::Request::SharedPtr req,
                             brains_custom_interfaces::srv::EnableApiFSDS::Response::SharedPtr res) {
        // TODO: replace the following block with a templated method in BaseClient
        rpc_call_wrapper(
                [this, req]() {
                    this->rpc_client->enableApiControl(req->enabled, vehicle_name);
                },
                "enable_api_callback");
    }

    void wss_callback() {
        // TODO: replace the following block with a templated method in BaseClient
        msr::airlib::WheelStates data;
        rpc_call_wrapper(
                [this, &data]() {
                    data = this->rpc_client->simGetWheelStates(vehicle_name);
                },
                "car_state_callback",
                &wss_statistics);

        brains_custom_interfaces::msg::WssData wss_msg;

        wss_msg.header.stamp = this->now();

        wss_msg.fl_rpm = data.fl.rpm;
        wss_msg.fr_rpm = data.fr.rpm;
        wss_msg.rl_rpm = data.rl.rpm;
        wss_msg.rr_rpm = data.rr.rpm;

        wss_msg.steering_angle = data.fl.steering_angle;

        wss_pub->publish(wss_msg);
        wss_statistics.increment_msg_count();
    }

    void imu_callback(std::string imu_name) {
        // TODO: replace the following block with a templated method in BaseClient
        msr::airlib::ImuBase::Output data;
        rpc_call_wrapper(
                [this, &data, imu_name]() {
                    data = this->rpc_client->getImuData(imu_name, vehicle_name);
                },
                "imu_callback",
                &(imu_statistics[imu_name]));

        sensor_msgs::msg::Imu imu_msg;
        imu_msg.header.stamp = this->now();
        // imu_msg->header.frame_id = "imu_" + sensor_name;  // TODO: add frame_id to airsim
        imu_msg.orientation.x = data.orientation.x();
        imu_msg.orientation.y = data.orientation.y();
        imu_msg.orientation.z = data.orientation.z();
        imu_msg.orientation.w = data.orientation.w();
        imu_msg.angular_velocity.x = data.angular_velocity.x();
        imu_msg.angular_velocity.y = data.angular_velocity.y();
        imu_msg.angular_velocity.z = data.angular_velocity.z();
        imu_msg.linear_acceleration.x = data.linear_acceleration.x();
        imu_msg.linear_acceleration.y = data.linear_acceleration.y();
        imu_msg.linear_acceleration.z = data.linear_acceleration.z();
        imu_msg.angular_velocity_covariance[0] = data.sigma_arw * data.sigma_arw;
        imu_msg.angular_velocity_covariance[4] = data.sigma_arw * data.sigma_arw;
        imu_msg.angular_velocity_covariance[8] = data.sigma_arw * data.sigma_arw;
        imu_msg.linear_acceleration_covariance[0] = data.sigma_vrw * data.sigma_vrw;
        imu_msg.linear_acceleration_covariance[4] = data.sigma_vrw * data.sigma_vrw;
        imu_msg.linear_acceleration_covariance[8] = data.sigma_vrw * data.sigma_vrw;
        this->imu_pubs[imu_name]->publish(imu_msg);
        imu_statistics[imu_name].increment_msg_count();
    }

    void gps_callback(std::string gps_name) {
        // TODO: replace the following block with a templated method in BaseClient
        msr::airlib::GpsBase::Output data;
        rpc_call_wrapper(
                [this, &data, gps_name]() {
                    data = this->rpc_client->getGpsData(gps_name, vehicle_name);
                },
                "gps_callback",
                &(gps_statistics[gps_name]));

        sensor_msgs::msg::NavSatFix gps_msg;
        // gps_msg.header.frame_id = "fsds/" + vehicle_name;  // TODO: add frame_id to airsim
        msr::airlib::GeoPoint gps_location = data.gnss.geo_point;
        msr::airlib::GpsBase::GnssReport gnss_gps_report = data.gnss;
        gps_msg.header.stamp = this->now();
        gps_msg.latitude = gps_location.latitude;
        gps_msg.longitude = gps_location.longitude;
        gps_msg.altitude = gps_location.altitude;
        gps_msg.position_covariance[0] = gnss_gps_report.eph * gnss_gps_report.eph;
        gps_msg.position_covariance[4] = gnss_gps_report.eph * gnss_gps_report.eph;
        gps_msg.position_covariance[8] = gnss_gps_report.epv * gnss_gps_report.epv;
        this->gps_pubs[gps_name]->publish(gps_msg);
        this->gps_statistics[gps_name].increment_msg_count();
    }

    void gss_callback() {
        // TODO: replace the following block with a templated method in BaseClient
        msr::airlib::GSSSimple::Output data;
        rpc_call_wrapper(
                [this, &data]() {
                    data = this->rpc_client->getGroundSpeedSensorData(vehicle_name);
                },
                "gss_callback",
                &gss_statistics);

        geometry_msgs::msg::TwistWithCovarianceStamped gss_msg;
        gss_msg.header.stamp = this->now();
        gss_msg.twist.twist.angular.x = data.angular_velocity.x();
        gss_msg.twist.twist.angular.y = data.angular_velocity.y();
        gss_msg.twist.twist.angular.z = data.angular_velocity.z();
        gss_msg.twist.twist.linear.x = data.linear_velocity.x();
        gss_msg.twist.twist.linear.y = data.linear_velocity.y();
        gss_msg.twist.twist.linear.z = data.linear_velocity.z();
        // The 0.1 covariances for everything were just guessed, don't assume theese are correct
        gss_msg.twist.covariance[0] = 0.1;
        gss_msg.twist.covariance[7] = 0.1;
        gss_msg.twist.covariance[14] = 0.1;
        gss_msg.twist.covariance[21] = 0.1;
        gss_msg.twist.covariance[28] = 0.1;
        gss_msg.twist.covariance[35] = 0.1;
        this->gss_pub->publish(gss_msg);
        this->gss_statistics.increment_msg_count();
    }

public:
    MainNode()
        : BaseClient("main_node"), car_state_statistics("car_state"), car_controls_statistics("car_controls"),
          gss_statistics("gss"), wss_statistics("wss") {
        // declare parameters
        auto manual_mode = this->declare_parameter<bool>("manual_mode", false);
        this->declare_parameter<double>(velocity_estimation_uncertainty_percentage, 0.1);
        auto car_state_freq = this->declare_parameter<double>("car_state_freq", 100.0);
        auto wss_freq = this->declare_parameter<double>("wss_freq", 100.0);
        auto gss_freq = this->declare_parameter<double>("gss_freq", 100.0);
        auto imu_freqs = this->declare_parameter<std::vector<double>>("imu_freqs", {100.0});
        auto gps_freqs = this->declare_parameter<std::vector<double>>("gps_freqs", {10.0});
        // create what exists by default
        car_state_pub = this->create_publisher<brains_custom_interfaces::msg::CarState>("/fsds/car_state", 10);
        velocity_estimation_pub = this->create_publisher<brains_custom_interfaces::msg::VelocityEstimation>(
                "/fsds/velocity_estimation", 10);
        car_state_timer = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / car_state_freq),
                std::bind(&MainNode::car_state_callback, this));
        if (wss_freq > 0.0) {
            wss_pub = this->create_publisher<brains_custom_interfaces::msg::WssData>("/fsds/wss", 10);
            wss_timer = this->create_wall_timer(
                    std::chrono::duration<double>(1.0 / wss_freq),
                    std::bind(&MainNode::wss_callback, this));
        }

        car_controls_sub = this->create_subscription<brains_custom_interfaces::msg::CarControls>(
                "/fsds/car_controls", 10, std::bind(&MainNode::car_controls_callback, this, std::placeholders::_1));
        restart_srv = this->create_service<brains_custom_interfaces::srv::RestartFSDS>(
                "/fsds/restart",
                std::bind(&MainNode::restart_callback, this, std::placeholders::_1, std::placeholders::_2));
        map_name_srv = this->create_service<brains_custom_interfaces::srv::MapNameFSDS>(
                "/fsds/map_name",
                std::bind(&MainNode::map_name_callback, this, std::placeholders::_1, std::placeholders::_2));
        enable_api_srv = this->create_service<brains_custom_interfaces::srv::EnableApiFSDS>(
                "/fsds/enable_api",
                std::bind(&MainNode::enable_api_callback, this, std::placeholders::_1, std::placeholders::_2));

        // parse the AirSim Settings to check the numbers of sensors to create
        for (const auto& curr_vehicle_elem : msr::airlib::AirSimSettings::singleton().vehicles) {
            vehicle_name = curr_vehicle_elem.first;
            auto& vehicle_setting = curr_vehicle_elem.second;

            for (const auto& sensor : vehicle_setting->sensors) {
                std::string sensor_name = sensor.first;
                auto& sensor_setting = sensor.second;
                if (sensor_setting->enabled) {
                    switch (sensor_setting->sensor_type) {
                        case msr::airlib::SensorBase::SensorType::Imu: {
                            try {

                                imu_pubs[sensor_name] = this->create_publisher<sensor_msgs::msg::Imu>(
                                        "/fsds/imu/" + sensor_name, 10);
                                auto imu_timer = this->create_wall_timer(
                                        std::chrono::duration<double>(1.0 / imu_freqs.at(imu_pubs.size() - 1)),
                                        [this, sensor_name]() { this->imu_callback(sensor_name); });
                                imu_timers[sensor_name] = imu_timer;
                                imu_statistics[sensor_name] = brains_fsds_bridge::Statistics(sensor_name);
                            } catch (std::out_of_range& ex) {
                                RCLCPP_FATAL(this->get_logger(),
                                             "There are not enough specified imu_freqs (only %lu) for the number of IMUs in the settings.json file (%lu)",
                                             imu_freqs.size(), imu_pubs.size());
                                throw ex;
                            }
                            break;
                        }
                        case msr::airlib::SensorBase::SensorType::Gps: {
                            try {

                                gps_pubs[sensor_name] = this->create_publisher<sensor_msgs::msg::NavSatFix>(
                                        "/fsds/gps/" + sensor_name, 10);
                                auto gps_timer = this->create_wall_timer(
                                        std::chrono::duration<double>(1.0 / gps_freqs.at(gps_pubs.size() - 1)),
                                        [this, sensor_name]() { this->gps_callback(sensor_name); });
                                gps_timers[sensor_name] = gps_timer;
                                gps_statistics[sensor_name] = brains_fsds_bridge::Statistics(sensor_name);
                            } catch (std::out_of_range& ex) {
                                RCLCPP_FATAL(this->get_logger(),
                                             "There are not enough specified gps_freqs (only %lu) for the number of GPSs in the settings.json file (%lu)",
                                             gps_freqs.size(), gps_pubs.size());
                                throw ex;
                            }
                            break;
                        }
                        case msr::airlib::SensorBase::SensorType::GSS: {
                            if (!this->gss_pub) {
                                if (gss_freq <= 0.0) {
                                    throw std::invalid_argument("gss_freq must be > 0.0");
                                }
                                gss_pub = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
                                        "/fsds/gss", 10);
                                gss_timer = this->create_wall_timer(
                                        std::chrono::duration<double>(1.0 / gss_freq),
                                        std::bind(&MainNode::gss_callback, this));
                            }
                            break;
                        }
                        default:
                            break;
                    }
                }
            }
        }
    }

    ~MainNode() = default;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MainNode>());
    rclcpp::shutdown();
    return 0;
}
