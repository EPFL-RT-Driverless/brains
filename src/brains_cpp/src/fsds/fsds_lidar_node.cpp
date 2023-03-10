// Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
#include "brains_cpp/common.hpp"
#include "common/AirSimSettings.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class CameraNode : public BaseClient {
private:
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> lidar_pub;
    std::shared_ptr<rclcpp::TimerBase> lidar_timer;
    Statistics lidar_statistics;
    void lidar_callback(std::string lidar_name, std::string vehicle_name) {
        sensor_msgs::msg::PointCloud2 lidar_msg;
        msr::airlib::LidarData lidar_data;
        rpc_call_wrapper(
                [this, &lidar_data, &lidar_name, &vehicle_name]() {
                    lidar_data = rpc_client->getLidarData(lidar_name, vehicle_name);
                },
                "lidar_callback",
                &lidar_statistics);
        if (lidar_data.point_cloud.size() > 3) {
            lidar_msg.height = 1;
            lidar_msg.width = lidar_data.point_cloud.size() / 3;

            lidar_msg.fields.resize(3);
            lidar_msg.fields[0].name = "x";
            lidar_msg.fields[1].name = "y";
            lidar_msg.fields[2].name = "z";
            int offset = 0;

            for (size_t d = 0; d < lidar_msg.fields.size(); ++d, offset += 4) {
                lidar_msg.fields[d].offset = offset;
                lidar_msg.fields[d].datatype = sensor_msgs::msg::PointField::FLOAT32;
                lidar_msg.fields[d].count = 1;
            }

            lidar_msg.is_bigendian = false;
            lidar_msg.point_step = offset;  // 4 * num fields
            lidar_msg.row_step = lidar_msg.point_step * lidar_msg.width;

            lidar_msg.is_dense = true;  // todo
            std::vector<float> data_std = lidar_data.point_cloud;

            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&data_std[0]);
            std::vector<unsigned char> lidar_msg_data(bytes, bytes + sizeof(float) * data_std.size());
            lidar_msg.data = std::move(lidar_msg_data);
        }
        // lidar_msg.header.frame_id = "fsds/" + lidar_name;  // TODO: figure out how to set this
        lidar_msg.header.stamp = this->now();
        lidar_pub->publish(lidar_msg);
        lidar_statistics.increment_msg_count();
    }
    void print_statistics() override {
        std::stringstream dbg_msg;
        dbg_msg << "'n--------- brains_fsds_bridge statistics ---------\n";
        dbg_msg << lidar_statistics.summary();
        dbg_msg << "------------------------------------------";
        RCLCPP_INFO(this->get_logger(), "%s", dbg_msg.str().c_str());
    }
    void reset_statistics() override {
        lidar_statistics.reset();
    }

public:
    CameraNode()
        : BaseClient("lidar_node") {
        auto lidar_name = this->declare_parameter<std::string>("lidar_name", "lidar");
        // parse it to check the numbers of sensors to create
        std::string vehicle_name;
        double framerate = 0.0;
        for (const auto& curr_vehicle_elem : msr::airlib::AirSimSettings::singleton().vehicles) {
            vehicle_name = curr_vehicle_elem.first;
            auto& vehicle_setting = curr_vehicle_elem.second;
            auto it(vehicle_setting->sensors.find(lidar_name));
            if (it != vehicle_setting->sensors.end()) {
                if (it->second->enabled) {
                    framerate = static_cast<msr::airlib::AirSimSettings::LidarSetting*>(it->second.get())->horizontal_rotation_frequency;
                    RCLCPP_INFO(this->get_logger(),
                                "Lidar %s found for vehicle %s in settings.json. Framerate: %lf",
                                lidar_name.c_str(), vehicle_name.c_str(), framerate);
                    break;
                } else {
                    RCLCPP_FATAL(this->get_logger(),
                                 "Lidar %s is disabled in settings.json. Please make some changes and restart FSDS.",
                                 lidar_name.c_str());
                    rclcpp::shutdown();
                    exit(1);
                }
            } else {
                RCLCPP_FATAL(this->get_logger(),
                             "Lidar %s not found for vehicle %s in settings.json. Please make some changes and restart FSDS.",
                             lidar_name.c_str(), vehicle_name.c_str());
                rclcpp::shutdown();
                exit(1);
            }
        }

        lidar_statistics = Statistics("lidar/" + lidar_name);
        lidar_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fsds/lidar/" + lidar_name, 10);
        lidar_timer = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / framerate),
                [this, lidar_name, vehicle_name]() { this->lidar_callback(lidar_name, vehicle_name); });
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();
    return 0;
}
