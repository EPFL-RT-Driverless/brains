#include "brains_fsds_bridge/common.hpp"
#include "common/AirSimSettings.hpp"
#include "common/ImageCaptureBase.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class CameraNode : public brains_fsds_bridge::BaseClient {
private:
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub;
    std::shared_ptr<rclcpp::TimerBase> image_timer;
    brains_fsds_bridge::Statistics image_statistics;
    void image_callback(std::string camera_name, std::string vehicle_name)
    {
        std::vector<msr::airlib::ImageCaptureBase::ImageRequest> reqs({ msr::airlib::ImageCaptureBase::ImageRequest(camera_name, msr::airlib::ImageCaptureBase::ImageType::Scene, false, false) });
        std::vector<msr::airlib::ImageCaptureBase::ImageResponse> img_responses;
        try {
            std::lock_guard<std::recursive_mutex> guard(rpc_mutex);
            brains_fsds_bridge::Statistics::AutoTimer timer(image_statistics);
            img_responses = rpc_client->simGetImages(reqs, vehicle_name);
        } catch (std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in image_callback: %s", e.what());
            return;
        }
        // if a render request failed for whatever reason, this img will be empty.
        if (img_responses.size() == 0 || img_responses[0].time_stamp == 0) {
            RCLCPP_ERROR(this->get_logger(), "No image received from AirSim");
            return;
        }
        msr::airlib::ImageCaptureBase::ImageResponse img_response = img_responses.front();
        sensor_msgs::msg::Image img_msg;
        img_msg.data = img_response.image_data_uint8;
        img_msg.height = img_response.height;
        img_msg.width = img_response.width;
        img_msg.step = img_response.width * 3; // image_width * num_bytes
        img_msg.encoding = "bgr8";
        img_msg.is_bigendian = 0;
        img_msg.header.stamp = this->now();
        // img_msg.header.frame_id = "/fsds/" + camera_name;  // TODO: figure out how to set this
        image_pub->publish(img_msg);
        image_statistics.increment_msg_count();
    }
    void print_statistics() override
    {
        std::stringstream dbg_msg;
        dbg_msg << "--------- brains_fsds_bridge statistics ---------" << std::endl;
        dbg_msg << image_statistics.summary() << std::endl;
        dbg_msg << "------------------------------------------" << std::endl;
        RCLCPP_DEBUG(this->get_logger(), "%s", dbg_msg.str().c_str());
    }
    void reset_statistics() override
    {
        image_statistics.reset();
    }

public:
    CameraNode()
        : BaseClient("camera_node")
    {
        auto camera_name = this->declare_parameter<std::string>("camera_name", "camera");
        auto framerate = this->declare_parameter<int>("framerate", 20);
        // load settings.json and check that camera_name exists. If not, print fatal error and exit.
        try {
            std::string settings_text = rpc_client->getSettingsString();
            msr::airlib::AirSimSettings::initializeSettings(settings_text);

            msr::airlib::AirSimSettings::singleton().load();
            for (const auto& warning : msr::airlib::AirSimSettings::singleton().warning_messages) {
                RCLCPP_WARN_STREAM(this->get_logger(), "Configuration warning: " << warning);
            }
            for (const auto& error : msr::airlib::AirSimSettings::singleton().error_messages) {
                RCLCPP_ERROR_STREAM(this->get_logger(), "Configuration error: " << error);
            }
        } catch (std::exception& ex) {
            throw std::invalid_argument(std::string("Failed loading settings.json.") + ex.what());
        }
        // parse it to check the numbers of sensors to create
        std::string vehicle_name;
        for (const auto& curr_vehicle_elem : msr::airlib::AirSimSettings::singleton().vehicles) {
            vehicle_name = curr_vehicle_elem.first;
            auto& vehicle_setting = curr_vehicle_elem.second;
            if (vehicle_setting->cameras.find(camera_name) == vehicle_setting->cameras.end()) {
                RCLCPP_FATAL(this->get_logger(),
                    "Camera %s not found for vehicle %s in settings.json. Please make some changes and restart FSDS.",
                    camera_name.c_str(), vehicle_name.c_str());
                rclcpp::shutdown();
                exit(1);
            }
        }
        image_statistics = brains_fsds_bridge::Statistics("image/" + camera_name);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>("/fsds/camera/" + camera_name, 10);
        image_timer = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / framerate),
            [this, camera_name, vehicle_name]() { this->image_callback(camera_name, vehicle_name); });
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();
    return 0;
}
