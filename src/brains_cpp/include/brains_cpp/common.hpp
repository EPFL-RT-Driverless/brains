#ifndef BRAINS_FSDS_BRIDGE_COMMON_HPP

#include "common/AirSimSettings.hpp"
#include "common/common_utils/StrictMode.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rpc/rpc_error.h"
#include "vehicles/car/api/CarRpcLibClient.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace brains_fsds_bridge {
    typedef std::chrono::duration<float> Duration;

    class Statistics {
    private:
        inline static double _time_elapsed = 1.0;
        std::string m_name;
        uint m_msg_count;
        std::vector<float> m_duration_history{};

    public:
        Statistics() = default;  // Having the default constructor is necessary for the workspace to build!
        Statistics(const std::string name)
            : m_name(name){};

        std::string summary() const {
            std::stringstream ss;
            if (m_msg_count != 0) {
                double ros_msg_hz = _time_elapsed == 0.0f ? 1 : m_msg_count / _time_elapsed;
                ss << m_name << " msgs/s: " << ros_msg_hz << "\n";
            }
            if (!m_duration_history.empty()) {
                float max_latency = *std::max_element(m_duration_history.begin(), m_duration_history.end());
                ss << m_name << " rpc max latency: " << max_latency / 1000.0 << "ms\n";
            }
            return ss.str();
        }

        void add_duration_recording(const Duration& duration) {
            m_duration_history.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
        }

        void increment_msg_count() {
            ++m_msg_count;
        }

        // There is probably a better way of resetting the vector which prevents allocating all the space needed for its elements again
        void reset() {
            m_msg_count = 0;
            m_duration_history = {};
        }

        uint msg_count() const {
            return m_msg_count;
        }

        class AutoTimer {
        public:
            AutoTimer(Statistics& statistics)
                : _statistics(statistics) {
                _start = std::chrono::high_resolution_clock::now();
            };

            ~AutoTimer() {
                _end = std::chrono::high_resolution_clock::now();
                _statistics.add_duration_recording(std::move(_end - _start));
            }

        private:
            std::chrono::time_point<std::chrono::high_resolution_clock> _start, _end;
            Statistics& _statistics;
        };
    };

    class BaseClient : public rclcpp::Node {
    private:
        rclcpp::Time* first_timeout;
        std::string host_ip;
        std::recursive_mutex rpc_mutex;
        double timeout;
        std::shared_ptr<rclcpp::TimerBase> statistics_timer;

    protected:
        std::unique_ptr<msr::airlib::CarRpcLibClient> rpc_client;

        virtual void print_statistics() = 0;
        virtual void reset_statistics() = 0;

        void setup_airsim() {
            try {
                rpc_client = std::make_unique<msr::airlib::CarRpcLibClient>(host_ip, RpcLibPort, timeout);
                RCLCPP_INFO(this->get_logger(), "Waiting for connection");
                rpc_client->confirmConnection(timeout);
                RCLCPP_INFO(this->get_logger(), "Connected to the simulator!");
            } catch (rpc::rpc_error& e) {
                std::stringstream error_msg;
                error_msg << "Exception raised by the API, something went wrong." << std::endl
                          << e.get_error().as<std::string>();
                RCLCPP_FATAL(this->get_logger(), "%s", error_msg.str().c_str());
                // throw std::runtime_error(error_msg.str());
            }
        }

    public:
        BaseClient(std::string name)
            : Node(name), first_timeout(nullptr) {
            // declare parameters
            this->host_ip = this->declare_parameter<std::string>("host_ip", "localhost");
            this->timeout = this->declare_parameter<double>("timeout", 1.0);
            auto statistics_freq = this->declare_parameter<double>("statistics_freq", 1.0);

            // connect RPC client to FSDS
            this->setup_airsim();
            // get settings from FSDS
            std::string settings_text;
            this->rpc_call_wrapper(
                    [this, &settings_text]() {
                        settings_text = rpc_client->getSettingsString();
                    },
                    "getSettingsString");
            try {
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
            // setup statistics timer
            statistics_timer = this->create_wall_timer(std::chrono::duration<double>(1.0 / statistics_freq), [this]() {
                this->print_statistics();
                this->reset_statistics();
            });
        }

        void rpc_call_wrapper(std::function<void()> to_call, std::string method_name, Statistics* stats = nullptr) {
            try {
                std::lock_guard<std::recursive_mutex> lock(this->rpc_mutex);
                if (stats != nullptr) {
                    Statistics::AutoTimer timer(*stats);
                    to_call();
                } else {
                    to_call();
                }
                delete first_timeout;
                first_timeout = nullptr;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Exception in %s: %s", method_name.c_str(), e.what());
                // TODO try static cast to rpc::rpc_error or check the string e.what to see if it's a timeout
                bool is_timeout = true;
                if (is_timeout) {
                    if (first_timeout == nullptr) {
                        first_timeout = new rclcpp::Time(now());
                    } else {
                        //rclcpp::Time now_ = now();
                        //rclcpp::Duration diff = now_ - *first_timeout;
                        //RCLCPP_ERROR(this->get_logger(), "diff %f", diff.seconds());
                        if ((this->now() - *this->first_timeout).seconds() > 1.0) {
                            try {
                                this->setup_airsim();
                            } catch (const std::exception& e) {
                                RCLCPP_ERROR(this->get_logger(), "Could not reconnect to airsim: %s", e.what());
                                return;
                            }
                        }
                    }
                }
                return;
            }
        }
    };
}  // namespace brains_fsds_bridge
#endif  // BRAINS_FSDS_BRIDGE_COMMON_HPP
