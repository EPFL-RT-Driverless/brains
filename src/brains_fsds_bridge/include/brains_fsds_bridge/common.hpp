#ifndef BRAINS_FSDS_BRIDGE_COMMON_HPP
#include "common/common_utils/StrictMode.hpp"
// STRICT_MODE_OFF // todo what does this do?
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
// STRICT_MODE_ON
#include "rclcpp/rclcpp.hpp"
#include "vehicles/car/api/CarRpcLibClient.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace brains_fsds_bridge {
template <typename T>
inline T rad2deg(const T radians)
{
    return (radians / M_PI) * 180.0;
}

template <typename T>
inline T deg2rad(const T degrees)
{
    return (degrees / 180.0) * M_PI;
}

template <typename T>
inline T wrap_to_pi(T radians)
{
    int m = (int)(radians / (2 * M_PI));
    radians = radians - m * 2 * M_PI;
    if (radians > M_PI)
        radians -= 2.0 * M_PI;
    else if (radians < -M_PI)
        radians += 2.0 * M_PI;
    return radians;
}

template <typename T>
inline void wrap_to_pi_inplace(T& a)
{
    a = wrap_to_pi(a);
}

template <class T>
inline T angular_dist(T from, T to)
{
    wrap_to_pi_inplace(from);
    wrap_to_pi_inplace(to);
    T d = to - from;
    if (d > M_PI)
        d -= 2. * M_PI;
    else if (d < -M_PI)
        d += 2. * M_PI;
    return d;
}
rclcpp::Time make_ts(uint64_t unreal_ts)
{
    // unreal timestamp is a unix nanosecond timestamp just like ros.
    // We can do direct translation as long as ros is not running in simulated time mode.
    return rclcpp::Time(unreal_ts);
}

typedef std::chrono::duration<float> Duration;

class Statistics {
private:
    inline static double _time_elapsed = 1.0;
    std::string m_name;
    uint m_msg_count;
    std::vector<float> m_duration_history {};

public:
    Statistics() = default; // Having the default constructor is necessary for the workspace to build!
    Statistics(const std::string name)
        : m_name(name) {};

    std::string summary() const
    {
        std::stringstream ss;
        if (m_msg_count != 0) {
            double ros_msg_hz = _time_elapsed == 0.0f ? 1 : m_msg_count / _time_elapsed;
            ss << m_name << " msgs/s: " << ros_msg_hz << "\n";
        }
        if (!m_duration_history.empty()) {
            float max_latency = *std::max_element(m_duration_history.begin(), m_duration_history.end());
            ss << m_name << " rpc max latency: " << max_latency << "us\n";
        }
        return ss.str();
    }

    void add_duration_recording(const Duration& duration)
    {
        m_duration_history.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
    }

    void increment_msg_count()
    {
        ++m_msg_count;
    }

    // There is probably a better way of resetting the vector which prevents allocating all the space needed for its elements again
    void reset()
    {
        m_msg_count = 0;
        m_duration_history = {};
    }

    uint msg_count() const
    {
        return m_msg_count;
    }

    class AutoTimer {
    public:
        AutoTimer(Statistics& statistics)
            : _statistics(statistics)
        {
            _start = std::chrono::high_resolution_clock::now();
        };

        ~AutoTimer()
        {
            _end = std::chrono::high_resolution_clock::now();
            _statistics.add_duration_recording(std::move(_end - _start));
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> _start, _end;
        Statistics& _statistics;
    };
};

class BaseClient : public rclcpp::Node {
protected:
    std::string host_ip;
    double timeout;
    std::unique_ptr<msr::airlib::CarRpcLibClient> rpc_client;
    std::recursive_mutex rpc_mutex;
    std::shared_ptr<rclcpp::TimerBase> statistics_timer;

    virtual void print_statistics() = 0;
    virtual void reset_statistics() = 0;

    void setup_airsim()
    {
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
        : Node(name)
    {
        this->host_ip = this->declare_parameter<std::string>("host_ip", "localhost");
        this->timeout = this->declare_parameter<double>("timeout", 1.0);
        auto statistics_freq = this->declare_parameter<double>("statistics_freq", 1.0);

        this->setup_airsim();
        this->create_wall_timer(std::chrono::duration<double>(1.0 / statistics_freq), [this]() {
            this->print_statistics();
            this->reset_statistics();
        });
    }

    // TODO: method wrapper that locks the rpc mutex before calling the method.
    // It also instantiates the AutoTimer object which will record the duration of the method call.
};
} // namespace brains_fsds_bridge
#endif // BRAINS_FSDS_BRIDGE_COMMON_HPP
