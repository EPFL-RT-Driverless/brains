import matplotlib.pyplot as plt
import numpy as np

from work.ve.common import print_stats_errs


def load_data():
    data = np.load(
        "/Users/tudoroancea/Developer/racing_team/brains/src/brains_python/work/datasets/data/ve_dataset/fsds_competition_1_15.0.npz"
    )
    print("available keys: ", data.files)
    return data


def mse_all():
    data = load_data()
    true_v_x = data["states"][:, 3]
    true_v_y = data["states"][:, 4]
    gss_v_x = data["gss_velocities"][:, 0]
    gss_v_y = data["gss_velocities"][:, 1]
    wheel_radius = 0.2
    wheel_base = 1.0
    wheels_v_x_fl = (
        data["wheel_speeds"][:, 0] * wheel_radius * 2 * np.pi / 60
        + 0.5 * wheel_base * data["states"][:, 5]
    )
    wheels_v_x_fr = (
        data["wheel_speeds"][:, 1] * wheel_radius * 2 * np.pi / 60
        - 0.5 * wheel_base * data["states"][:, 5]
    )
    wheels_v_x_rl = (
        data["wheel_speeds"][:, 2] * wheel_radius * 2 * np.pi / 60
        + 0.5 * wheel_base * data["states"][:, 5]
    )
    wheels_v_x_rr = (
        data["wheel_speeds"][:, 3] * wheel_radius * 2 * np.pi / 60
        - 0.5 * wheel_base * data["states"][:, 5]
    )
    # compute and print the MSE for each velocity component
    print_stats_errs(true_v_x, gss_v_x, "gss_v_x")
    print_stats_errs(true_v_y, gss_v_y, "gss_v_y")
    print_stats_errs(true_v_x, wheels_v_x_fl, "wheels_v_x_fl")
    print_stats_errs(true_v_x, wheels_v_x_fr, "wheels_v_x_fr")
    print_stats_errs(true_v_x, wheels_v_x_rl, "wheels_v_x_rl")
    print_stats_errs(true_v_x, wheels_v_x_rr, "wheels_v_x_rr")

    plt.subplot(2, 1, 1)
    plt.plot(true_v_x, label="true_v_x")
    plt.plot(gss_v_x, label="gss_v_x")
    plt.plot(wheels_v_x_fl, label="wheels_v_x_fl")
    plt.plot(wheels_v_x_fr, label="wheels_v_x_fr")
    plt.plot(wheels_v_x_rl, label="wheels_v_x_rl")
    plt.plot(wheels_v_x_rr, label="wheels_v_x_rr")
    plt.legend()
    plt.title(r"$v_x$")
    plt.subplot(2, 1, 2)
    plt.plot(true_v_y, label="true_v_y")
    plt.plot(gss_v_y, label="gss_v_y")
    plt.legend()
    plt.title(r"$v_y$")
    plt.show()


def uncertainty_accelerations():
    data = load_data()
    true_a_x = data["imu_linear_accelerations"][:, 0]
    true_a_y = data["imu_linear_accelerations"][:, 1]
    true_v_x = data["states"][:, 3]
    true_v_y = data["states"][:, 4]
    true_r = data["states"][:, 5]
    wheelbase = 1.0
    dt = 0.01
    estimated_a_x = (true_v_x[2:] - true_v_x[:-2]) / dt - true_r[1:-1] * true_v_y[1:-1]
    estimated_a_y = (true_v_y[2:] - true_v_y[:-2]) / dt + true_r[1:-1] * true_v_x[1:-1]
    errors_a_x = true_a_x[1:-1] - estimated_a_x
    errors_a_y = true_a_y[1:-1] - estimated_a_y
    print(
        f"error_a_x: mean = {np.mean(errors_a_x)}, std = {np.std(errors_a_x)}\n error_a_y: mean = {np.mean(errors_a_y)}, std = {np.std(errors_a_y)}"
    )


def uncertainty_yaw_rate():
    data = load_data()
    true_r = data["states"][:, 5]
    imu_r = data["imu_angular_velocities"][:, 2]
    errors_r = true_r - imu_r
    print(f"error_r: mean = {np.mean(errors_r)}, std = {np.std(errors_r)}")


if __name__ == "__main__":
    uncertainty_accelerations()
    uncertainty_yaw_rate()
    mse_all()
