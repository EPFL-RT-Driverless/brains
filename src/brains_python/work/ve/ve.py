import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from common import stats_errs, print_stats_errs
from scipy.stats.distributions import chi2

np.random.seed(127)


class VE0:
    """
    Estimates the vehicle state (v_x, v_y, r, a_x, a_y) with an EKF based on observations coming from several sensors:
    - INS (v_x, v_y, r, a_x, a_y)
    - GSS (v_x, v_y)
    - Wheel speeds (v_x)
    """

    nx = 5

    # state estimation mean and covariance
    mu: np.ndarray
    Sigma: np.ndarray

    # diagnostics of each sensor, used for outlier detection
    sensors = ["ins", "gss", "wheel_speeds"]
    diagnostics: dict
    mahalanobis_distances: dict
    chi2_95 = 5.991
    chi2_99 = 9.210

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        sampling_time: float,
    ):
        self.mu = initial_state
        self.Sigma = initial_covariance
        self.dt = sampling_time
        x = ca.SX.sym("x", self.nx)
        dt = ca.SX.sym("dt")
        f = ca.Function(
            "f",
            [x],
            [ca.vertcat(x[3] + x[2] * x[1], x[4] - x[2] * x[0], 0, 0, 0)],
        )
        self.f = f
        # find discretized dynamics g with RK4
        k1 = f(x)
        k2 = f(x + self.dt / 2 * k1)
        k3 = f(x + self.dt / 2 * k2)
        k4 = f(x + self.dt * k3)
        self.g = ca.Function(
            "g", [x, dt], [x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)]
        )
        self.G = ca.Function("G", [x, dt], [ca.jacobian(self.g(x, dt), x)])
        self.diagnostics = {sensor: 0.0 for sensor in self.sensors}
        self.mahalanobis_distances = {sensor: 0.0 for sensor in self.sensors}

    def predict(self, motion_covariance: np.ndarray, dt: float = None):
        if dt is None:
            dt = self.dt

        x = self.mu.copy()
        # Forward Euler discretization
        new_x = x + self.dt * np.array(
            [x[3] + x[2] * x[1], x[4] - x[2] * x[0], 0, 0, 0]
        )
        G = np.array(
            [
                [1, x[2] * dt, -x[1] * dt, dt, 0],
                [-x[2] * dt, 1, x[0] * dt, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        # RK4 discretization
        # new_x = self.g(x, dt).full().ravel()
        # G = self.G(x, dt).full()

        self.mu = new_x
        self.Sigma = G @ self.Sigma @ G.T + motion_covariance

    def update(
        self,
        measurements: dict[str, np.ndarray],
        measurement_covariances: dict[str, np.ndarray],
    ):
        assert measurements.keys() == measurement_covariances.keys()
        # assert that all the keys are in sensors
        assert measurements.keys() <= set(self.sensors)
        for sensor_name in measurements.keys():
            z = measurements[sensor_name]
            R = measurement_covariances[sensor_name]
            if sensor_name == "ins":
                zhat = self.mu
                H = np.eye(self.nx)
            elif sensor_name == "gss":
                zhat = self.mu[:2]
                H = np.hstack((np.eye(2), np.zeros((2, self.nx - 2))))
            elif sensor_name == "wheel_speeds":
                zhat = self.mu[0]
                H = np.array([[1, 0, 0, 0, 0]])
            else:
                raise ValueError(f"Unknown sensor {sensor_name}")

            # innovations
            r = z - zhat
            S = H @ self.Sigma @ H.T + R
            Sinv = np.linalg.inv(S)
            # Kalman gain
            K = self.Sigma @ H.T @ Sinv
            # compute diagnostics from mahalanobis distance
            self.mahalanobis_distances[sensor_name] = (
                r.T @ Sinv @ r if sensor_name != "wheel_speeds" else r**2 * Sinv[0, 0]
            )
            self.diagnostics[sensor_name] = 1 - min(
                self.mahalanobis_distances[sensor_name] / self.chi2_99, 1.0
            )
            if self.mahalanobis_distances[sensor_name] < self.chi2_99:
                # update state estimation
                self.mu = (
                    self.mu + K @ r
                    if sensor_name != "wheel_speeds"
                    else self.mu + K.ravel() * r
                )
                self.Sigma = (np.eye(self.nx) - K @ H) @ self.Sigma
            else:
                print(
                    f"Mahalanobis distance {self.mahalanobis_distances[sensor_name]} is too large"
                )


def main():
    dt = 0.01
    track_name = "fsds_competition_1"
    vmax = 10.0
    filename = f"{track_name}_{vmax}.npz"
    # noinspection PyTypeChecker
    data: np.lib.npyio.NpzFile = np.load(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../datasets/data/ve_dataset",
                filename,
            )
        )
    )
    print(data.files[:6])
    ve = VE0(
        initial_state=np.zeros(5),
        initial_covariance=np.zeros((5, 5)),
        sampling_time=dt,
    )
    runtimes = []
    estimated_velocities = []
    raw_ins_velocities = []
    true_velocities = []
    motion_covariance = np.diag([0.1, 0.1, 0.05, 14.0, 3.0]) ** 2
    diagnostics = {"ins": [], "gss": [], "wheel_speeds": []}
    mahalanobis_distances = {"ins": [], "gss": [], "wheel_speeds": []}
    for file_id in range(0, data["states"].shape[0], int(dt / 0.01)):
        true_velocities.append(data["states"][file_id][3:])
        sigma_v_x = 0.5
        sigma_v_y = 0.1
        sigma_r = 0.07
        sigma_a_x = 14.119
        sigma_a_y = 2.711
        ins_measurement = np.concatenate(
            (
                true_velocities[-1]
                + np.random.multivariate_normal(
                    np.zeros(3), np.diag([sigma_v_x, sigma_v_y, sigma_r]) ** 2
                ),
                data["imu_linear_accelerations"][file_id, :2],
            )
        )
        raw_ins_velocities += [ins_measurement[:3]]
        ins_uncertainty = (
            np.diag([sigma_v_x, sigma_v_y, sigma_r, sigma_a_x, sigma_a_y]) ** 2
        )
        wheels_measurement = data["wheel_speeds"][file_id] * 0.2 * 2 * np.pi / 60
        wheels_uncertainty = np.array([2.0, 2.0, 2.0, 2.0]) ** 2
        gss_uncertainty = np.diag([1.0, 0.01]) ** 2
        gss_measurement = data["gss_velocities"][
            file_id, :2
        ] + np.random.multivariate_normal(np.zeros(2), gss_uncertainty)
        start = perf_counter()
        ve.predict(motion_covariance)
        ve.update(
            measurements={
                # "gss": gss_measurement,
                "ins": ins_measurement,
            },
            measurement_covariances={
                # "gss": gss_uncertainty,
                "ins": ins_uncertainty,
            },
        )
        wheels_measurement[0] += 0.5 * ve.mu[3]
        wheels_measurement[1] -= 0.5 * ve.mu[3]
        wheels_measurement[2] += 0.5 * ve.mu[3]
        wheels_measurement[3] -= 0.5 * ve.mu[3]
        # for i in range(4):
        #     ve.update(
        #         measurements={"wheel_speeds": wheels_measurement[i]},
        #         measurement_covariances={"wheel_speeds": wheels_uncertainty[i]},
        #     )

        end = perf_counter()

        runtimes.append(end - start)
        estimated_velocities.append(ve.mu[:3])
        diagnostics["ins"].append(ve.diagnostics["ins"])
        diagnostics["gss"].append(ve.diagnostics["gss"])
        diagnostics["wheel_speeds"].append(ve.diagnostics["wheel_speeds"])
        mahalanobis_distances["ins"].append(ve.mahalanobis_distances["ins"])
        mahalanobis_distances["gss"].append(ve.mahalanobis_distances["gss"])
        mahalanobis_distances["wheel_speeds"].append(
            ve.mahalanobis_distances["wheel_speeds"]
        )

    runtimes = np.array(runtimes)
    true_velocities = np.array(true_velocities)
    estimated_velocities = np.array(estimated_velocities)
    raw_ins_velocities = np.array(raw_ins_velocities)

    print("Average runtime: {} ms".format(np.mean(runtimes) * 1000))
    print_stats_errs(
        estimated_velocities[:, 0], true_velocities[:, 0], "v_x", relative=True
    )
    print_stats_errs(
        estimated_velocities[:, 1], true_velocities[:, 1], "v_y", relative=True
    )
    print_stats_errs(
        estimated_velocities[:, 2], true_velocities[:, 2], "r", relative=True
    )

    plt.figure(figsize=(10, 7))
    plt.subplot(3, 2, 1)
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        raw_ins_velocities[:, 0],
        label="raw ins",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        estimated_velocities[:, 0],
        label="estimated",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)), true_velocities[:, 0], label="true"
    )
    plt.legend()
    plt.title("v_x [m/s]")
    plt.subplot(3, 2, 3)
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        raw_ins_velocities[:, 1],
        label="raw ins",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        estimated_velocities[:, 1],
        label="estimated",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)), true_velocities[:, 1], label="true"
    )
    plt.legend()
    plt.title("v_y [m/s]")
    plt.subplot(3, 2, 5)
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        raw_ins_velocities[:, 2],
        label="raw ins",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        estimated_velocities[:, 2],
        label="estimated",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)), true_velocities[:, 2], label="true"
    )
    plt.legend()
    plt.title("r [rad/s]")

    plt.subplot(3, 2, 2)
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(diagnostics["ins"]),
        label="ins",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(diagnostics["gss"]),
        label="gss",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(diagnostics["wheel_speeds"]),
        label="wheel_speeds",
    )
    plt.legend()
    plt.title("diagnostics")
    plt.subplot(3, 2, 4)
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(mahalanobis_distances["ins"]),
        label="ins",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(mahalanobis_distances["gss"]),
        label="gss",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.array(mahalanobis_distances["wheel_speeds"]),
        label="wheel_speeds",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.ones(len(estimated_velocities)) * chi2.ppf(0.95, 2),
        "r:",
    )
    plt.plot(
        dt * np.arange(len(estimated_velocities)),
        np.ones(len(estimated_velocities)) * chi2.ppf(0.99, 2),
        "r:",
    )
    plt.legend()
    plt.title("mahalanobis distances")

    plt.tight_layout()
    plt.savefig("ve0.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
