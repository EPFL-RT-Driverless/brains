import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from common import stats_errs, print_stats_errs

np.random.seed(127)


class VE0:
    nx = 5

    mu: np.ndarray
    Sigma: np.ndarray
    dt: float

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

    def motion_model(self, x=None, dt=None):
        if x is None:
            x = self.mu.copy()
        if dt is None:
            dt = self.dt

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

        return new_x, G

    def observation_model_ins(self, x=None):
        if x is None:
            x = self.mu.copy()

        h = x
        H = np.eye(self.nx)

        return h, H

    def observation_model_gss(self, x=None):
        if x is None:
            x = self.mu.copy()

        h = x[:2]
        H = np.hstack((np.eye(2), np.zeros((2, self.nx - 2))))

        return h, H

    def observation_model_wheels(self, x=None):
        if x is None:
            x = self.mu.copy()

        h = x[0]
        H = np.array([[1, 0, 0, 0, 0]])

        return h, H

    def predict(self, motion_covariance: np.ndarray):
        self.mu, G = self.motion_model()
        self.Sigma = G @ self.Sigma @ G.T + motion_covariance

    def update_ins(self, measurement: np.ndarray, measurement_covariance: np.ndarray):
        h, H = self.observation_model_ins()
        K = (
            self.Sigma
            @ H.T
            @ np.linalg.inv(H @ self.Sigma @ H.T + measurement_covariance)
        )
        self.mu = self.mu + K @ (measurement - h)
        self.Sigma = (np.eye(self.nx) - K @ H) @ self.Sigma

    def update_gss(self, measurement: np.ndarray, measurement_covariance: np.ndarray):
        h, H = self.observation_model_gss()
        K = (
            self.Sigma
            @ H.T
            @ np.linalg.inv(H @ self.Sigma @ H.T + measurement_covariance)
        )
        self.mu = self.mu + K @ (measurement - h)
        self.Sigma = (np.eye(self.nx) - K @ H) @ self.Sigma

    def update_wheels(self, measurement: float, measurement_covariance: float):
        h, H = self.observation_model_wheels()
        K = (
            self.Sigma
            @ H.T
            @ np.linalg.inv(H @ self.Sigma @ H.T + measurement_covariance)
        )
        self.mu = self.mu + K.ravel() * (measurement - h)
        self.Sigma = (np.eye(self.nx) - K @ H) @ self.Sigma


def main():
    dt = 0.01
    track_name = "fsds_competition_1"
    filename = f"{track_name}_15.0.npz"
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
        initial_state=np.zeros(5, dtype=np.float64),
        initial_covariance=np.zeros((5, 5)),
        sampling_time=dt,
    )
    runtimes = []
    estimated_velocities = []
    raw_ins_velocities = []
    true_velocities = []
    motion_covariance = np.diag([0.1, 0.1, 0.1, 2.0, 2.0]) ** 2
    for file_id in range(0, data["states"].shape[0], int(dt / 0.01)):
        true_velocities.append(data["states"][file_id][3:])
        sigma_v_x = 0.5
        sigma_v_y = 0.06
        sigma_r = 0.25
        sigma_a_x = 2.1
        sigma_a_y = 0.6
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
        # gss_measurement = data["gss_velocities"][file_id, :2]
        # gss_uncertainty = np.diag([0.1, 0.0001]) ** 2
        start = perf_counter()
        ve.predict(motion_covariance)
        # ve.update_gss(gss_measurement, gss_uncertainty)
        ve.update_ins(ins_measurement, ins_uncertainty)
        wheels_measurement[0] += 0.5 * ve.mu[3]
        wheels_measurement[1] -= 0.5 * ve.mu[3]
        wheels_measurement[2] += 0.5 * ve.mu[3]
        wheels_measurement[3] -= 0.5 * ve.mu[3]
        for i in range(4):
            ve.update_wheels(wheels_measurement[i], np.diag([5.0]) ** 2)

        end = perf_counter()
        runtimes.append(end - start)
        estimated_velocities.append(ve.mu[:3])

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
    plt.subplot(3, 1, 1)
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
    plt.subplot(3, 1, 2)
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
    plt.subplot(3, 1, 3)
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

    plt.tight_layout()
    plt.savefig("ve0.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
