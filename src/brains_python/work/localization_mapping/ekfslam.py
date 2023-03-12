import os.path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from tqdm import tqdm

import track_database as tdb
from brains_python.localization_mapping.ekfslam import EKFSLAM, EKFSLAMMode
from track_database.utils import plot_cones

np.random.seed(127)


def print_stat(arr, name):
    return "{:<15}: {:<6.3f} ± {:>6.3f}".format(name, np.mean(arr), np.std(arr))


def run():
    # general options ===========================================================
    dt = 0.01
    track_name = "fsds_competition_1"
    vxmax = 10.0
    filename = f"{track_name}_{vxmax}.npz"
    mode = EKFSLAMMode.LOCALIZATION
    live_plot = False
    pause_time = 0.1
    dead_reckoning_only = False
    odometry_err_percentage = 4 / 100

    # load track and data =======================================================
    track = tdb.load_track(track_name)
    # noinspection PyTypeChecker
    data: np.lib.npyio.NpzFile = np.load(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../datasets/data/localization_dataset",
                filename,
            )
        )
    )

    # initialize EKF-SLAM =======================================================
    slamer = EKFSLAM(
        mode=mode,
        initial_state=np.array([0.0, 0.0, np.pi / 2]),
        initial_state_uncertainty=0.0 * np.eye(3),
        initial_map=data["global_cones_positions"],
        initial_landmark_uncertainty=1e-1 * np.eye(2),
        sampling_time=dt,
    )
    # tpr = data[f"rel_cones_positions_{0}"]
    # [slamer._add_landmark(tpr[i]) for i in range(tpr.shape[0])]

    # run EKF-SLAM ==============================================================
    odometry_update_runtimes = []
    yaw_update_runtimes = []
    cones_update_runtimes = []
    poses = np.empty((0, 3), dtype=float)
    true_poses = np.empty((0, 3), dtype=float)
    data_association_statistics = np.empty((0, 3), dtype=float)
    if live_plot:
        plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show(block=False)
    for file_id in tqdm(range(0, (len(data.files) - 5), int(dt / 0.01))):
        # odometry update
        true_state = data["states"][file_id]
        Q1 = (
            np.array(
                [
                    odometry_err_percentage * true_state[3],
                    odometry_err_percentage * true_state[4],
                    odometry_err_percentage * true_state[5],
                ]
            )
            ** 2
        )
        Q2 = (
            np.array(
                [
                    odometry_err_percentage * true_state[3],
                    odometry_err_percentage * true_state[4],
                    odometry_err_percentage * true_state[5],
                ]
            )
            * 2
        ) ** 2
        odometry_measurement = true_state[-3:] + np.random.multivariate_normal(
            np.zeros(3), np.diag(Q1)
        )
        start = perf_counter()
        slamer.odometry_update(
            odometry=odometry_measurement,
            odometry_uncertainty=Q2,
            sampling_time=dt,
        )
        stop = perf_counter()
        odometry_update_runtimes.append(1000 * (stop - start))

        # yaw update
        start = perf_counter()
        slamer.yaw_update(
            yaw=true_state[2] + 2e-2 * np.random.randn(1),
            yaw_uncertainty=1e-3,
        )
        stop = perf_counter()
        yaw_update_runtimes.append(1000 * (stop - start))

        # cones update
        if file_id % 5 == 0:
            cones = data[f"rel_cones_positions_{file_id}"]
            e_rho = 0.1
            e_theta = 0.1
            R1 = np.array([e_rho, e_theta]) ** 2
            R2 = (np.array([e_rho, e_theta]) * 1) ** 2
            observations = cones + np.random.multivariate_normal(
                np.zeros(2), np.diag(R1), cones.shape[0]
            )
            inverse_observations = np.array(
                [
                    slamer.mu[0]
                    + observations[:, 0] * np.cos(observations[:, 1] + slamer.mu[2]),
                    slamer.mu[1]
                    + observations[:, 0] * np.sin(observations[:, 1] + slamer.mu[2]),
                ]
            ).T
            start = perf_counter()
            if not dead_reckoning_only:
                slamer.cones_update(
                    observations=observations,
                    observations_uncertainties=R2,
                )
            stop = perf_counter()
            cones_update_runtimes.append(1000 * (stop - start))
            if not dead_reckoning_only:
                data_association_statistics = np.vstack(
                    (data_association_statistics, slamer.data_association_statistics)
                )
            if live_plot:
                # plot stuff
                plt.clf()
                plot_cones(
                    track.blue_cones,
                    track.yellow_cones,
                    track.big_orange_cones,
                    track.small_orange_cones,
                    show=False,
                )
                plt.axis("equal")
                plt.tight_layout()
                plt.plot(
                    true_poses[:, 0], true_poses[:, 1], "g-", label="True trajectory"
                )
                plt.plot(poses[:, 0], poses[:, 1], "b-", label="Estimated trajectory")
                plt.scatter(slamer.map[:, 0], slamer.map[:, 1], c="r", label="Map")
                plt.scatter(
                    inverse_observations[:, 0],
                    inverse_observations[:, 1],
                    c="g",
                    label="Reconstructed landmarks",
                )
                plt.legend()
                plt.xlim(
                    -(np.max(np.abs(inverse_observations[:, 0])) + 1),
                    np.max(np.abs(inverse_observations[:, 0])) + 1,
                )
                plt.ylim(
                    0.0,
                    np.max(inverse_observations[:, 1]) + 1,
                )

        poses = np.vstack((poses, slamer.state))
        true_poses = np.vstack((true_poses, true_state[:3]))
        if live_plot:
            plt.pause(pause_time)

        if np.linalg.norm(slamer.state[:2] - true_state[:2]) > 3:
            print("Localization error too high, aborting")
            break

    odometry_update_runtimes = np.array(odometry_update_runtimes)
    yaw_update_runtimes = np.array(yaw_update_runtimes)
    cones_update_runtimes = np.array(cones_update_runtimes)

    print(
        "Runtimes:\n",
        "\t" + print_stat(odometry_update_runtimes, "odometry update") + " ms\n",
        "\t" + print_stat(yaw_update_runtimes, "yaw update") + " ms\n",
        "\t" + print_stat(cones_update_runtimes, "cones update") + " ms",
    )

    localization_error = np.linalg.norm(poses[:, :2] - true_poses[:, :2], axis=1)
    orientation_error = np.abs(poses[:, 2] - true_poses[:, 2])
    print(
        "Error statistics:\n",
        "\t" + print_stat(localization_error, "position error") + " m\n",
        "\t" + print_stat(orientation_error, "orientation error") + " rad",
    )

    print(
        "Average data association statistics:\n{:.3f} % associated, {:.3f}% discarded, {:.3f}% new".format(
            *np.mean(data_association_statistics, axis=0)
        )
    )

    if not live_plot:
        track = tdb.load_track(track_name)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plot_cones(
            track.blue_cones,
            track.yellow_cones,
            track.big_orange_cones,
            track.small_orange_cones,
            show=False,
        )
        plt.axis("equal")
        plt.tight_layout()
        plt.plot(true_poses[:, 0], true_poses[:, 1], "g-", label="True trajectory")
        plt.plot(poses[:, 0], poses[:, 1], "b-", label="Estimated trajectory")
        plt.scatter(slamer.map[:, 0], slamer.map[:, 1], s=10, c="r", label="Map")

        # plot landmarks covariance ellipses
        for i in range(slamer.map.shape[0]):
            x, y = slamer.map[i, :]
            cov = slamer.Sigma[3 + 2 * i : 3 + 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2]
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
            angle = np.rad2deg(np.arccos(v[0, 0]))
            print(
                "x: {}, y: {}, lam: {}, angle : {}, cov : {}".format(
                    x, y, lam, angle, cov
                )
            )
            ell = Ellipse(
                xy=(x, y),
                width=2 * 5e2 * lam[0],
                height=2 * 5e2 * lam[1],
                angle=angle,
            )
            ell.set_facecolor("none")
            ell.set_edgecolor("r")
            if i == 0:
                ell.set_label(r"Landmark covariance ($\times 500$)")

            ax.add_patch(ell)

        plt.legend()
        plt.xlim(-70, -30)
        plt.ylim(-30, 30)
        plt.savefig("ekfslam.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    run()
