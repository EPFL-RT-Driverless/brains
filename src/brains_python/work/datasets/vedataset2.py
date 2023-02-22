from itertools import product
from common import *


def main():
    final_data = []
    for exp_id, ((_, track_name), v_x_max) in enumerate(
        product(missions_tracks[:], max_longitudinal_speeds[:])
    ):
        print(
            "exp_id={}, track_name={}, v_x_max={}".format(exp_id, track_name, v_x_max)
        )
        data = np.load(
            f"/Users/tudoroancea/Developer/racing_team/brains/src/brains_python/work/datasets/data/ve_dataset/{track_name}_{v_x_max}.npz"
        )
        final_data.append(
            np.hstack(
                (
                    exp_id * np.ones((data["states"].shape[0], 1)),
                    data["states"][:, 3:5]
                    + np.random.multivariate_normal(
                        np.zeros(2),
                        np.diag([1.0, 0.1]) ** 2,
                        data["states"].shape[0],
                    ),
                    np.expand_dims(data["imu_angular_velocities"][:, 2], -1),
                    data["imu_linear_accelerations"][:, :2],
                    data["gss_velocities"][:, :2],
                    data["wheel_speeds"],
                    np.array(
                        [
                            0.2 * data["controls"][:, 0],
                            0.2 * data["controls"][:, 0],
                            0.3 * data["controls"][:, 0],
                            0.3 * data["controls"][:, 0],
                        ]
                    ).T
                    * 675.0,
                    np.expand_dims(data["controls"][:, 1], -1),
                    data["states"][:, 3:6],
                )
            ),
        )
    final_data = np.vstack(final_data)
    # noinspection PyTypeChecker
    np.savetxt(
        "data/ve_dataset/combined.csv",
        final_data,
        delimiter=",",
        header=",".join(
            ["exp_id"] + [f"y{i}" for i in range(16)] + [f"x{i}" for i in range(3)]
        ),
        fmt="%g",
        comments="",
    )


if __name__ == "__main__":
    main()
