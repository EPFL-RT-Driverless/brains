import numpy as np

from brains_python.common import Mission
from brains_python.control import (
    MotionPlanner,
    MotionPlannerParams,
    fsds_car_params,
    CarParams,
)
from brains_python.control.utils import AVAILABLE_MISSION_TRACK_TUPLES
from track_database import load_track
from trajectory_planning_helpers import calc_normal_vectors
from time import perf_counter


def main(n_cross_track: int, n_longitudinal, n_heading: int):
    N = 40  # horizon size
    sampling_time = 0.05
    result = []
    t0 = perf_counter()
    for v_x_max in {2.5, 5.0, 10.0, 15.0}:
        car_params = CarParams(**fsds_car_params)
        car_params.v_x_max = v_x_max
        for mission, track_name in {
            (Mission.ACCELERATION, "acceleration"),
            (Mission.SHORT_SKIDPAD, "short_skidpad"),
            (Mission.TRACKDRIVE, "fsds_competition_1"),
            (Mission.TRACKDRIVE, "fsds_competition_2"),
            (Mission.TRACKDRIVE, "fsds_competition_3"),
            (Mission.TRACKDRIVE, "fsds_default"),
        }:
            if mission in {Mission.AUTOCROSS, Mission.SKIDPAD}:
                continue
            t1 = perf_counter()
            track = load_track(track_name)
            motion_planner = MotionPlanner(
                MotionPlannerParams(
                    mission=mission,
                    center_points=track.center_line,
                    widths=track.track_widths,
                    additional_attributes=[],
                ),
                car_params=car_params,
            )

            def add_data(s: float, t: float):
                time_horizon = np.linspace(t, t + N * sampling_time, N + 1)
                reference_arc_lengths = motion_planner.arc_length_vs_time(time_horizon)
                X_ref = motion_planner.X_ref_vs_arc_length(reference_arc_lengths)
                Y_ref = motion_planner.Y_ref_vs_arc_length(reference_arc_lengths)
                phi_ref = motion_planner.phi_ref_vs_arc_length(reference_arc_lengths)
                v_x_ref = motion_planner.v_x_ref_vs_arc_length(reference_arc_lengths)

                cross_track_errors = np.random.randn(n_cross_track)
                longitudinal_errors = np.random.randn(n_longitudinal)
                heading_errors = np.random.randn(n_heading)
                for cross_track_error in cross_track_errors:
                    for longitudinal_error in longitudinal_errors:
                        for heading_error in heading_errors:
                            normal_vector = calc_normal_vectors(np.array([s]))
                            X = X_ref[0] + cross_track_error * normal_vector[0, 0]
                            Y = Y_ref[0] + cross_track_error * normal_vector[0, 1]
                            phi = phi_ref[0] + heading_error
                            v_x = v_x_ref[0] + longitudinal_error
                            v_y = np.random.randn()
                            r = np.random.randn()
                            result.append(
                                np.concatenate(
                                    [
                                        np.array([X, Y, phi, v_x, v_y, r]),
                                        X_ref,
                                        Y_ref,
                                        phi_ref,
                                        v_x_ref,
                                    ]
                                )
                            )

            if mission == Mission.ACCELERATION:
                add_data(0.0, 0.0)
            elif mission == Mission.SHORT_SKIDPAD:
                # phase 1:
                offset = motion_planner.car_params.v_x_max * N * sampling_time
                sigmas = []
                sigma0 = max(0.0, motion_planner.key_points[0][0] - offset)
                if offset < 0.5 * (
                    motion_planner.key_points[1][0] - motion_planner.key_points[0][0]
                ):
                    sigmas.append((sigma0, motion_planner.key_points[0][0] + offset))
                    sigmas.append(
                        (
                            motion_planner.key_points[1][0] - offset,
                            motion_planner.key_points[1][0] + offset,
                        )
                    )
                    sigmas.append(
                        (
                            motion_planner.key_points[2][0] - offset,
                            motion_planner.key_points[2][0],
                        )
                    )
                else:
                    sigmas.append((sigma0, motion_planner.key_points[2][0]))

                for a, b in sigmas:
                    for s in np.linspace(a, b, int(b - a)):
                        add_data(s, motion_planner.time_vs_arc_length(s))

            elif mission == Mission.TRACKDRIVE:
                for s in np.linspace(
                    0,
                    motion_planner.total_length,
                    int(motion_planner.total_time),
                    endpoint=False,
                ):
                    add_data(s, motion_planner.time_vs_arc_length(s))
            else:
                # we have already continued above
                continue

            t2 = perf_counter()
            print(
                f"Time for {mission} on {track_name} with max velocity {v_x_max} : {t2 - t1} s"
            )

    result = np.array(result)
    print(f"Total time: {perf_counter() - t0} s")
    np.save("data/control_dataset.npy", result)


if __name__ == "__main__":
    main(n_cross_track=10, n_heading=10, n_longitudinal=10)
