#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
import os.path
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from control_module.ihm_forces import *

# Simulation params
mercury = True
graphics = True
liveplot = False
short_track = True
sim_length = 600 if mercury else 300
sim_pause = 0.0001
show_next_path_points = True


def generate_skidpad(
    track_width: float = 1.5,
    start_length: float = 15.0,
    end_length: float = 25.0,
    inner_radius: float = 7.625,
    number_points_loops: int = 100,
    number_points_start: int = 50,
    number_points_end: int = 20,
    short: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic function that generates a list of points for the reference path (central
    line) and the cone positions of a skidpad.

    The skidpad can have any dimensions and number of reference points on the loops as
    well as the start and finish line, everything is specified by the following arguments.

    The coordinates are chosen in such a way that the straight line portion is oriented
    along the Y-axis and the origin is at the center of the skidpad.

    Parameters
    ----------
    track_width :
        half of the total track width, i.e. half of the distance between two
        corresponding blue and yellow cones
    start_length :
        length of the start line
    end_length :
        length of the end line
    inner_radius :
        radius of the inner circle of cones of a loop
    number_points_loops :
        number of reference points on the loops
    number_points_start :
        number of reference points on the start line
    number_points_end :
        number of reference points on the end line
    short :
        whether to generate a reference path with only once each loop (default) or not.

    Returns
    -------
    see comment

    reference_points :
        np.ndarray with 2 rows, and each column corresponding to a reference point
    left_cones :
        np.ndarray with 2 rows, and each column corresponding to a cone position on the
        left side of the track
    right_cones :
        np.ndarray with 2 rows, and each column corresponding to a cone position on the
        right side of the track

    """
    # generate reference path ===============================================================
    startLine = np.zeros((2, number_points_start))
    startLine[1, :] = np.linspace(
        -start_length, 0.0, number_points_start, endpoint=False
    )
    endLine = np.zeros((2, number_points_end + 1))
    endLine[1, :] = np.linspace(0.0, end_length, number_points_end + 1)
    # endLine = np.delete(endLine, 0, 1)

    rightCircle = np.array(
        [
            track_width
            + inner_radius
            + (track_width + inner_radius)
            * np.cos(np.linspace(np.pi, -np.pi, number_points_loops, endpoint=False)),
            (track_width + inner_radius)
            * np.sin(np.linspace(np.pi, -np.pi, number_points_loops, endpoint=False)),
        ]
    )
    leftCircle = np.array(
        [
            -(track_width + inner_radius)
            + (track_width + inner_radius)
            * np.cos(np.linspace(0, 2 * np.pi, number_points_loops, endpoint=False)),
            (track_width + inner_radius)
            * np.sin(np.linspace(0, 2 * np.pi, number_points_loops, endpoint=False)),
        ]
    )
    reference_points = np.concatenate(
        (
            startLine,
            rightCircle,
        ),
        axis=1,
    )
    if not short:
        reference_points = np.concatenate(
            (
                reference_points,
                rightCircle,
                leftCircle,
            ),
            axis=1,
        )

    reference_points: np.ndarray = np.concatenate(
        (
            reference_points,
            leftCircle,
            endLine,
        ),
        axis=1,
    )

    # generate cone positions ===============================================================
    # first left cones
    thetaRight = np.linspace(np.pi, -np.pi, 16, endpoint=False)
    innerCircle = np.array(
        [inner_radius * np.cos(thetaRight), inner_radius * np.sin(thetaRight)]
    )
    outerCircle = np.array(
        [
            (inner_radius + 2 * track_width) * np.cos(thetaRight),
            (inner_radius + 2 * track_width) * np.sin(thetaRight),
        ]
    )
    start_left = np.zeros((2, 3))
    start_left[0, :] = -track_width
    start_left[1, :] = np.linspace(-start_length, 0.0, 3)
    end_left = np.zeros((2, 3))
    end_left[0, :] = -track_width
    end_left[1, :] = np.linspace(0.0, end_length, 3)
    inner_left_circle = np.array([[-(track_width + inner_radius)], [0.0]]) + innerCircle
    outer_right_circle = np.array([[(track_width + inner_radius)], [0.0]]) + outerCircle
    outer_right_circle = np.delete(
        outer_right_circle, np.where(outer_right_circle[0, :] <= track_width), 1
    )
    left_cones: np.ndarray = np.concatenate(
        (
            start_left,
            outer_right_circle,
            inner_left_circle,
            end_left,
        ),
        axis=1,
    )

    # second right cones
    start_right = np.zeros((2, 3))
    start_right[0, :] = track_width
    start_right[1, :] = np.linspace(-start_length, 0.0, 3)
    end_right = np.zeros((2, 3))
    end_right[0, :] = track_width
    end_right[1, :] = np.linspace(0.0, end_length, 3)
    inner_right_circle = np.array([[(track_width + inner_radius)], [0.0]]) + innerCircle
    outer_left_circle = np.array([[-(track_width + inner_radius)], [0.0]]) + outerCircle
    outer_left_circle = np.delete(
        outer_left_circle, np.where(outer_left_circle[0, :] >= -track_width), 1
    )
    right_cones: np.ndarray = np.concatenate(
        (
            start_right,
            inner_right_circle,
            outer_left_circle,
            end_right,
        ),
        axis=1,
    )

    return reference_points, left_cones, right_cones


class Skidpad:
    """
    Wrapper around generate_skidpad() to ensure that the number of reference points on
    the start and end line are such that the breaks are evenly spaced.
    WARNING : this may lead to an unforeseen length of the start line.
     To make sure that the initial theta you give to the controller is the right one,
     choose an initial Y<=0 and compute the initial theta using the method get_initial_theta()
    """

    # geometric parameters of the skidpad
    inner_radius: float
    track_width: float
    loop_radius: float
    N_loop: int
    N_start: int
    short: bool
    total_track_length: float
    interval_length: float

    # points describing the skidpad (points of the reference path and cones positions)
    path_points: np.ndarray
    left_cones: np.ndarray
    right_cones: np.ndarray

    def __init__(
        self, track_width: float, inner_radius: float, N_loop: int, short: bool = True
    ):
        self.track_width = track_width
        self.inner_radius = inner_radius
        self.loop_radius = inner_radius + track_width
        self.N_loop = N_loop
        self.N_start = int(np.ceil(N_loop * 2.0 / (np.pi * 2.0)))
        self.short = short
        self.interval_length = self.loop_radius * 2 * np.pi / self.N_loop
        self.total_track_length = (
            self.interval_length
            * 2
            * (self.N_loop + self.N_start + (0 if self.short else self.N_loop))
        )

        self.path_points, self.left_cones, self.right_cones = generate_skidpad(
            track_width=track_width,
            inner_radius=inner_radius,
            start_length=self.N_start * self.interval_length,
            end_length=self.N_start * self.interval_length,
            number_points_start=self.N_start,
            number_points_end=self.N_start,
            number_points_loops=self.N_loop,
            short=short,
        )

    def initial_id(self, y_0: float) -> float:
        assert y_0 <= 0, "y_0 must be non-positive"
        return (self.N_loop * y_0 + 2 * np.pi * self.loop_radius * self.N_start) / (
            4.0
            * np.pi
            * self.loop_radius
            * ((self.N_loop if self.short else 2 * self.N_loop) + self.N_start)
        )

    def n_points(self):
        return self.path_points.shape[1]

    def n_intervals(self):
        return self.path_points.shape[1] - 1


def updatePlots():
    """Deletes old data sets in the current plot and adds the new data sets given by
    the variables x, u, pred_x and pred_u (defined in the __main__) to the plot.
    """
    global x, u, pred_x, pred_u, controller, k, next_path_points
    global fig, ax_list

    # Delete old data in plot
    ax_list[0].get_lines().pop(-1).remove()  # remove old prediction of trajectory
    ax_list[0].get_lines().pop(-1).remove()  # remove old trajectory
    ax_list[0].get_lines().pop(-1).remove()  # remove old current position
    if next_path_points is not None and k > 1:
        ax_list[0].get_lines().pop(-1).remove()  # remove the old next path points

    ax_list[1].get_lines().pop(-1).remove()  # remove old prediction of heading angle
    ax_list[1].get_lines().pop(-1).remove()  # remove old heading angle
    ax_list[2].get_lines().pop(-1).remove()  # remove old prediction of velocity
    ax_list[2].get_lines().pop(-1).remove()  # remove old velocity
    ax_list[3].get_lines().pop(-1).remove()  # remove old prediction of steering angle
    ax_list[3].get_lines().pop(-1).remove()  # remove old steering angle
    ax_list[4].get_lines().pop(-1).remove()  # remove old prediction of acceleration
    ax_list[4].get_lines().pop(-1).remove()  # remove old acceleration
    ax_list[5].get_lines().pop(-1).remove()  # remove old prediction of steering rate
    ax_list[5].get_lines().pop(-1).remove()  # remove old steering rate

    # Update plot with current simulation data
    ax_list[0].plot(x[0, k + 1], x[1, k + 1], "gx")  # plot current position
    ax_list[0].plot(x[0, 0 : k + 2], x[1, 0 : k + 2], "b-")  # plot new trajectory
    ax_list[0].plot(
        pred_x[0, 1:], pred_x[1, 1:], "g-"
    )  # plot new prediction of trajectory
    if next_path_points is not None:
        ax_list[0].plot(
            next_path_points[0, :], next_path_points[1, :], "m."
        )  # plot next path points

    ax_list[1].plot(np.rad2deg(x[2, 0 : k + 2]), "b-")  # plot new heading angle
    ax_list[1].plot(
        range(k + 1, k + controller.N), np.rad2deg(pred_x[2, 1:]), "g-"
    )  # plot new prediction of heading angle
    ax_list[2].plot(x[3, 0 : k + 2], "b-")  # plot new velocity
    ax_list[2].plot(
        range(k + 1, k + controller.N), pred_x[3, 1:], "g-"
    )  # plot new prediction of velocity
    ax_list[3].plot(np.rad2deg(x[4, 0 : k + 2]), "b-")  # plot new steering angle
    ax_list[3].plot(
        range(k + 1, k + controller.N), np.rad2deg(pred_x[4, 1:]), "g-"
    )  # plot new prediction of steering angle
    ax_list[4].step(
        range(0, k + 1), u[0, 0 : k + 1], "b-"
    )  # plot new acceleration force
    ax_list[4].step(
        range(k, k + controller.N), pred_u[0, :], "g-"
    )  # plot new prediction of acceleration force
    ax_list[5].step(
        range(0, k + 1), np.rad2deg(u[1, 0 : k + 1]), "b-"
    )  # plot new steering rate
    ax_list[5].step(
        range(k, k + controller.N), np.rad2deg(pred_u[1, :]), "g-"
    )  # plot new prediction of steering rate

    plt.pause(sim_pause)


def createPlot(
    path_points,
    left_cones=None,
    right_cones=None,
):
    """Creates a plot and adds the initial data provided by the arguments"""
    # global x, u, pred_x, pred_u, controller
    global fig

    # Create empty plot
    gs = GridSpec(5, 2, figure=fig)

    # Plot trajectory
    ax_pos = fig.add_subplot(gs[:, 0])
    if left_cones is not None and right_cones is not None:
        ax_pos.scatter(left_cones[0, :], left_cones[1, :], color="b", marker="^")
        ax_pos.scatter(right_cones[0, :], right_cones[1, :], color="y", marker="^")
        ax_pos.scatter(
            left_cones[0, 0], left_cones[1, 0], color="tab:orange", marker="^"
        )
        ax_pos.scatter(
            right_cones[0, 0], right_cones[1, 0], color="tab:orange", marker="^"
        )
        ax_pos.scatter(
            left_cones[0, -1], left_cones[1, -1], color="tab:orange", marker="^"
        )
        ax_pos.scatter(
            right_cones[0, -1], right_cones[1, -1], color="tab:orange", marker="^"
        )

    plt.title("Position")
    ax_pos.axis("equal")
    plt.xlim([np.min(path_points[0, :]) - 1.0, np.max(path_points[0, :]) + 1.0])
    plt.ylim([np.min(path_points[1, :]) - 1.0, np.max(path_points[1, :]) + 1.0])
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")

    (l0,) = ax_pos.plot(
        np.transpose(path_points[0, :]), np.transpose(path_points[1, :]), "r-"
    )
    (l1,) = ax_pos.plot(xinit[0], xinit[1], "bx")
    (l1bis,) = ax_pos.plot(x[0, -1], x[1, -1], "gx")
    (l2,) = ax_pos.plot(x[0, :], x[1, :], "b-")
    (l3,) = ax_pos.plot(pred_x[0, 1:], pred_x[1, 1:], "g-")
    ax_pos.legend(
        [l0, l1, l1bis, l2, l3],
        [
            "desired trajectory",
            "init pos",
            "current pos",
            "car trajectory",
            "predicted car traj.",
        ],
        loc="lower right",
    )

    # Plot heading angle
    ax_phi = fig.add_subplot(gs[0, 1])
    plt.grid("both")
    plt.title("Heading angle")
    plt.xlim([0.0, x.shape[1] - 1])
    ax_phi.plot(np.arange(x.shape[1]), np.rad2deg(x[2, :]), "b-")
    ax_phi.plot(
        np.arange(x.shape[1] - 1, x.shape[1] + controller.N - 2),
        np.rad2deg(pred_x[2, 1:]),
        "g-",
    )

    # Plot velocity
    ax_vel = fig.add_subplot(gs[1, 1])
    plt.grid("both")
    plt.title("Velocity")
    plt.xlim([0.0, x.shape[1] - 1])
    plt.plot(
        [0, x.shape[1] - 1], np.transpose([controller.v_max, controller.v_max]), "r:"
    )
    plt.plot(
        [0, x.shape[1] - 1], np.transpose([-controller.v_max, -controller.v_max]), "r:"
    )
    ax_vel.plot(np.arange(x.shape[1]), x[3, :], "-b")
    ax_vel.plot(
        np.arange(x.shape[1] - 1, x.shape[1] + controller.N - 2), pred_x[3, 1:], "g-"
    )

    # Plot steering angle
    ax_delta = fig.add_subplot(gs[2, 1])
    plt.grid("both")
    plt.title("Steering angle")
    plt.xlim([0.0, x.shape[1] - 1])
    plt.plot(
        [0, x.shape[1] - 1],
        np.rad2deg(np.transpose([controller.delta_max, controller.delta_max])),
        "r:",
    )
    plt.plot(
        [0, x.shape[1] - 1],
        np.rad2deg(np.transpose([-controller.delta_max, -controller.delta_max])),
        "r:",
    )
    ax_delta.plot(np.arange(x.shape[1]), np.rad2deg(x[4, :]), "b-")
    ax_delta.plot(
        np.arange(x.shape[1] - 1, x.shape[1] + controller.N - 2),
        np.rad2deg(pred_x[4, 1:]),
        "g-",
    )

    # Plot acceleration force
    ax_a = fig.add_subplot(gs[3, 1])
    plt.grid("both")
    plt.title("Acceleration")
    plt.xlim([0.0, u.shape[1] - 1])
    plt.plot(
        [0, u.shape[1] - 1], np.transpose([controller.a_max, controller.a_max]), "r:"
    )
    plt.plot(
        [0, u.shape[1] - 1], np.transpose([-controller.a_max, -controller.a_max]), "r:"
    )
    ax_a.step(np.arange(u.shape[1]), u[0, :], "b-")
    ax_a.step(
        np.arange(u.shape[1] - 1, u.shape[1] + controller.N - 1), pred_u[0, :], "g-"
    )

    # Plot steering rate
    ax_r = fig.add_subplot(gs[4, 1])
    plt.grid("both")
    plt.title("Steering rate")
    plt.xlim([0.0, u.shape[1] - 1])
    plt.plot(
        [0, u.shape[1] - 1],
        np.rad2deg(np.transpose([controller.r_max, controller.r_max])),
        "r:",
    )
    plt.plot(
        [0, u.shape[1] - 1],
        np.rad2deg(np.transpose([-controller.r_max, -controller.r_max])),
        "r:",
    )
    ax_r.step(np.arange(u.shape[1]), np.rad2deg(u[1, :]), "b-")
    ax_r.step(
        np.arange(u.shape[1] - 1, u.shape[1] + controller.N - 1), pred_u[1, :], "g-"
    )

    plt.tight_layout()


if __name__ == "__main__":
    skidpad = (
        Skidpad(
            track_width=1.5,
            inner_radius=7.625,
            N_loop=20,
            short=True if short_track else False,
        )
        if mercury
        else Skidpad(
            track_width=1.0,
            inner_radius=4.0,
            N_loop=16,
            short=True if short_track else False,
        )
    )

    # translate the points so that the origin is at the initial state
    skidpad.left_cones -= skidpad.path_points[:, 0].reshape(2, 1)
    skidpad.right_cones -= skidpad.path_points[:, 0].reshape(2, 1)
    skidpad.path_points -= skidpad.path_points[:, 0].reshape(2, 1)

    # Simulation ----------

    # Variables for storing simulation data
    x = np.zeros((5, sim_length + 1))  # states
    u = np.zeros((2, sim_length))  # inputs
    runtimes = np.zeros(sim_length)  # runtimes

    # Set initial conditions and initial guess to start solver from
    xinit = np.array(
        [
            0.0,
            0.0,
            np.pi / 2,
            0.0,
            0.0,
        ]
    )
    x[:, 0] = xinit
    last_idx = 0

    # get controller
    controller = IHMForces(
        initial_state=xinit,
        center_points=skidpad.path_points,
        total_track_length=skidpad.total_track_length,
        car_params=mercury_params_kin_5 if mercury else minimercury_params_kin_5,
        costs=mercury_costs if mercury else minimercury_costs,
        v_ref=15.0,
        solver_dir_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ihm_solver_" + str(PhysicalModel.KIN_4),
        ),
    )

    prediction = np.tile(
        np.concatenate((np.zeros(2), xinit)).reshape(7, 1), (1, controller.N)
    )
    pred_u = prediction[: controller.control_dim, :]
    pred_x = prediction[controller.control_dim :, :]

    # generate plot with initial values
    if graphics:
        fig = plt.figure(figsize=(15, 7))
        plt.clf()

    if graphics and liveplot:
        createPlot(controller.path_points, skidpad.left_cones, skidpad.right_cones)
        ax_list = fig.axes

    # Simulation
    for k in range(sim_length):
        start = time()

        controller.current_state = x[:, k]
        pred_x, pred_u, next_path_points = controller._compute_control_dev()
        prediction = np.concatenate((pred_u, pred_x), axis=0)

        stop = time()
        # sys.stderr.write("\tSimulation step took {} ms\n".format(1000 * (stop - start)))
        runtimes[k] = 1000 * (stop - start)

        # Apply optimized input u of first stage to system and save simulation data
        u[:, k] = pred_u[:, 0]
        x[:, k + 1] = pred_x[:, 1]

        # plot results of current simulation step
        if graphics and liveplot:
            updatePlots()

        if (np.abs(x[0, k + 1]) <= 1.0 and x[1, k + 1] >= 33.0) or k == sim_length - 1:
            x = np.delete(x, np.s_[k + 2 :], axis=1)
            u = np.delete(u, np.s_[k + 1 :], axis=1)
            runtimes = np.delete(runtimes, np.s_[k + 1 :])
            print("Average iteration runtime : {} ms".format(np.mean(runtimes)))
            if graphics and liveplot:
                plt.show()
            break
        else:
            if graphics and liveplot:
                plt.draw()

    if graphics and not liveplot:
        createPlot(controller.path_points, skidpad.left_cones, skidpad.right_cones)
        plt.show()
