#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless

# state :  x = [X, Y, phi, v, delta]
# control : u = [a, r]
# stage variable : z = [a, r, X, Y, phi, v, delta]

# where X,Y are the position, v the velocity in heading angle phi of the car, and delta
# is the steering angle relative to the heading angle.
# The inputs are acceleration a and steering rate r.
# The physical constants l_r and l_f denote the distance from the car's center of
# gravity to the rear wheels and the front wheels.

# The dynamics of the system are given by a simple kinematic bicycle model:

#    dX/dt = v*cos(phi + beta)
#    dY/dt = v*sin(phi + beta)
#    dv/dt = a
#    dphi/dt = v/l_r*sin(beta)
#    ddelta/dt = r

#    with:
#    beta = arctan(l_r/(l_f + l_r)*tan(delta))

# The car starts from standstill with a certain heading angle, and the
# optimization problem is to minimize the distance of the car's position
# to a given set of points on a path with respect to time.

# Quadratic costs for the acceleration force and steering rate are added to
# the objective to avoid excessive maneuvers.

# There are bounds on v and delta and their derivatives a and r to avoid changing them
# too quickly.
import os.path
from time import time
from typing import Tuple

import casadi as ca
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Simulation params
mercury = True
sim_length = 300 if not mercury else 600
sim_pause = 0.0001
show_next_path_points = True
from_dir = True

# physical constants of the car
if mercury:
    l_r = 0.895  # distance rear wheels to center of gravity of the car
    l_f = 0.675  # distance front wheels to center of gravity of the car
    delta_max = np.deg2rad(40.0)  # [rad]
else:
    l_r = 0.1904  # distance rear wheels to center of gravity of the car
    l_f = 0.2121  # distance front wheels to center of gravity of the car
    delta_max = np.deg2rad(30.0)  # [rad]

v_max = 15.0  # [m/s]
a_max = 15.0  # [m/s^2]
r_max = delta_max  # [rad/s]

# MPC params
sampling_time = 0.02  # [s]
interval_length = v_max * sampling_time  # [m]
cost_deviation = 1000.0
cost_a = 2.0
cost_r = 10.0


def continuous_dynamics(x, u):
    """Defines dynamics of the car, i.e. equality constraints.
    parameters:
    state x = [X, Y, phi, v, delta]
    input u = [a, r]
    """
    # set intermediate parameters
    beta = ca.arctan(l_r / (l_f + l_r) * ca.tan(x[4]))

    # calculate dx/dt
    return ca.vertcat(
        x[3] * ca.cos(x[2] + beta),  # dX/dt = v*cos(phi+beta)
        x[3] * ca.sin(x[2] + beta),  # dY/dt = v*sin(phi+beta)
        x[3] / l_r * ca.sin(beta),  # dphi/dt = v/l_r*sin(beta)
        u[0],  # dv/dt = a
        u[1],  # ddelta/dt = r
    )


def obj(z, p):
    """Least square costs on deviating from the path
    stage variable : z = [a, r, X, Y, phi, v, delta]
    p = point on path that is to be headed for, and cost parameters
    """
    return (
        p[2] * (z[2] - p[0]) ** 2  # costs on deviating on the path in x-direction
        + p[2] * (z[3] - p[1]) ** 2  # costs on deviating on the path in y-direction
        + p[3] * z[0] ** 2  # penalty on input a
        + p[4] * z[1] ** 2  # penalty on input r
    )


def objN(z, p):
    """Increased east square costs on deviating from the path
    stage variable : z = [a, r, X, Y, phi, v, delta]
    current_target = point on path that is to be headed for
    """
    return 10 * obj(z, p)


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


def extract_next_path_points(path_points, pos, N, last_idx):
    """Extract the next N points on the path for the next N stages starting from
    the current car position pos
    """
    considered_points = path_points[:, last_idx : last_idx + N]
    closest_point_id: int = last_idx + np.argmin(
        np.sum(np.square(considered_points - pos.reshape(2, 1)), axis=0)
    )
    return (
        path_points[:, closest_point_id + 1 : closest_point_id + N + 1],
        closest_point_id,
    )


def generate_path_planner(from_directory: bool = False):
    """Generates and returns a FORCESPRO solver that calculates a path based on
    constraints and dynamics while minimizing an objective function
    """
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel()
    model.N = 20  # horizon length
    model.nvar = 7  # number of variables
    model.neq = 5  # number of equality constraints
    model.npar = 5  # number of runtime parameters

    # Objective function
    model.objective = obj
    model.objectiveN = objN  # increased costs for the last stage

    # We use an explicit RK4 integrator here to discretize continuous dynamics
    model.eq = lambda z: forcespro.nlp.integrate(
        continuous_dynamics,
        z[2:7],
        z[0:2],
        integrator=forcespro.nlp.integrators.RK4,
        stepsize=sampling_time,
    )
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((5, 2)), np.eye(5)], axis=1)

    # Inequality constraints
    model.lb = np.array(
        [
            -a_max,  # a
            -r_max,  # r
            -np.inf,  # X
            -np.inf,  # Y
            -np.inf,  # phi
            0.0,  # v
            -delta_max,  # delta
        ]
    )
    model.ub = np.array(
        [
            a_max,  # a
            r_max,  # r
            np.inf,  # X
            np.inf,  # Y
            np.inf,  # phi
            v_max,  # v
            delta_max,  # delta
        ]
    )

    # Initial condition on vehicle states x
    model.xinitidx = range(2, 7)

    # Solver generation
    # -----------------

    if from_directory:
        solver = forcespro.nlp.Solver.from_directory(
            os.path.join(os.path.dirname(__file__), "ihm_solver")
        )
    else:
        # Set solver options
        codeoptions = forcespro.CodeOptions("ihm_solver")
        codeoptions.printlevel = 0
        codeoptions.optlevel = 2
        codeoptions.sse = 1
        codeoptions.avx = 1
        codeoptions.optimize_choleskydivision = 1
        codeoptions.optimize_registers = 1
        codeoptions.optimize_uselocalsall = 1
        codeoptions.optimize_operationsrearrange = 1
        codeoptions.optimize_loopunrolling = 1
        codeoptions.optimize_enableoffset = 1

        codeoptions.cleanup = 0
        codeoptions.overwrite = 1
        codeoptions.timing = 1
        codeoptions.maxit = 200
        codeoptions.nlp.hessian_approximation = "bfgs"
        codeoptions.solvemethod = "SQP_NLP"
        codeoptions.nlp.bfgs_init = 2.5 * np.identity(7)
        codeoptions.sqp_nlp.maxqps = 1
        codeoptions.sqp_nlp.reg_hessian = 5e-9  # increase this if exitflag=-8

        solver = model.generate_solver(options=codeoptions)

    return model, solver


def updatePlots(x, u, pred_x, pred_u, model, k, next_path_points=None):
    """Deletes old data sets in the current plot and adds the new data sets
    given by the arguments x, u and predicted_z to the plot.
    x: matrix consisting of a set of state column vectors
    u: matrix consisting of a set of input column vectors
    pred_x: predictions for the next N state vectors
    pred_u: predictions for the next N input vectors
    model: model struct required for the code generation of FORCESPRO
    k: simulation step
    """
    fig = plt.gcf()
    ax_list = fig.axes

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
        range(k + 1, k + model.N), np.rad2deg(pred_x[2, 1:]), "g-"
    )  # plot new prediction of heading angle
    ax_list[2].plot(x[3, 0 : k + 2], "b-")  # plot new velocity
    ax_list[2].plot(
        range(k + 1, k + model.N), pred_x[3, 1:], "g-"
    )  # plot new prediction of velocity
    ax_list[3].plot(np.rad2deg(x[4, 0 : k + 2]), "b-")  # plot new steering angle
    ax_list[3].plot(
        range(k + 1, k + model.N), np.rad2deg(pred_x[4, 1:]), "g-"
    )  # plot new prediction of steering angle
    ax_list[4].step(
        range(0, k + 1), u[0, 0 : k + 1], "b-"
    )  # plot new acceleration force
    ax_list[4].step(
        range(k, k + model.N), pred_u[0, :], "g-"
    )  # plot new prediction of acceleration force
    ax_list[5].step(
        range(0, k + 1), np.rad2deg(u[1, 0 : k + 1]), "b-"
    )  # plot new steering rate
    ax_list[5].step(
        range(k, k + model.N), np.rad2deg(pred_u[1, :]), "g-"
    )  # plot new prediction of steering rate

    plt.pause(sim_pause)


def createPlot(
    x,
    u,
    start_pred,
    model,
    path_points,
    xinit,
    left_cones=None,
    right_cones=None,
):
    """Creates a plot and adds the initial data provided by the arguments"""

    # Create empty plot
    fig = plt.figure(figsize=(15, 7))
    plt.clf()
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
    (l1bis,) = ax_pos.plot(x[0, 0], x[1, 0], "gx")
    (l2,) = ax_pos.plot(x[0, 0], x[1, 0], "b-")
    (l3,) = ax_pos.plot(start_pred[2, :], start_pred[3, :], "g-")
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
    # plt.ylim([-500.0, 100.0])
    plt.xlim([0.0, sim_length - 1])
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.ub[4], model.ub[4]])), "r:"
    )
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.lb[4], model.lb[4]])), "r:"
    )
    ax_phi.plot(np.rad2deg(x[2, 0]), "b-")
    ax_phi.plot(np.rad2deg(start_pred[4, :]), "g-")

    # Plot velocity
    ax_vel = fig.add_subplot(gs[1, 1])
    plt.grid("both")
    plt.title("Velocity")
    plt.xlim([0.0, sim_length - 1])
    plt.plot([0, sim_length - 1], np.transpose([model.ub[5], model.ub[5]]), "r:")
    plt.plot([0, sim_length - 1], np.transpose([model.lb[5], model.lb[5]]), "r:")
    ax_vel.plot(0.0, x[3, 0], "-b")
    ax_vel.plot(start_pred[5, :], "g-")

    # Plot steering angle
    ax_delta = fig.add_subplot(gs[2, 1])
    plt.grid("both")
    plt.title("Steering angle")
    plt.xlim([0.0, sim_length - 1])
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.ub[6], model.ub[6]])), "r:"
    )
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.lb[6], model.lb[6]])), "r:"
    )
    ax_delta.plot(np.rad2deg(x[4, 0]), "b-")
    ax_delta.plot(np.rad2deg(start_pred[6, :]), "g-")

    # Plot acceleration force
    ax_a = fig.add_subplot(gs[3, 1])
    plt.grid("both")
    plt.title("Acceleration")
    plt.xlim([0.0, sim_length - 1])
    plt.plot([0, sim_length - 1], np.transpose([model.ub[0], model.ub[0]]), "r:")
    plt.plot([0, sim_length - 1], np.transpose([model.lb[0], model.lb[0]]), "r:")
    ax_a.step(0, u[0, 0], "b-")
    ax_a.step(range(model.N), start_pred[0, :], "g-")

    # Plot steering rate
    ax_r = fig.add_subplot(gs[4, 1])
    plt.grid("both")
    plt.title("Steering rate")
    plt.xlim([0.0, sim_length - 1])
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.ub[1], model.ub[1]])), "r:"
    )
    plt.plot(
        [0, sim_length - 1], np.rad2deg(np.transpose([model.lb[1], model.lb[1]])), "r:"
    )
    ax_r.step(0.0, np.rad2deg(u[1, 0]), "b-")
    ax_r.step(range(model.N), start_pred[1, :], "g-")

    plt.tight_layout()


def main():
    # generate the track and the reference_points
    if mercury:
        skidpad = Skidpad(
            track_width=1.5, inner_radius=7.625, N_loop=20, short=False
        )  # track for Mercury
    else:
        skidpad = Skidpad(
            track_width=1.0, inner_radius=4.0, N_loop=16, short=False
        )  # track for Minimercury

    X_ref = ca.interpolant(
        "X_ref",
        "bspline",
        [np.linspace(0, skidpad.total_track_length, skidpad.n_points())],
        skidpad.path_points[0, :],
    )
    Y_ref = ca.interpolant(
        "Y_ref",
        "bspline",
        [np.linspace(0, skidpad.total_track_length, skidpad.n_points())],
        skidpad.path_points[1, :],
    )

    # first reference points for the initial acceleration phase
    t_v_max = v_max / a_max
    theta_ref = (
        0.5 * a_max * np.square(np.arange(0, t_v_max, sampling_time, dtype=float))
    )
    # next reference points evenly spaced at max velocity
    theta_ref = np.concatenate(
        (
            theta_ref,
            np.arange(
                theta_ref[-1] + interval_length,
                skidpad.total_track_length,
                interval_length,
                dtype=float,
            ),
        )
    )

    path_points = np.zeros((2, theta_ref.shape[0]))
    for i in range(theta_ref.shape[0]):
        path_points[0, i] = X_ref(theta_ref[i])
        path_points[1, i] = Y_ref(theta_ref[i])

    # test the splines
    # theta_ref = np.linspace(0.0, skidpad.total_track_length, 10*skidpad.n_points())
    # plt.plot(skidpad.path_points[0, :], skidpad.path_points[1, :], "r+")
    # plt.plot(X_ref(theta_ref), Y_ref(theta_ref), "b+")
    # plt.axis("equal")
    # plt.show()
    # return ""

    # generate code for estimator
    model, solver = generate_path_planner(from_directory=from_dir)

    # Simulation ----------

    # Variables for storing simulation data
    x = np.zeros((5, sim_length + 1))  # states
    u = np.zeros((2, sim_length))  # inputs
    runtimes = np.zeros(sim_length)  # runtimes

    # Set initial conditions and initial guess to start solver from
    xinit = np.array([path_points[0, 0], path_points[1, 0], np.pi / 2, 0.0, 0.0])
    x[:, 0] = xinit
    last_idx = 0

    problem = {
        "x0": np.tile(np.concatenate((np.zeros(2), xinit), axis=0), model.N),
        "xinit": xinit,
    }

    start_pred = np.reshape(
        problem["x0"], (model.nvar, model.N), order="F"
    )  # first prediction corresponds to initial guess

    # generate plot with initial values
    createPlot(
        x,
        u,
        start_pred,
        model,
        path_points,
        xinit,
        skidpad.left_cones,
        skidpad.right_cones,
    )

    def final():
        print("Average iteration runtime : {} ms".format(np.mean(runtimes)))
        fig = plt.gcf()
        ax_list = fig.axes
        ax_list[0].get_lines().pop(-1).remove()  # remove old prediction of trajectory
        ax_list[0].legend(
            ["desired trajectory", "init pos", "car trajectory"], loc="lower right"
        )
        plt.show()

    # Simulation
    for k in range(sim_length):
        start = time()
        # Set initial condition
        problem["xinit"] = x[:, k]

        # Set runtime parameters (here, the next N points on the path)
        next_path_points, last_idx = extract_next_path_points(
            path_points, x[0:2, k], model.N, last_idx
        )
        if next_path_points.shape[1] != model.N:
            final()
            break

        problem["all_parameters"] = np.reshape(
            np.concatenate(
                (
                    next_path_points,
                    np.tile(
                        np.array(
                            [
                                [cost_deviation],
                                [cost_a],
                                [cost_r],
                            ]
                        ),
                        (1, model.N),
                    ),
                ),
                axis=0,
            ),
            (model.npar * model.N, 1),
            order="F",
        )

        # Time to solve the NLP!
        output, exitflag, info = solver.solve(problem)

        # Make sure the solver has exited properly.
        assert exitflag == 1, "bad exitflag : {}".format(exitflag)

        # Extract output
        temp = np.zeros((model.nvar, model.N))
        for i in range(0, model.N):
            temp[:, i] = output["x{0:02d}".format(i + 1)]
        pred_u = temp[0:2, :]
        pred_x = temp[2:7, :]

        stop = time()
        # sys.stderr.write("\tSimulation step took {} ms\n".format(1000 * (stop - start)))
        runtimes[k] = 1000 * (stop - start)

        # Apply optimized input u of first stage to system and save simulation data
        u[:, k] = pred_u[:, 0]
        x[:, k + 1] = np.transpose(model.eq(np.concatenate((u[:, k], x[:, k]))))

        # plot results of current simulation step
        updatePlots(
            x,
            u,
            pred_x,
            pred_u,
            model,
            k,
            next_path_points if show_next_path_points else None,
        )

        if k == sim_length - 1:
            final()
        else:
            plt.draw()


if __name__ == "__main__":
    main()
