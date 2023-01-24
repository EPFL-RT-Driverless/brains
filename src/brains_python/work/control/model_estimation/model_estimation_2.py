import matplotlib.pyplot as plt
from casadi import *
import numpy as np
from data_visualization import *

sampling_time = 0.01
np.random.seed(127)


def plot_everything(
    true_x: np.ndarray, predicted_x: np.ndarray, _u: np.ndarray, _du: np.ndarray
):
    simulation_plot = Plot(
        row_nbr=4,
        col_nbr=3,
        mode=PlotMode.STATIC,
        sampling_time=sampling_time,
        interval=1,
        figsize=(15, 8),
        port=5002,
        verbose=False,
        show_car=False,
    )
    simulation_plot.add_subplot(
        row_idx=range(4),
        col_idx=0,
        subplot_name="map",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves={
            "true_trajectory": {
                "data": true_x[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
            "predicted_trajectory": {
                "data": predicted_x[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 1, "zorder": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name="phi",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            "true_phi": {
                "data": np.rad2deg(true_x[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
            "predicted_phi": {
                "data": np.rad2deg(predicted_x[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name="v_x",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            "true_v_x": {
                "data": true_x[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
            "predicted_v_x": {
                "data": predicted_x[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=1,
        subplot_name="T",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=False,
        curves={
            "T": {
                "data": _u[:, 0],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=3,
        col_idx=1,
        subplot_name="delta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            "delta": {
                "data": np.rad2deg(_u[:, 1]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name="r",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            "true_r": {
                "data": np.rad2deg(true_x[:, 5]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
            "predicted_r": {
                "data": np.rad2deg(predicted_x[:, 5]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name="v_y",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            "true_v_y": {
                "data": true_x[:, 4],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
            "predicted_v_y": {
                "data": predicted_x[:, 4],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=2,
        subplot_name="dT",
        subplot_type=SubplotType.TEMPORAL,
        unit="1/s",
        show_unit=False,
        curves={
            "dT": {
                "data": _du[:, 0],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=3,
        col_idx=2,
        subplot_name="ddelta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            "ddelta": {
                "data": np.rad2deg(_du[:, 1]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "green", "linewidth": 1},
            },
        },
    )
    simulation_plot.plot(show=True, save_path=None)


true_states = np.loadtxt("states.csv", delimiter=",", skiprows=1)
ref_diffs = np.diff(true_states[:, 2])
ref_diffs[ref_diffs > 1.5 * np.pi] -= 2 * np.pi
ref_diffs[ref_diffs < -1.5 * np.pi] += 2 * np.pi
true_states[:, 2] = np.insert(
    true_states[0, 2] + np.cumsum(ref_diffs),
    0,
    true_states[0, 2],
)
controls = np.loadtxt("controls.csv", delimiter=",", skiprows=1)
control_derivatives = np.loadtxt("control_derivatives.csv", delimiter=",", skiprows=1)
# plot_everything(true_x=true_states, predicted_x=true_states, _u=controls, _du=control_derivatives)
# plt.show()

X = MX.sym("X")
Y = MX.sym("Y")
phi = MX.sym("phi")
v_x = MX.sym("v_x")
v_y = MX.sym("v_y")
r = MX.sym("r")
T = MX.sym("T")
delta = MX.sym("delta")
dT = MX.sym("dT")
ddelta = MX.sym("ddelta")
x = vertcat(X, Y, phi, v_x, v_y, r, T, delta)
u = vertcat(dT, ddelta)

m = MX.sym("m")
l_r = MX.sym("l_r")
l_f = MX.sym("l_f")
C_m = MX.sym("C_m")
C_r0 = MX.sym("C_r0")
C_r2 = MX.sym("C_r2")
p = vertcat(m, l_r, l_f, C_m, C_r0, C_r2)

F_x = C_m * T - C_r0 - C_r2 * v_x**2
f_cont = Function(
    "ode",
    [x, u, p],
    [
        vertcat(
            v_x * cos(phi) - v_y * sin(phi),
            v_x * sin(phi) + v_y * cos(phi),
            r,
            F_x / m,
            (ddelta * v_x + delta * F_x / m) * l_r / (l_r + l_f),
            (ddelta * v_x + delta * F_x / m) * 1 / (l_r + l_f),
            dT,
            ddelta,
        )
    ],
)

num_nodes = 4
dt = sampling_time / num_nodes
new_state = x
for i in range(num_nodes):
    k1 = f_cont(new_state, u, p)
    k2 = f_cont(new_state + dt / 2 * k1, u, p)
    k3 = f_cont(new_state + dt / 2 * k2, u, p)
    k4 = f_cont(new_state + dt * k3, u, p)
    new_state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

f_disc = Function("f", [x, u, p], [new_state])


def fit(start: int, end: int, strategy: str = "multiple_shooting"):
    global true_states, controls, control_derivatives
    _true_states = np.array([true_states[start:end, :].T, controls[start:end, :].T])
    _controls = control_derivatives[start:end, :].T
    N = _true_states.shape[1]
    if strategy == "multiple_shooting":
        X = MX.sym("X", 8, N)
        Xnext = f_disc.map(N - 1, "openmp")(
            X[:, :-1], _controls[:, 1:], repmat(p, 1, N - 1)
        )
        gapps = Xnext - X[:, 1:]
        x0 = _true_states[:, 0]
        R = Xnext - _true_states[:, :1]
        solver = nlpsol(
            "sysid",
            "ipopt",
            {"x": veccat(p, X), "f": dot(R, R), "g": vertcat(vec(gapps), x0 - X[:, 0])},
            {"ipopt": {"max_iter": 200, "sb": "yes"}},
        )
        p_guess = np.array([2.5e02, 8.0e-01, 4.0e-01, 3500.0, 54.0, 3.0])
        X_guess = _true_states
        sol = solver(
            x0=veccat(p_guess, X_guess),
            lbg=0,
            ubg=0,
            lbx=[
                200,
                0.5,
                0.2,
                1000,
                1e-6,
                1e-6,
            ]
            + [-inf] * 8 * N,
            ubx=[
                300,
                1.0,
                1,
                2000,
                100,
                10,
            ]
            + [inf] * 8 * N,
        )
        determined_params = sol["x"][:6].full().flatten()
        print("error = ", sol["f"])
        print("determined_params = ", determined_params)
    else:
        # single shooting method
        # trajectory = f_disc.mapaccum("all_samples", N)
        #
        #
        # if Importer.has_plugin("clang"):
        #     with_jit = True
        #     compiler = "clang"
        # elif Importer.has_plugin("shell"):
        #     with_jit = True
        #     compiler = "shell"
        # else:
        #     print("WARNING; running without jit. This may result in very slow evaluation times")
        #     with_jit = False
        #     compiler = ""
        #
        # x0 = np.array([0, 0, pi / 2, 0, 0, 0])
        # R = trajectory(x0, _controls.T, p) - _true_states.T
        # solver = nlpsol(
        #     "sysid",
        #     "ipopt",
        #     {"x": p, "f": dot(R, R), "g": []},
        #     {
        #         # "jit": with_jit,
        #         # "jit_options": {"compiler": compiler},
        #         "print_time": True,
        #         "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 1000},
        #     },
        # )
        # sol = solver(
        #     x0=DM(
        #         [
        #             255.0,  # m
        #             0.8,  # l_r
        #             0.4,  # l_f
        #             1845.0,  # C_m
        #             54.0,  # C_r0
        #             0.837,  # C_r2
        #             0.255,  # B
        #             1.421,  # C
        #             1.595,  # D
        #             600.0,  # I_z
        #         ]
        #     ),
        #     lbg=0,
        #     ubg=0,
        #     lbx=0,
        #     ubx=2000,
        # )
        # determined_params = sol["x"].full().flatten()
        # predicted_states = trajectory(x0, _controls.T, determined_params).full().T
        # print("error = ", sol["f"])
        # print("determined_params = ", determined_params)
        pass
    return determined_params


# determined_params = fit(0, 1000, "multiple_shooting")
determined_params = np.array([255.0, 0.8, 0.2, 3100.0, 120.0, 1.5])
true_states = true_states[:]
controls = controls[:]
bruh = f_disc.mapaccum("all_samples", true_states.shape[0] - 1)
predicted_states = (
    bruh(
        np.append(true_states[0, :], controls[0, :]),
        control_derivatives[1:, :].T,
        repmat(determined_params.T, 1, true_states.shape[0] - 1),
    )
    .full()
    .T
)
plot_everything(
    true_x=true_states[:-1],
    predicted_x=predicted_states[:, :6],
    _u=controls[:-1],
    _du=control_derivatives[1:, :],
)
plt.show()
