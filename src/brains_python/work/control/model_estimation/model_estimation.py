import matplotlib.pyplot as plt
from casadi import *
import numpy as np
from data_visualization import *

sampling_time = 0.01
np.random.seed(127)


def plot_everything(true_x: np.ndarray, predicted_x: np.ndarray, _u: np.ndarray):
    simulation_plot = Plot(
        row_nbr=2,
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
        row_idx=range(2),
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
        row_idx=0,
        col_idx=2,
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
        row_idx=1,
        col_idx=2,
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
    simulation_plot.plot(show=False, save_path=None)


true_states = np.loadtxt("states.csv", delimiter=",", skiprows=1)[:, :4]
ref_diffs = np.diff(true_states[:, 2])
ref_diffs[ref_diffs > 1.5 * np.pi] -= 2 * np.pi
ref_diffs[ref_diffs < -1.5 * np.pi] += 2 * np.pi
true_states[:, 2] = np.insert(
    true_states[0, 2] + np.cumsum(ref_diffs),
    0,
    true_states[0, 2],
)
controls = np.loadtxt("controls.csv", delimiter=",", skiprows=1)
# plot_everything(true_x=true_states, predicted_x=true_states, _u=controls)
# plt.show()

X = MX.sym("X")
Y = MX.sym("Y")
phi = MX.sym("phi")
v_x = MX.sym("v_x")
T = MX.sym("T")
delta = MX.sym("delta")
x = vertcat(X, Y, phi, v_x)
u = vertcat(T, delta)

# m = MX.sym("m")
# l_r = MX.sym("l_r")
# l_f = MX.sym("l_f")
m = 260.0
l_f = 0.4
l_r = 0.8
C_m = MX.sym("C_m")
C_r0 = MX.sym("C_r0")
C_r2 = MX.sym("C_r2")
p = vertcat(
    # m,
    # l_r,
    # l_f,
    C_m,
    C_r0,
    C_r2,
)

F_x = C_m * T - C_r0 - C_r2 * v_x**2
beta = atan(tan(delta) * l_r / (l_r + l_f))
f_cont = Function(
    "ode",
    [x, u, p],
    [
        vertcat(
            v_x * cos(phi + beta),
            v_x * sin(phi + beta),
            v_x * tan(delta) / (l_r + l_f),
            F_x / m,
        )
    ],
)

num_nodes = 5
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
    global true_states, controls
    _true_states = true_states[start:end, :].T
    _controls = controls[start:end, :].T
    N = _true_states.shape[1]
    if strategy == "multiple_shooting":
        X = MX.sym("X", x.size()[0], N)
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
            {"ipopt": {"max_iter": 50, "sb": "yes", "linear_solver": "ma57"}},
        )
        p_guess = np.array(
            [
                # 2.5e02,
                # 8.0e-01,
                # 4.0e-01,
                3600.0,
                5.0,
                2,
            ]
        )
        X_guess = _true_states
        sol = solver(
            x0=veccat(p_guess, X_guess),
            lbg=0,
            ubg=0,
            lbx=[
                # 200,
                # 0.5,
                # 0.2,
                3500,
                0.0,
                1.0,
            ]
            + [-inf] * x.size()[0] * N,
            ubx=[
                # 300,
                # 1.0,
                # 1,
                3700,
                10,
                3,
            ]
            + [inf] * x.size()[0] * N,
        )
        determined_params = sol["x"][: p.size()[0]].full().flatten()
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


# determined_params = fit(0, 3000, "multiple_shooting")
determined_params = np.array(
    [
        # 255.0,
        # 0.8,
        # 0.35,
        3550.0,
        0.0,
        3.5,
    ]
)
true_states = true_states[:]
controls = controls[:]
bruh = f_disc.mapaccum("all_samples", true_states.shape[0] - 1)
predicted_states = (
    bruh(
        true_states[0, :],
        controls[1:, :].T,
        repmat(determined_params.T, 1, true_states.shape[0] - 1),
    )
    .full()
    .T
)
plot_everything(
    true_x=true_states[:-1],
    predicted_x=predicted_states,
    _u=controls[:-1],
)
plt.show()
