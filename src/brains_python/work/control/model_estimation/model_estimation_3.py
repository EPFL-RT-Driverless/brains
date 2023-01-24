from casadi import *
import numpy as np
from data_visualization import *
from matplotlib import pyplot as plt

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
                "curve_style": CurvePlotStyle.STEP,
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
                "curve_style": CurvePlotStyle.STEP,
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
# plot_everything(
#     true_x=true_states,
#     predicted_x=true_states,
#     _u=controls,
#     _du=control_derivatives,
# )

X = MX.sym("X")
Y = MX.sym("Y")
phi = MX.sym("phi")
v_x = MX.sym("v_x")
v_y = MX.sym("v_y")
r = MX.sym("r")
T = MX.sym("T")
delta = MX.sym("delta")
x = vertcat(X, Y, phi, v_x, v_y, r)
u = vertcat(T, delta)

# m = MX.sym("m")
# l_r = MX.sym("l_r")
# l_f = MX.sym("l_f")
m = 255.0
l_r = 0.8
l_f = 0.35
C_m = MX.sym("C_m")
C_r0 = MX.sym("C_r0")
C_r2 = MX.sym("C_r2")
B_r = MX.sym("B")
C_r = MX.sym("C")
D_r = MX.sym("D")
B_f = MX.sym("B")
C_f = MX.sym("C")
D_f = MX.sym("D")
I_z = MX.sym("I_z")
p = vertcat(
    # m,
    # l_r,
    # l_f,
    C_m,
    C_r0,
    C_r2,
    B_r,
    C_r,
    D_r,
    B_f,
    C_f,
    D_f,
    I_z,
)

F_r_x = C_m * T - C_r0 - C_r2 * v_x**2
F_f_x = C_m * T - C_r0 - C_r2 * v_x**2

alpha_R = atan((v_y - l_r * r) / v_x)
alpha_F = atan((v_y + l_f * r) / v_x) - delta
F_f_z = m * 9.8 * l_f / (l_f + l_r)
F_r_z = m * 9.8 * l_r / (l_f + l_r)
F_f_y = -F_f_z * D_f * sin(C_f * atan(B_f * alpha_F))
F_r_y = -F_r_z * D_r * sin(C_r * atan(B_r * alpha_R))
a_x = (F_r_x - F_f_y * sin(delta) + F_f_x * cos(delta) + m * v_y * r) / m
a_y = (F_r_y + F_f_y * cos(delta) + F_f_x * sin(delta) - m * v_x * r) / m
rdot = (l_f * F_f_y * cos(delta) + l_f * F_f_x * sin(delta) - l_r * F_r_y) / I_z

f_cont = Function(
    "ode",
    [x, u, p],
    [
        vertcat(
            v_x * cos(phi) - v_y * sin(phi),
            v_x * sin(phi) + v_y * cos(phi),
            r,
            a_x,
            a_y,
            rdot,
        )
    ],
)
bruh = Function(
    "bruh",
    [x, u, p],
    [
        rdot,
        F_r_x,
        F_r_y,
        F_f_x,
        F_f_y,
    ],
)

num_nodes = 2
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
    _true_states = true_states[start:end, : x.size()[0]].T
    _controls = controls[start:end, :].T
    N = _true_states.shape[1]
    if strategy == "multiple_shooting":
        # X = MX.sym("X", x.size()[0])
        # g = [X - _true_states[:, 0]]
        # L = norm_2(X - _true_states[:, 0])
        # for i in range(1, N):
        #     Xnext = f_disc(X, _controls[:, i], p)
        #     g.append(Xnext - _true_states[:, i + 1])
        #     X = MX.sym("X", x.size()[0])
        #     L += norm_2(X - _true_states[:, i])
        #
        # X = MX.sym("X", x.size()[0], N)
        # Xnext = f_disc.map(N - 1, "openmp")(
        #     X[:, :-1], _controls[:, 1:], repmat(p, 1, N - 1)
        # )
        # gapps = Xnext - X[:, 1:]
        # x0 = _true_states[:, 0]
        # R = Xnext - _true_states[:, :1]
        X = MX.sym("X", x.size()[0], N)
        Xnext = f_disc.map(N - 1, "openmp")(
            X[:, :-1], _controls[:, 1:], repmat(p, 1, N - 1)
        )
        gapps = Xnext - X[:, 1:]
        x0 = _true_states[:, 0]
        R = Xnext - _true_states[:, 1:]
        g = [X[:, 0] - x0, gapps]
        L = norm_2(R)
        nlp = {"x": veccat(p, X), "f": L, "g": vertcat(*g)}

        solver = nlpsol(
            "sysid",
            "ipopt",
            nlp,
            {"ipopt": {"max_iter": 100, "sb": "yes", "linear_solver": "ma97"}},
        )
        p_guess = np.array(
            [
                # 255,  # m
                # 0.8,  # l_r
                # 0.4,  # l_f
                400.0,  # C_m
                0.0,  # C_r0
                0.3,  # C_r2
                17.0,  # B_r
                1.5,  # C_r
                1.66,  # D_r
                17.15,  # B_f
                1.9,  # C_f
                1.63,  # D_f
                500,  # I_z
                # 2.30000020e02,
                # 9.99999996e-01,
                # 2.00000000e-01,
                # 6.39039764e02,
                # 1.00000000e02,
                # 9.99999980e00,
                # 2.00000000e01,
                # 2.99999691e00,
                # 1.00004906e00,
                # 1.99999988e01,
                # 2.99999997e00,
                # 1.00001165e00,
                # 2.98993636e02,
            ]
        )
        X_guess = _true_states
        sol = solver(
            x0=veccat(p_guess, X_guess),
            lbg=0,
            ubg=0,
            lbx=[
                # 230,  # m
                # 0.5,  # l_r
                # 0.2,  # l_f
                300,  # C_m
                0.0,  # C_r0
                0.1,  # C_r2
                12,  # B_r
                1.0,  # C_r
                1.0,  # D_r
                12,  # B_f
                1.0,  # C_f
                1.0,  # D_f
                200,  # I_z
            ]
            + [-inf] * 6 * N,
            ubx=[
                # 300,  # m
                # 1.0,  # l_r
                # 0.6,  # l_f
                2000,  # C_m
                100,  # C_r0
                10,  # C_r2
                20,  # B_r
                3,  # C_r
                6,  # D_r
                20,  # B_f
                3,  # C_f
                6,  # D_f
                600,  # I_z
            ]
            + [inf] * 6 * N,
        )
        determined_params = sol["x"][:13].full().flatten()
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


# determined_params = fit(1000, 2000, "multiple_shooting")
determined_params = np.array(
    [
        [
            # 255,  # m
            # 0.8,  # l_r
            # 0.4,  # l_f
            3550.0,  # C_m
            0,  # C_r0
            3.5,  # C_r2
            20.0,  # B_r
            0.8,  # C_r
            4.0,  # D_r
            17.0,  # B_f
            2.0,  # C_f
            4.2,  # D_f
            400,  # I_z
        ]
    ]
)
# [2.00000000e+02 5.00000000e-01 1.00000000e+00 1.00000000e+03
#  1.00000000e+02 1.00000000e+01 2.00000000e+01 1.15679857e+00
#  3.00000000e+00 1.00000000e+02]
true_states = true_states[1000:3000]
controls = controls[1000:3000]
control_derivatives = control_derivatives[1000:3000]
bruh2 = f_disc.mapaccum("all_samples", true_states.shape[0] - 1)
predicted_states = (
    bruh2(
        true_states[0, :],
        controls[1:, :].T,
        repmat(determined_params.T, 1, true_states.shape[0] - 1),
    )
    .full()
    .T
)
rdots = []
F_r_x = []
F_f_x = []
F_r_y = []
F_f_y = []
for i in range(true_states.shape[0] - 1):
    rdot_, F_r_x_, F_r_y_, F_f_x_, F_f_y_ = bruh(
        true_states[i, :], controls[i + 1, :], determined_params.T
    )
    rdots.append(float(rdot_))
    F_r_x.append(float(F_r_x_))
    F_r_y.append(float(F_r_y_))
    F_f_x.append(float(F_f_x_))
    F_f_y.append(float(F_f_y_))

plt.subplots(3, 2)
plt.subplot(3, 2, 1)
plt.plot(rdots)
plt.title("rdot")
plt.subplot(3, 2, 2)
plt.plot(F_r_x)
plt.title("F_r_x")
plt.subplot(3, 2, 3)
plt.plot(F_r_y)
plt.title("F_r_y")
plt.subplot(3, 2, 4)
plt.plot(F_f_x)
plt.title("F_f_x")
plt.subplot(3, 2, 5)
plt.plot(F_f_y)
plt.title("F_f_y")
plt.tight_layout()


plot_everything(
    true_x=true_states[:-1],
    predicted_x=predicted_states,
    _u=controls[:-1],
    _du=control_derivatives[:-1],
)
