# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import json
import os
from enum import Enum
from time import time
from typing import Tuple

import casadi as ca

try:
    from forcespro import CodeOptions
    from forcespro.nlp import SymbolicModel, Solver
    from forcespro.dump import load
except ImportError:
    print(
        "forcespro wasn't found in the PYTHONPATH env variable, please add it if you really want to use IHM."
    )
    Solver = None
    SymbolicModel = None
    load = lambda: None

import numpy as np


class PhysicalModel(Enum):
    KIN_4 = 0
    KIN_6 = 1
    DYN_6 = 2
    KIN_DYN_6 = 3

    def __str__(self):
        return self.name

    def __int__(self):
        return 4 if self.value == 0 else 6


def obj(z: ca.SX, p: ca.SX) -> ca.SX:
    """
    Least square costs on deviating from the path
    stage variable : z = [dT, ddelta, X, Y, phi, v_x, T, delta]
    or
    stage variable : z = [dT, ddelta, X, Y, phi, v_x, v_y, r, T, delta]
    parameter vector : p = [X_ref, Y_ref, q_d, q_dT, q_ddelta, m, l_R, l_F, C_m, C_r0, C_r2]
    or
    parameter vector : p = [X_ref, Y_ref, q_d, q_dT, q_ddelta, m, l_R, l_F, C_m, C_r0, C_r2, B, C, D, I_z]
    """
    return (
        p[2] * (z[2] - p[0]) ** 2  # costs on deviating on the path in x-direction
        + p[2] * (z[3] - p[1]) ** 2  # costs on deviating on the path in y-direction
        + p[3] * z[0] ** 2  # penalty on input dT
        + p[4] * z[1] ** 2  # penalty on input ddelta
    )


def objN(z: ca.SX, p: ca.SX) -> ca.SX:
    """Increased east square costs on deviating from the path"""
    return 10.0 * obj(z, p)


def KIN_4_cont_dynamics(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """
    Defines dynamics of the car, i.e. equality constraints, using KIN_4 model.
    Args:
    -----
    :param x: state [X, Y, phi, v_x, T, delta]
    :param u: input [dT, ddelta]
    :param p: parameters [X_ref, Y_ref, q_d, q_dT, q_ddelta, m, l_r, l_f, C_m, C_r0, C_r2]
    """
    # set intermediate parameters
    beta = ca.arctan(
        (p[6] / (p[6] + p[7])) * ca.tan(x[5])
    )  # beta = arctan(l_r/(l_r+l_f) * tan(delta))
    F_x = p[8] * x[4] - p[9] - p[10] * x[3] ** 2  # F_x = C_m * T - C_r0 - C_r2 * v_x^2
    # - p[9] * ca.if_else(ca.fabs(x[3]) >= 1.0e-3, 1.0, 0.0)

    # compute dx/dt
    return ca.vertcat(
        x[3] * ca.cos(x[2] + beta),  # dX/dt = v_x*cos(phi+beta)
        x[3] * ca.sin(x[2] + beta),  # dY/dt = v_x*sin(phi+beta)
        (x[3] / p[6]) * ca.sin(beta),  # dphi/dt = v_x/l_r*sin(beta)
        ((p[8] * x[4] - p[9] - p[10] * x[3] ** 2)) / p[5],  # dv_x/dt = F_x / m
        u[0],  # dT/dt = dT
        u[1],  # ddelta/dt = ddelta
    )


def KIN_6_cont_dynamics(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """
    Defines dynamics of the car, i.e. equality constraints, using KIN_6 model.
    Args:
    -----
    :param x: state [X, Y, phi, v_x, v_y, r, T, delta]
    :param u: input [dT, ddelta]
    :param p: parameters [X_ref, Y_ref, q_d, q_dT, q_ddelta, m, l_r, l_f, C_m, C_r0, C_r2]
    """
    F_x = (
        p[8] * x[8]
        - p[9] * ca.if_else(ca.fabs(x[3]) >= 1.0e-3, 1.0, 0.0)
        - p[10] * x[3] ** 2
    )  # F_x = C_m * T - C_r0 - C_r2 * v_x^2

    # compute dx/dt
    return ca.vertcat(
        x[3] * ca.cos(x[2]) - x[4] * ca.sin(x[2]),
        x[3] * ca.sin(x[2]) + x[4] * ca.cos(x[2]),
        x[5],
        F_x / p[5],
        (u[1] * x[3] + x[7] * F_x / p[7]) * p[6] / (p[6] + p[7]),
        (u[1] * x[3] + x[7] * F_x / p[7]) / (p[6] + p[7]),
        u[0],
        u[1],
    )


def DYN_6_cont_dynamics(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """
    Defines dynamics of the car, i.e. equality constraints, using DYN_6 model.
    Args:
    -----
    :param x: state [X, Y, phi, v_x, v_y, r, T, delta]
    :param u: input [dT, ddelta]
    :param p: parameters [X_ref, Y_ref, q_d, q_dT, q_ddelta, m, l_r, l_f, C_m, C_r0, C_r2, B, C, D, I_z]
    """
    F_x = (
        p[8] * x[6]
        - p[9] * ca.if_else(ca.fabs(x[3]) >= 1.0e-3, 1.0, 0.0)
        - p[10] * x[3] ** 2
    )  # F_x = C_m * T - C_r0 - C_r2 * v_x^2
    F_R_y = ca.if_else(
        ca.fabs(x[3]) >= 1.0e-3,
        p[13]
        * ca.sin(p[12] * ca.arctan(p[11] * ca.arctan((x[4] - p[6] * x[5]) / x[3]))),
        0.0,
    )
    F_F_y = ca.if_else(
        ca.fabs(x[3]) >= 1.0e-3,
        p[13]
        * ca.sin(
            p[12] * ca.arctan(p[11] * ca.arctan((x[4] + p[7] * x[5]) / x[3]) - x[7])
        ),
        0.0,
    )

    # compute dx/dt
    return ca.vertcat(
        x[3] * ca.cos(x[2]) - x[4] * ca.sin(x[2]),
        x[3] * ca.sin(x[2]) + x[4] * ca.cos(x[2]),
        x[5],
        (F_x - F_F_y * ca.sin(x[7]) + p[5] * x[4] * x[5]) / p[5],
        (F_R_y + F_F_y * ca.cos(x[7]) - p[5] * x[3] * x[5]) / p[5],
        (F_F_y * p[5] * ca.cos(x[7]) - F_R_y * p[6]) / p[14],
        u[0],
        u[1],
    )


def generate_solver(
    horizon_size: int,
    sampling_time: float,
    physical_model: PhysicalModel,
    **kwargs,
) -> None:
    """
    Generates a forcespro solver in the current directory and attaches a JSON file
    containing the problem formulation.
    """
    solver_name = "ihm_solver_" + str(physical_model)
    # Model Definition =========================================================
    model = SymbolicModel()
    model.N = horizon_size  # horizon length
    # Objective function
    model.objective = obj
    model.objectiveN = objN  # increased costs for the last stage
    if physical_model == PhysicalModel.KIN_4:
        # Problem dimensions
        model.nvar = 8
        model.neq = 6
        model.npar = 11

        # Model (continuous) dynamics
        model.continuous_dynamics = KIN_4_cont_dynamics

        # Inequality constraints as runtime parameters: dT, ddelta, v_x, T, delta
        model.lbidx = [0, 1, 5, 6, 7]
        model.ubidx = [0, 1, 5, 6, 7]

    elif physical_model == PhysicalModel.KIN_6:
        # Problem dimensions
        model.nvar = 10
        model.neq = 8
        model.npar = 11

        # Model (continuous) dynamics
        model.continuous_dynamics = KIN_6_cont_dynamics

        # Inequality constraints as runtime parameters: dT, ddelta, v_x, T, delta
        model.lbidx = [0, 1, 5, 8, 9]
        model.ubidx = [0, 1, 5, 8, 9]

    elif physical_model == PhysicalModel.DYN_6:
        # Problem dimensions
        model.nvar = 10
        model.neq = 8
        model.npar = 15

        # Model (continuous) dynamics
        model.continuous_dynamics = DYN_6_cont_dynamics

        # Inequality constraints as runtime parameters
        model.lbidx = [0, 1, 5, 8, 9]
        model.ubidx = [0, 1, 5, 8, 9]

    elif physical_model == PhysicalModel.KIN_DYN_6:
        raise ValueError("{} model not implemented yet".format(str(physical_model)))
    else:
        raise ValueError("Unknown physical model")

    # Initial condition on vehicle states x
    model.xinitidx = range(model.nvar - model.neq, model.nvar)
    # matrix defining inter-stage equality constraints
    model.E = np.concatenate(
        [np.zeros((model.neq, model.nvar - model.neq)), np.eye(model.neq)], axis=1
    )

    # Solver generation and model dumping ======================================

    # Set solver options
    codeoptions = CodeOptions(solver_name)
    if kwargs.get("verbose", False):
        codeoptions.printlevel = 2
    else:
        codeoptions.printlevel = 0
    if kwargs.get("opt", False) or kwargs.get("optim", False):
        codeoptions.optlevel = 3
        if kwargs.get("arm", False) or kwargs.get("jetson", False):
            codeoptions.optimize_choleskydivision = 1
            codeoptions.optimize_registers = 1
            codeoptions.optimize_uselocalsall = 1
            codeoptions.optimize_operationsrearrange = 1
            codeoptions.optimize_loopunrolling = 1
            codeoptions.optimize_enableoffset = 1
        else:
            codeoptions.sse = 1
            codeoptions.avx = 1
    else:
        codeoptions.optlevel = 0

    if kwargs.get("arm", False) or kwargs.get("jetson", False):
        codeoptions.cleanup = 0
    else:
        codeoptions.cleanup = 1

    parallel = kwargs.get("parallel", 0)
    if parallel > 0:
        codeoptions.parallel = parallel

    codeoptions.overwrite = 1
    codeoptions.timing = 1
    codeoptions.maxit = 200
    codeoptions.nlp.hessian_approximation = "bfgs"
    codeoptions.solvemethod = "SQP_NLP"
    codeoptions.nlp.bfgs_init = 2.5 * np.identity(model.nvar)
    codeoptions.sqp_nlp.maxqps = 1
    codeoptions.sqp_nlp.reg_hessian = 5e-9  # increase this if exitflag=-8
    codeoptions.nlp.integrator.type = "ERK4"
    codeoptions.nlp.integrator.Ts = sampling_time
    codeoptions.nlp.integrator.nodes = 5
    codeoptions.nlp.stack_parambounds = True

    start = time()
    model.generate_solver(options=codeoptions)
    stop = time()
    print(f"Solver generation took {stop - start} seconds")

    start = time()
    # remove previous model files
    for file in os.listdir(solver_name):
        if file.startswith(solver_name) and file.endswith(".json"):
            os.remove(os.path.join(solver_name, file))

    # dump initial model
    tag, full_filename = forcespro.dump.save_formulation(
        model, codeoptions, outputs=None, label=None, path=solver_name
    )

    # add sampling_time and physmodel to the json file
    infile = open(full_filename, "r")
    data = json.load(infile)
    data["model"]["sampling_time"] = sampling_time
    data["model"]["physical_model"] = str(physical_model)
    outputfile = open(full_filename, "w")
    outputfile.write(json.dumps(data, indent=4))
    stop = time()
    print(f"Model dumping took {stop - start} seconds")


def load_solver(
    solver_dir_path: str,
) -> Tuple[SymbolicModel, Solver, int, float, PhysicalModel]:
    """
    Loads a forcespro solver for IHM from a directory, along with the model used to
    generate it, the horizon size, the sampling time and the physical model used.
    """
    solver = Solver.from_directory(solver_dir_path)
    model_filename = ""
    for file in os.listdir(solver_dir_path):
        if file.startswith(solver.name) and file.endswith(".json"):
            model_filename = os.path.join(solver_dir_path, file)
            break
    if model_filename == "":
        raise ValueError("No model file found in solver directory")

    model, options, _, _, _ = load(model_filename)
    with open(model_filename, "r") as f:
        data = json.load(f)
        N = data["model"]["N"]
        sampling_time = data["model"]["sampling_time"]
        physmodel = PhysicalModel[data["model"]["physical_model"]]

    return model, solver, N, sampling_time, physmodel
