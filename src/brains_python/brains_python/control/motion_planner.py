# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from typing import Tuple, Optional, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import trajectory_planning_helpers as tph
from brains_python.common import Mission
from brains_python.control.controller import CarParams
from strongpods import PODS

__all__ = ["MotionPlanner", "MotionPlannerParams"]


class MotionPlannerParams(PODS):
    """PODS (Plain Old Data Structure) containing all the parameters used by the MotionPlanner class."""

    # track info
    center_points: np.ndarray  # points defining the center line of the track, shape (N, 2)
    widths: np.ndarray  # right and left widths (in this order) of the track at each
    # center point, shape (N, 2)
    # the provided arrays above should always be specified as unclosed,
    # i.e. the last and first point are different
    psi_s: float = np.pi / 2  # the orientation angle of the center line at the start of
    # the track (in radians). Only needed if the track is not closed.
    psi_e: float = np.pi / 2  # the orientation angle of the center line at the start of
    # the track (in radians). Only needed if the track is not closed.

    # algorithm behavior
    mission: Mission  # Mission to be performed, i.e. the FS event
    additional_attributes: Optional[list]  # the names of the attributes that should
    # be kept in memory. Possible names are : center_points, center_right_widths,
    # center_left_widths, boundaries, reference_curvature, reference_time,
    # right_width_vs_time, left_width_vs_time

    def __init__(self, **kwargs):
        current_params, remaining_params = MotionPlannerParams._transform_dict(kwargs)
        for key, val in current_params.items():
            setattr(self, key, val)
        super().__init__(**remaining_params)


class MotionPlanner:
    """
    Class encapsulating the motion planning procedure (trajectory generation and
    velocity planning).
    """

    # specified at initialization
    motion_planner_params: MotionPlannerParams
    car_params: CarParams

    # reference values
    reference_points: np.ndarray
    reference_heading: np.ndarray
    reference_velocity: np.ndarray
    reference_arc_lengths: np.ndarray

    # linear spline interpolation
    X_ref_vs_arc_length: InterpolatedUnivariateSpline
    Y_ref_vs_arc_length: InterpolatedUnivariateSpline
    phi_ref_vs_arc_length: InterpolatedUnivariateSpline
    v_x_ref_vs_arc_length: InterpolatedUnivariateSpline
    time_vs_arc_length: InterpolatedUnivariateSpline
    arc_length_vs_time: InterpolatedUnivariateSpline
    # X_ref_vs_time: InterpolatedUnivariateSpline
    # Y_ref_vs_time: InterpolatedUnivariateSpline

    # misc
    total_length: float  # total length of the reference trajectory
    total_time: float  # total race time of the reference trajectory
    key_points: list[tuple[float, np.ndarray]]
    additional_attributes: dict[str, Union[np.ndarray, InterpolatedUnivariateSpline]]

    def __init__(
        self,
        motion_planner_params: MotionPlannerParams,
        car_params: CarParams,
        **kwargs,
    ):
        """
        Initialize the MotionPlanner class using the provided parameters.
        In particular must be provided track data (coordinates of points defining the
        centre line and left and right track widths at those points), the mission type
        (which FS event) and the specifics of the wanted procedure (e.g. whether to use
        curvature minimization or not).
        """
        self.motion_planner_params = motion_planner_params
        if self.motion_planner_params.additional_attributes is None:
            self.motion_planner_params.additional_attributes = []
        self.car_params = car_params
        self.additional_attributes = {}

        # input verification ======================================================
        if not self.closed:
            assert (
                self.psi_s is not None and self.psi_e is not None
            ), "psi_s and psi_e must be defined for unclosed tracks"

        assert self.center_points.shape[1] == 2
        assert self.widths.shape == self.center_points.shape
        if self.min_curv:
            assert self.car_params.W is not None, "W must be defined when min_curv=True"

        # fit cubic splines to fit the given center points =========================
        (
            original_center_coeffs_x,
            original_center_coeffs_y,
            _,
            original_center_normvectors,
        ) = tph.calc_splines(
            path=self.center_points,
            closed=self.closed,
            psi_s=self.psi_s,  # psi_s is ignored if closed is True
            psi_e=self.psi_e,  # psi_e is ignored if closed is True
        )

        if not self.closed:
            original_center_normvectors = np.vstack(
                (
                    original_center_normvectors,
                    tph.calc_normal_vectors(np.array(self.psi_e)),
                )
            )

        spline_lengths = tph.calc_spline_lengths(
            coeffs_x=original_center_coeffs_x,
            coeffs_y=original_center_coeffs_y,
            no_interp_points=100,
        )

        # compute the reference points =============================================
        if self.min_curv:
            # compute intermediate points more tightly spaced but not too much to avoid
            # long computation times in curvature minimization ========================
            (
                new_center_points,
                new_center_original_spline_idx,
                new_center_original_t_values,
                _,
            ) = tph.interp_splines(
                coeffs_x=original_center_coeffs_x,
                coeffs_y=original_center_coeffs_y,
                spline_lengths=spline_lengths,
                closed=self.closed,
                stepsize_approx=1.0,
            )

            # minimum curvature optimization =======================================
            (_, _, new_center_M, new_center_normvectors) = tph.calc_splines(
                path=new_center_points,
                closed=self.closed,
                psi_s=self.psi_s,
                psi_e=self.psi_e,
            )

            if not self.closed:
                new_center_normvectors = np.vstack(
                    (
                        new_center_normvectors,
                        tph.calc_normal_vectors(np.array(self.psi_e)),
                    )
                )

            new_widths = tph.interp_track_widths(
                1.5 * np.ones_like(self.widths),
                new_center_original_spline_idx,
                new_center_original_t_values,
            )

            # set additional attributes if necessary
            if "center_points" in self.motion_planner_params.additional_attributes:
                self.additional_attributes["center_points"] = new_center_points
            if (
                "center_right_widths"
                in self.motion_planner_params.additional_attributes
            ):
                self.additional_attributes["center_right_widths"] = new_widths[:, 0]
            if "center_left_widths" in self.motion_planner_params.additional_attributes:
                self.additional_attributes["center_left_widths"] = new_widths[:, 1]
            if "boundaries" in self.motion_planner_params.additional_attributes:
                self.additional_attributes["boundaries"] = np.hstack(
                    (
                        new_center_points
                        + np.expand_dims(new_widths[:, 0], 1) * new_center_normvectors,
                        new_center_points
                        - np.expand_dims(new_widths[:, 1], 1) * new_center_normvectors,
                    )
                )

            alpha_min_curv, _ = tph.opt_min_curv(
                reftrack=np.hstack((new_center_points, new_widths)),
                normvectors=new_center_normvectors,
                A=new_center_M,
                kappa_bound=1.0,
                w_veh=self.car_params.W * 1.5,
                closed=self.closed,
                psi_s=self.psi_s,
                psi_e=self.psi_e,
                print_debug=False,
                method="quadprog",
            )

            # compute the new racing line =============================================
            (
                reference_points,
                _,
                reference_coeffs_x,
                reference_coeffs_y,
                _,
                reference_new_center_spline_idx,
                reference_new_center_t_values,
                reference_new_center_s_values,
                reference_spline_lengths,
                reference_el_lengths,
            ) = tph.create_raceline(
                refline=new_center_points,
                normvectors=new_center_normvectors,
                alpha=alpha_min_curv,
                stepsize_interp=0.1,
                closed=self.closed,
                psi_s=self.psi_s,
                psi_e=self.psi_e,
            )

        else:
            # compute the new racing line =============================================
            (
                reference_points,
                _,
                reference_coeffs_x,
                reference_coeffs_y,
                reference_normvectors,
                reference_new_center_spline_idx,
                reference_new_center_t_values,
                reference_new_center_s_values,
                reference_spline_lengths,
                reference_el_lengths,
            ) = tph.create_raceline(
                refline=self.center_points,
                normvectors=original_center_normvectors,
                alpha=np.zeros(self.center_points.shape[0]),
                stepsize_interp=0.1 if self.mission != Mission.AUTOCROSS else 0.5,
                closed=self.closed,
                psi_s=self.psi_s,
                psi_e=self.psi_e,
            )
            headings, _ = tph.calc_head_curv_an(
                coeffs_x=reference_coeffs_x,
                coeffs_y=reference_coeffs_y,
                ind_spls=reference_new_center_spline_idx,
                t_spls=reference_new_center_t_values,
                calc_curv=False,
            )
            new_widths = tph.interp_track_widths(
                self.widths,
                reference_new_center_spline_idx,
                reference_new_center_t_values,
            )
            # set additional attributes if necessary
            if "center_points" in self.motion_planner_params.additional_attributes:
                self.additional_attributes["center_points"] = reference_points
            if (
                "center_right_widths"
                in self.motion_planner_params.additional_attributes
            ):
                self.additional_attributes["center_right_widths"] = new_widths[:, 0]
            if "center_left_widths" in self.motion_planner_params.additional_attributes:
                self.additional_attributes["center_left_widths"] = new_widths[:, 1]
            if "boundaries" in self.motion_planner_params.additional_attributes:
                norm_vectors = tph.calc_normal_vectors(headings)
                self.additional_attributes["boundaries"] = np.hstack(
                    (
                        reference_points
                        + np.expand_dims(new_widths[:, 0], 1) * norm_vectors,
                        reference_points
                        - np.expand_dims(new_widths[:, 1], 1) * norm_vectors,
                    )
                )

        self.reference_points = reference_points
        self.reference_arc_lengths = reference_new_center_s_values

        if self.closed:
            self.total_length = (
                reference_new_center_s_values[-1] + reference_el_lengths[-1]
            )
        else:
            self.total_length = reference_new_center_s_values[-1]

        # compute reference heading and curvature =====================================
        self.reference_heading, reference_curvature = tph.calc_head_curv_an(
            coeffs_x=reference_coeffs_x,
            coeffs_y=reference_coeffs_y,
            ind_spls=reference_new_center_spline_idx,
            t_spls=reference_new_center_t_values,
        )
        ref_diffs = np.diff(self.reference_heading)
        ref_diffs[ref_diffs > 1.5 * np.pi] -= 2 * np.pi
        ref_diffs[ref_diffs < -1.5 * np.pi] += 2 * np.pi
        self.reference_heading = np.insert(
            self.reference_heading[0] + np.cumsum(ref_diffs),
            0,
            self.reference_heading[0],
        )

        if "reference_curvature" in self.motion_planner_params.additional_attributes:
            self.additional_attributes["reference_curvature"] = reference_curvature

        # compute reference velocities =====================================================
        np.seterr(divide="ignore")
        self.reference_velocity = np.minimum(
            np.sqrt(self.car_params.a_y_max / np.abs(reference_curvature)),
            self.car_params.v_x_max * np.ones_like(reference_curvature),
        )
        np.seterr(divide="warn")
        # TODO: use calc_t_profile
        if self.closed:
            reference_time = np.insert(
                np.cumsum(
                    reference_el_lengths
                    / _movmean(
                        np.append(self.reference_velocity, self.reference_velocity[0]),
                        2,
                    )
                ),
                0,
                0.0,
            )
        else:
            reference_time = np.insert(
                np.cumsum(reference_el_lengths / _movmean(self.reference_velocity, 2)),
                0,
                0.0,
            )

        self.total_time = reference_time[-1]
        if "reference_time" in self.motion_planner_params.additional_attributes:
            self.additional_attributes["reference_time"] = reference_time

        # create final interpolation objects =========================================
        self.X_ref_vs_arc_length = InterpolatedUnivariateSpline(
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            np.append(reference_points[:, 0], reference_points[0, 0])
            if self.closed
            else reference_points[:, 0],
            k=1,
            ext="const",
        )
        self.Y_ref_vs_arc_length = InterpolatedUnivariateSpline(
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            np.append(reference_points[:, 1], reference_points[0, 1])
            if self.closed
            else reference_points[:, 1],
            k=1,
            ext="const",
        )
        self.phi_ref_vs_arc_length = InterpolatedUnivariateSpline(
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            np.append(self.reference_heading, self.reference_heading[0])
            if self.closed
            else self.reference_heading,
            k=1,
            ext="const",
        )
        self.v_x_ref_vs_arc_length = InterpolatedUnivariateSpline(
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            np.append(self.reference_velocity, self.reference_velocity[0])
            if self.closed
            else self.reference_velocity,
            k=1,
            ext="const",
        )
        self.time_vs_arc_length = InterpolatedUnivariateSpline(
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            reference_time,
            k=1,
            ext="const",
        )
        self.arc_length_vs_time = InterpolatedUnivariateSpline(
            reference_time,
            np.append(reference_new_center_s_values, self.total_length)
            if self.closed
            else reference_new_center_s_values,
            k=1,
            ext="const",
        )
        # self.X_ref_vs_time = InterpolatedUnivariateSpline(
        #     reference_time,
        #     np.append(reference_points[:, 0], reference_points[0, 0])
        #     if self.closed
        #     else reference_points[:, 0],
        #     k=1,
        #     ext="const",
        # )
        # self.Y_ref_vs_time = InterpolatedUnivariateSpline(
        #     reference_time,
        #     np.append(reference_points[:, 1], reference_points[0, 1])
        #     if self.closed
        #     else reference_points[:, 1],
        #     k=1,
        #     ext="const",
        # )
        if (
            "right_width_vs_time" in self.motion_planner_params.additional_attributes
            and not self.min_curv
        ):
            self.additional_attributes[
                "right_width_vs_time"
            ] = InterpolatedUnivariateSpline(
                reference_time,
                np.append(new_widths[:, 0], new_widths[0, 0])
                if self.closed
                else new_widths[:, 0],
                k=1,
                ext="const",
            )
        if (
            "left_width_vs_time" in self.motion_planner_params.additional_attributes
            and not self.min_curv
        ):
            self.additional_attributes[
                "left_width_vs_time"
            ] = InterpolatedUnivariateSpline(
                reference_time,
                np.append(new_widths[:, 1], new_widths[0, 1])
                if self.closed
                else new_widths[:, 1],
                k=1,
                ext="const",
            )

        # create key_points ============================================================
        if self.mission == Mission.ACCELERATION:
            interest_points = [(90.0, np.array([0.0, 90]))]
        elif self.mission == Mission.SHORT_SKIDPAD:
            interest_points = [
                (15.0, np.array([0.0, 15.0])),
                (72.3, np.array([0.0, 15.0])),
                (129.6, np.array([0.0, 15.0])),
            ]
        elif self.mission == Mission.SKIDPAD:
            interest_points = [
                (15.0, np.array([0.0, 15.0])),
                (72.3, np.array([0.0, 15.0])),
                (129.6, np.array([0.0, 15.0])),
                (187.0, np.array([0.0, 15.0])),
                (244.3, np.array([0.0, 15.0])),
            ]
        elif self.mission == Mission.AUTOCROSS:
            interest_points = []
        elif self.mission == Mission.TRACKDRIVE:
            interest_points = []
        else:
            raise ValueError("Unknown mission type")
        self.key_points = []
        for s_guess, pos in interest_points:
            self.key_points.append(
                self.localize_point(pos=pos, guess=s_guess, tolerance=2.0)
            )
        if self.mission == Mission.TRACKDRIVE:
            self.key_points.append((self.total_length, self.reference_points[0]))

    def localize_point(
        self,
        pos: np.ndarray,
        guess: float,
        tolerance: Optional[float] = 10.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Localize a point on the reference trajectory by orthogonal projection.

        :param pos: point to localize, shape (2,)
        :type pos: np.ndarray
        :param guess: initial guess for the arc length
        :type guess: float
        :param tolerance: tolerance for the localization (how far the point can be from
        the initial guess), defaults to 20.0
        :type tolerance: Optional[float]
        :return: arc length and position of the point on the reference trajectory
        """
        assert pos is not None
        assert guess is not None
        if self.closed:
            s_return = tph.path_matching_global(
                path_cl=np.hstack(
                    (
                        np.reshape(
                            np.append(self.reference_arc_lengths, self.total_length),
                            (-1, 1),
                        ),
                        np.vstack((self.reference_points, self.reference_points[0, :])),
                    )
                ),
                ego_position=pos,
                s_expected=guess,
                s_range=tolerance,
            )[0]
        else:
            # get relevant part of the reference path
            s_min = max(0.0, guess - tolerance)
            s_max = min(self.total_length, guess + tolerance)

            # get indices of according points
            idx_start = (
                np.searchsorted(self.reference_arc_lengths, s_min, side="right") - 1
            )  # - 1 to include trajectory point before s_min
            idx_stop = (
                np.searchsorted(self.reference_arc_lengths, s_max, side="left") + 1
            )  # + 1 to include trajectory point after s_max when slicing
            s_return = tph.path_matching_local(
                path=np.hstack(
                    (
                        self.reference_arc_lengths[idx_start:idx_stop].reshape(-1, 1),
                        self.reference_points[idx_start:idx_stop, :],
                    )
                ),
                ego_position=pos,
                consider_as_closed=False,
            )[0]

        return s_return, np.array(
            [self.X_ref_vs_arc_length(s_return), self.Y_ref_vs_arc_length(s_return)]
        )

    def extract_horizon_arc_lengths(
        self,
        horizon_size: int,
        sampling_time: float,
        pos: np.ndarray,
        guess: float,
        tolerance: float = 10.0,
    ) -> Union[float, np.ndarray]:
        """
        Extract arc length localizations corresponding to a horizon

        :param horizon_size: number of points in the horizon (excluding the current
        point)
        :type horizon_size: int
        :param sampling_time: time between two points in the horizon
        :type sampling_time: float
        :param pos: current position of the vehicle
        :param guess: initial guess for the arc length
        :type guess: float
        :param tolerance: tolerance for the localization (how far the point can be from
        the initial guess), defaults to 20.0
        :type tolerance: Optional[float]
        :return: arc length of the first horizon point and positions of all the horizon points
        """
        # localize on the reference path
        arc_length_localization, _ = self.localize_point(
            pos, guess=guess, tolerance=tolerance
        )
        if horizon_size == 0:
            return arc_length_localization
        else:
            # get map passage time corresponding to the localization
            time_reference = self.time_vs_arc_length(arc_length_localization)
            # passage times vector for prediction horizon
            time_horizon = np.linspace(
                time_reference,
                time_reference + horizon_size * sampling_time,
                horizon_size + 1,
            )
            if self.closed:
                time_horizon = np.mod(time_horizon, self.total_time)

            # interpolate map to get reference arc lengths
            return self.arc_length_vs_time(time_horizon)

    # def extract_horizon(
    #     self,
    #     horizon_size: int,
    #     sampling_time: float,
    #     pos: np.ndarray,
    #     guess: float,
    #     tolerance: Optional[float] = 20.0,
    #     additional_stuff: list[str] = [],
    # ) -> tuple[float, np.ndarray]:
    #     """
    #     Extract a horizon of points on the reference trajectory at equidistant time steps.
    #
    #     :param horizon_size: number of points in the horizon
    #     :type horizon_size: int
    #     :param sampling_time: time between two points in the horizon
    #     :type sampling_time: float
    #     :param pos: current position of the vehicle
    #     :param guess: initial guess for the arc length
    #     :type guess: float
    #     :param tolerance: tolerance for the localization (how far the point can be from
    #     the initial guess), defaults to 20.0
    #     :type tolerance: Optional[float]
    #     :return: arc length of the first horizon point and positions of all the horizon points
    #     """
    #     # localize on the reference path
    #     arc_length_localization, pos_localization = self.localize_point(
    #         pos, guess=guess, tolerance=tolerance
    #     )
    #     if horizon_size == 1:
    #         return arc_length_localization, pos_localization
    #
    #     # get map passage time corresponding to the localization
    #     time_reference = self.time_vs_arc_length(arc_length_localization)
    #     # passage times vector for prediction horizon
    #     time_horizon = np.linspace(
    #         time_reference, time_reference + horizon_size * sampling_time, horizon_size
    #     )
    #     if self.closed:
    #         time_horizon = np.mod(time_horizon, self.total_time)
    #
    #     # interpolate map to get reference points
    #     arc_length_horizon = self.arc_length_vs_time(time_horizon)
    #     values = [
    #         self.X_ref_vs_arc_length(arc_length_horizon),
    #         self.Y_ref_vs_arc_length(arc_length_horizon),
    #     ]
    #     if additional_stuff:
    #         values.append(self.heading_vs_arc_length(arc_length_horizon))
    #         values.append(self.v_x_vs_arc_length(arc_length_horizon))
    #
    #     return arc_length_localization, np.array(values).T

    @property
    def stopping_arc_length(self) -> float:
        return self.key_points[-1][0]

    @property
    def center_points(self) -> np.ndarray:
        return self.motion_planner_params.center_points

    @property
    def mission(self) -> Mission:
        return self.motion_planner_params.mission

    @property
    def widths(self) -> np.ndarray:
        return self.motion_planner_params.widths

    @property
    def psi_s(self) -> float:
        return self.motion_planner_params.psi_s

    @property
    def psi_e(self) -> float:
        return self.motion_planner_params.psi_e

    @property
    def min_curv(self) -> bool:
        return self.mission == Mission.TRACKDRIVE

    @property
    def closed(self) -> bool:
        return self.mission == Mission.TRACKDRIVE


def _movmean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
