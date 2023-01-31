# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from enum import Enum

__all__ = ["Mission"]


class Mission(Enum):
    """Enum containing all the possible Formula Student events."""

    ACCELERATION = 0
    SKIDPAD = 1
    AUTOCROSS = 2
    TRACKDRIVE = 3
    SHORT_SKIDPAD = 4
