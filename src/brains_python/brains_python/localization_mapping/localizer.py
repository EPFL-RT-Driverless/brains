from abc import ABC, abstractmethod

import numpy as np


class Localizer(ABC):
    """
    Abstract class for localizer
    """

    map: np.ndarray

    @abstractmethod
    def localize(
        self,
        cones: np.ndarray,
        motion_data: np.ndarray,
        sampling_time: float,
    ) -> np.array:
        """
        Localize the car on the track
        """
        pass


class SLAM:
    """
    Abstract class for SLAM
    """

    @abstractmethod
    def localize(
        self,
        cones: np.ndarray,
        motion_data: np.ndarray,
        sampling_time: float,
    ) -> np.array:
        """
        Localize the car on the track and update the map
        """
        pass

    @abstractmethod
    def get_map(self) -> np.ndarray:
        """
        Get the map
        """
        pass
