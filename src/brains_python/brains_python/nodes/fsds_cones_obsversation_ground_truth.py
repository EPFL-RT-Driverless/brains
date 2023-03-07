from track_database import load_track
import rclpy
from rclpy.node import Node
import numpy as np
from brains_custom_interfaces.msg import CarState, ConesObservations
from brains_custom_interfaces.srv import MapNameFSDS


class ConesObservationGroundTruth(Node):
    def __init__(self):
        super().__init__("ConesObservationsGroundTruth")
        self.declare_parameter("cones_range_limits", [0, 12])
        self.declare_parameter("cones_bearing_limits", [-np.pi / 2, np.pi / 2])
        self.publisher = self.create_publisher(
            ConesObservations, "/brains/cones_observations_ground_truth", 10
        )
        self.subscriber = self.create_subscription(
            CarState, "/fsds/car_state", self.callback, 10
        )
        map_name_srv = self.create_client(MapNameFSDS, "/fsds/get_map_name")
        map_name = map_name_srv.call(MapNameFSDS.Request()).map_name
        self.track = load_track(map_name)
        # self._all_cones = np.vstack((track.right_cones, track.left_cones))

    def mask_cones(self, cones, x, y, phi):
        cones_bearing_limits, cones_range_limits = self.get_parameters(
            ["cones_bearing_limits", "cones_range_limits"]
        )
        cartesian = (cones - np.array([x, y])) @ np.array(
            [
                [np.cos(-phi), -np.sin(-phi)],
                [np.sin(-phi), np.cos(-phi)],
            ]
        ).T
        # transform to polar coordinates
        polar = np.hstack(
            (
                np.hypot(cartesian[:, 0], cartesian[:, 1])[:, np.newaxis],
                np.arctan2(cartesian[:, 1], cartesian[:, 0])[:, np.newaxis],
            )
        )
        # only keep the cones that are in front of the car and at less that 12m
        polar = polar[
            (cones_bearing_limits[0] <= polar[:, 1])
            & (polar[:, 1] <= cones_bearing_limits[1])
            & (cones_range_limits[0] <= polar[:, 0])
            & (polar[:, 0] <= cones_range_limits[1]),
            :,
        ]

        # add noise to the polar coordinates
        # polar += np.random.multivariate_normal(
        #     np.zeros(2),self.cones, size=polar.shape[0]
        # )

        return polar

    def callback(self, msg: CarState):
        tpr = [
            self.mask_cones(cones, msg.x, msg.y, msg.phi)
            for cones in [
                self.track.yellow_cones,
                self.track.blue_cones,
                self.track.big_orange_cones,
                self.track.small_orange_cones,
            ]
        ]

        polar = ConesObservations(
            rho=np.vstack([tpr[0][:, 0], tpr[1][:, 0], tpr[2][:, 0], tpr[3][:, 0]]),
            theta=np.vstack([tpr[0][:, 1], tpr[1][:, 1], tpr[2][:, 1], tpr[3][:, 1]]),
            color=np.vstack(
                [
                    np.zeros((tpr[0].shape[0], 1)),
                    np.ones((tpr[1].shape[0], 1)),
                    2 * np.ones((tpr[2].shape[0], 1)),
                    3 * np.ones((tpr[3].shape[0], 1)),
                ]
            ),
        )

        self.publisher.publish(polar)


def main(args=None):
    rclpy.init(args=args)
    node = ConesObservationGroundTruth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
