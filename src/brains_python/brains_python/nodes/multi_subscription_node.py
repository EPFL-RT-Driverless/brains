#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import functools
from typing import Any, Optional

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from abc import ABC, abstractmethod


class MultiSubscriptionNode(Node, ABC):
    publisher: Publisher
    bruh: dict[str, Subscription]
    subscriptions_msgs: dict[str, Optional[Any]]

    def __init__(
        self,
        node_name,
        subconfig: list[dict[str, Any]],
        pubconfig: dict[str, Any],
    ):
        super().__init__(node_name)
        self.publisher = self.create_publisher(
            pubconfig["msg_type"], pubconfig["topic"], pubconfig["queue_size"]
        )
        self.bruh = {}
        self.subscriptions_msgs = {}
        for v in subconfig:
            self.bruh[v["topic"]] = self.create_subscription(
                v["msg_type"],
                v["topic"],
                functools.partial(self.base_callback, topic_name=v["topic"]),
                v["queue_size"],
            )
            self.subscriptions_msgs[v["topic"]] = None

    @abstractmethod
    def processing(self, *args, **kwargs):
        pass

    def base_callback(self, msg, topic_name):
        self.subscriptions_msgs[topic_name] = msg
        if all([v is not None for v in self.subscriptions_msgs.values()]):
            self.processing(**self.subscriptions_msgs)
            for k in self.subscriptions_msgs.keys():
                self.subscriptions_msgs[k] = None
