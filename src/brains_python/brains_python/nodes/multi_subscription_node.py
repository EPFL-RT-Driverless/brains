#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import functools
from typing import Any, Optional

from rclpy.node import Node
from rclpy.subscription import Subscription
from abc import ABC, abstractmethod
from enum import Enum


class MultiSubscriptionPolicy(Enum):
    ALL = 1
    ANY = 2


class MultiSubscriptionMixin(Node, ABC):
    named_subscriptions: dict[str, Subscription]
    subscriptions_msgs: dict[str, Optional[Any]]
    policy: MultiSubscriptionPolicy

    def __init__(
        self,
        node_name,
        subconfig: list[dict[str, Any]],
        policy: MultiSubscriptionPolicy = MultiSubscriptionPolicy.ALL,
    ):
        super().__init__(node_name)
        self.policy = policy
        self.named_subscriptions = {}
        self.subscriptions_msgs = {}
        for v in subconfig:
            self.named_subscriptions[v["topic"]] = self.create_subscription(
                v["msg_type"],
                v["topic"],
                functools.partial(
                    self.base_subscription_callback, topic_name=v["topic"]
                ),
                v["queue_size"],
            )
            self.subscriptions_msgs[v["topic"]] = None

    @abstractmethod
    def processing(self, **kwargs):
        pass

    def base_subscription_callback(self, msg, topic_name):
        self.subscriptions_msgs[topic_name] = msg
        if self.policy == MultiSubscriptionPolicy.ALL:
            if all([v is not None for v in self.subscriptions_msgs.values()]):
                self.processing(**self.subscriptions_msgs)
                for k in self.subscriptions_msgs.keys():
                    self.subscriptions_msgs[k] = None
        elif self.policy == MultiSubscriptionPolicy.ANY:
            self.processing(topic_name=msg)
            self.subscriptions_msgs[topic_name] = None
