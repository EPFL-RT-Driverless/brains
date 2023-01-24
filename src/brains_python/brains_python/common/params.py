# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import warnings
from enum import Enum
from typing import Union, Tuple, get_origin, get_args

__all__ = ["Params"]


class Params:
    """
    Base class for all the parameters types needed (e.g. CarParams). You can see it as an evolution on typings.TypedDict
    that actually stores the values as class attributes instead of dict values, which lets you use PyCharm's autocomplete
    and type checking.
    Must always be instantiated with a dict of params given as keyword arguments to ensure that the required types
    are respected.

    When subclassing Params, you have to define your params but also implement the __init__ method as follows:
    >>> class MyParams(Params):
    >>>     a: int
    >>>     b: float
    >>>     c: str
    >>>     d: bool
    >>>     def __init__(self, **kwargs):
    >>>         current_params, remaining_params = Params._transform_dict(kwargs)
    >>>         for key, val in current_params.items():
    >>>             setattr(self, key, val)
    >>>         super().__init__(**remaining_params)
    """

    @classmethod
    def _transform_dict(cls, di: dict) -> Tuple[dict, dict]:
        """
        Transform a dict of params only containing values with primitives types (int, float,
         bool or str) into a dict with the right types to construct an instance of a
         subclass of Params. In particular, it instantiates the Enum values from strings.

        :param cls: subclass of Params specifying the types of the params
        :param di: dict of params to be transformed
        """
        res = {}
        for param_name, param_type in cls.__annotations__.items():
            if param_name in di:
                param_value = di.pop(param_name)

                if get_origin(param_type) != Union and issubclass(param_type, Enum):
                    if type(param_value) == str:
                        res[param_name] = param_type[param_value]
                    elif type(param_value) == param_type:
                        res[param_name] = param_value
                    else:
                        res[param_name] = None
                        warnings.warn(f"di[{param_name}] is not a string")
                else:
                    try:
                        if get_origin(param_type) == Union:
                            if isinstance(param_value, get_args(param_type)):
                                res[param_name] = param_value
                            else:
                                raise ValueError()
                        elif isinstance(param_value, param_type):
                            res[param_name] = param_value
                        else:
                            res[param_name] = param_type(param_value)
                    except ValueError as e:
                        warnings.warn(f"di[{param_name}] is not a valid {param_type}")
                        res[param_name] = None
            else:
                warnings.warn(f"di[{param_name}] does not exist")
                res[param_name] = None

        return res, di

    def __init__(self, **kwargs):
        """
        Initialize the instance with the given params.
        """
        pass
