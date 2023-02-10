# Copyright (c) 2023. Tudor Oancea, Adrien Remilieux, EPFL Racing Team Driverless
from typing import Union

import numpy as np
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import (
    Float32MultiArray,
    Float64MultiArray,
    Int8MultiArray,
    Int16MultiArray,
    Int32MultiArray,
    Int64MultiArray,
    UInt8MultiArray,
    UInt16MultiArray,
    UInt32MultiArray,
    UInt64MultiArray,
)

__all__ = ["MultiArray", "np_arr_from_ptr", "to_multiarray", "to_numpy"]

MultiArray = Union[
    (
        Float32MultiArray,
        Float64MultiArray,
        Int8MultiArray,
        Int16MultiArray,
        Int32MultiArray,
        Int64MultiArray,
        UInt8MultiArray,
        UInt16MultiArray,
        UInt32MultiArray,
        UInt64MultiArray,
    )
]


def np_arr_from_ptr(pointer, typestr, shape, copy=True, read_only_flag=False):
    """Generates numpy array from memory address
    https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html

    Parameters
    ----------
    pointer : int
        Memory address

    typestr : str
        A string providing the basic type of the homogenous array The
        basic string format consists of 3 parts: a character
        describing the byteorder of the data (<: little-endian, >:
        big-endian, |: not-relevant), a character code giving the
        basic type of the array, and an integer providing the number
        of bytes the type uses.

        The basic type character codes are:

        - t Bit field (following integer gives the number of bits in the bit field).
        - b Boolean (integer type where all values are only True or False)
        - i Integer
        - u Unsigned integer
        - f Floating point
        - c Complex floating point
        - m Timedelta
        - M Datetime
        - O Object (i.e. the memory contains a pointer to PyObject)
        - S String (fixed-length sequence of char)
        - U Unicode (fixed-length sequence of Py_UNICODE)
        - V Other (void * – each item is a fixed-size chunk of memory)

        See https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__

    shape : tuple
        Shape of array.

    copy : bool
        Copy array.  Default False

    read_only_flag : bool
        Read only array.  Default False.
    """
    buff = {"data": (pointer, read_only_flag), "typestr": typestr, "shape": shape}

    class numpy_holder:
        pass

    holder = numpy_holder()
    holder.__array_interface__ = buff
    return np.array(holder, copy=copy)


def to_multiarray(np_array: np.ndarray):
    arr_type = np_array.dtype.type

    if arr_type == np.uint8:
        multiarray = UInt8MultiArray()
    elif arr_type == np.int8:
        multiarray = Int8MultiArray()
    elif arr_type == np.uint16:
        multiarray = UInt16MultiArray()
    elif arr_type == np.int16:
        multiarray = Int16MultiArray()
    elif arr_type == np.uint32:
        multiarray = UInt32MultiArray()
    elif arr_type == np.int32:
        multiarray = Int32MultiArray()
    elif arr_type == np.int64:
        multiarray = Int64MultiArray()
    elif arr_type == np.uint64:
        multiarray = UInt64MultiArray()
    elif arr_type == np.float32:
        multiarray = Float32MultiArray()
    elif arr_type == np.float64:
        multiarray = Float64MultiArray()
    else:
        raise TypeError(f"Unexpected data type for Numpy ndarray : {str(arr_type)}")

    multiarray.layout.dim = [
        MultiArrayDimension(
            label="dim%d" % i, size=np_array.shape[i], stride=np_array.dtype.itemsize
        )
        for i in range(np_array.ndim)
    ]
    if np_array.dtype.byteorder == ">":
        multiarray.is_bigendian = True
    multiarray.data.frombytes(np_array.tobytes())

    return multiarray


def to_numpy(multiarray: MultiArray):
    dims = [x.size for x in multiarray.layout.dim]
    arr_type = type(multiarray)
    if arr_type == UInt8MultiArray:
        np_arr_type = np.uint8
    elif arr_type == Int8MultiArray:
        np_arr_type = np.int8
    elif arr_type == UInt16MultiArray:
        np_arr_type = np.uint16
    elif arr_type == Int16MultiArray:
        np_arr_type = np.int16
    elif arr_type == UInt32MultiArray:
        np_arr_type = np.uint32
    elif arr_type == UInt64MultiArray:
        np_arr_type = np.uint64
    elif arr_type == Float32MultiArray:
        np_arr_type = np.float32
    elif arr_type == Float64MultiArray:
        np_arr_type = np.float64
    else:
        raise TypeError(f"Unexpected data type for ROS multiarray : {str(arr_type)}")

    dt = np.dtype(np_arr_type)
    np_array = np.frombuffer(multiarray.data.tobytes(), dtype=dt).reshape(dims)

    return np_array
