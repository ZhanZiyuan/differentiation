#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
closures

https://bbs.huaweicloud.com/blogs/289295

https://cloud.baidu.com/article/3155346
"""

from typing import Callable, Union

import numpy as np
from scipy import interpolate


def get_derivative(func: Callable,
                   delta_x: float = 1.0e-10) -> Callable:
    """
    A factory function
    that returns the derivative function.
    """

    def derivative(x: Union[int, float]) -> Union[int, float]:
        """
        The derivative function.
        """
        return (func(x + delta_x) - func(x)) / delta_x

    return derivative


if __name__ == "__main__":

    time_sequence = np.linspace(192, 199, 8)
    angle_sequence = np.array(
        [
            7.085,
            10.497,
            14.019,
            17.683,
            21.487,
            25.403,
            29.402,
            33.467
        ]
    )

    angular_interpolated = interpolate.interp1d(
        x=time_sequence,
        y=angle_sequence,
        kind="cubic"
    )
    angular_velocity = get_derivative(angular_interpolated)

    print(angular_velocity(193.5))
    print(angular_velocity(195.8))

    x_arr = np.arange(1.0, 10.0, 0.1)
    x_square_arr = np.arange(1.0, 10.0, 0.1) ** 2

    y_interpolated = interpolate.interp1d(
        x=x_arr,
        y=x_square_arr,
        kind="cubic"
    )
    y_derivative = get_derivative(y_interpolated)

    print(y_derivative(2.5))
    print(y_derivative(6.0))
