#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Related articles:

https://ajz34.readthedocs.io/zh-cn/latest/ML_Notes/Autograd_Series/Autograd_TensorContract.html

https://ajz34.readthedocs.io/zh-cn/latest/ML_Notes/Autograd_Series/Autograd_Hess.html
"""

import time
from typing import Callable

import numpy as np
import torch


def get_partial_deriv_numerical(func: Callable,
                                x_value: np.ndarray,
                                order: int = 1) -> np.ndarray:
    """
    Get partial derivatives
    via the automatic differentiation.
    """
    x_value_tensor = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)

    if order == 1:
        y_value = func(x_value_tensor)
        grad = torch.autograd.grad(y_value, x_value_tensor, create_graph=True)[0]
        return grad.detach().numpy()

    elif order == 2:
        y_value = func(x_value_tensor)
        grad_1d = torch.autograd.grad(y_value, x_value_tensor, create_graph=True)[0]

        grad_2d = []
        for grad_i in grad_1d:
            x_value_tensor.grad = None
            grad_ij = torch.autograd.grad(grad_i, x_value_tensor, retain_graph=True)[0]
            grad_2d.append(grad_ij.detach().numpy())

        return np.array(grad_2d)

    else:
        raise ValueError("Unsupported number of order.")


if __name__ == "__main__":

    def func_01(x: np.ndarray) -> float:
        """
        __doc__
        """
        return (
            x[0]**2 + x[0]*x[1] + x[0]*x[2] + x[0]*x[3] + x[0]*x[4] + x[0]*x[5] + x[0]*x[6] + x[0]*x[7] + x[0]*x[8] + x[0]*x[9]
            + x[1]*x[0] + x[1]**2 + x[1]*x[2] + x[1]*x[3] + x[1]*x[4] + x[1]*x[5] + x[1]*x[6] + x[1]*x[7] + x[1]*x[8] + x[1]*x[9]
            + x[2]*x[0] + x[2]*x[1] + x[2]**2 + x[2]*x[3] + x[2]*x[4] + x[2]*x[5] + x[2]*x[6] + x[2]*x[7] + x[2]*x[8] + x[2]*x[9]
            + x[3]*x[0] + x[3]*x[1] + x[3]*x[2] + x[3]**2 + x[3]*x[4] + x[3]*x[5] + x[3]*x[6] + x[3]*x[7] + x[3]*x[8] + x[3]*x[9]
            + x[4]*x[0] + x[4]*x[1] + x[4]*x[2] + x[4]*x[3] + x[4]**2 + x[4]*x[5] + x[4]*x[6] + x[4]*x[7] + x[4]*x[8] + x[4]*x[9]
            + x[5]*x[0] + x[5]*x[1] + x[5]*x[2] + x[5]*x[3] + x[5]*x[4] + x[5]**2 + x[5]*x[6] + x[5]*x[7] + x[5]*x[8] + x[5]*x[9]
            + x[6]*x[0] + x[6]*x[1] + x[6]*x[2] + x[6]*x[3] + x[6]*x[4] + x[6]*x[5] + x[6]**2 + x[6]*x[7] + x[6]*x[8] + x[6]*x[9]
            + x[7]*x[0] + x[7]*x[1] + x[7]*x[2] + x[7]*x[3] + x[7]*x[4] + x[7]*x[5] + x[7]*x[6] + x[7]**2 + x[7]*x[8] + x[7]*x[9]
            + x[8]*x[0] + x[8]*x[1] + x[8]*x[2] + x[8]*x[3] + x[8]*x[4] + x[8]*x[5] + x[8]*x[6] + x[8]*x[7] + x[8]**2 + x[8]*x[9]
            + x[9]*x[0] + x[9]*x[1] + x[9]*x[2] + x[9]*x[3] + x[9]*x[4] + x[9]*x[5] + x[9]*x[6] + x[9]*x[7] + x[9]*x[8] + x[9]**2
        )

    x_input_01 = np.ones(10)

    start_time_01 = time.time()
    func_01_1d = get_partial_deriv_numerical(
        func=func_01,
        x_value=x_input_01,
        order=1
    )
    end_time_01 = time.time()

    start_time_02 = time.time()
    func_01_2d = get_partial_deriv_numerical(
        func=func_01,
        x_value=x_input_01,
        order=2
    )
    end_time_02 = time.time()

    print(
        f"When `x_input_01` is: {x_input_01}, \n"
        f"`func_01` is: {func_01(x_input_01)}. \n"
    )

    print(func_01_1d)
    print(
        f"Time consumption: {end_time_01 - start_time_01 :.4f}s\n"
    )
    print(func_01_2d)
    print(
        f"Time consumption: {end_time_02 - start_time_02 :.4f}s\n"
    )
