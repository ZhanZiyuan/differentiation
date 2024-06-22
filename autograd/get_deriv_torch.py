#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Callable, Sequence

import numpy as np
import torch


def get_func_1d(func: Callable,
                x_value: np.ndarray) -> np.ndarray:
    """
    __doc__
    """
    x_value_tensor = torch.tensor(
        x_value,
        dtype=torch.float64,
        requires_grad=True
    )

    y_value = func(x_value_tensor)
    y_value.backward(create_graph=True)

    return x_value_tensor.grad.detach().numpy()


def get_func_2d(func: Callable,
                x_value: np.ndarray) -> np.ndarray:
    """
    __doc__
    """
    x_value_tensor = torch.tensor(
        x_value,
        dtype=torch.float64,
        requires_grad=True
    )

    y_value = func(x_value_tensor)
    y_value.backward(create_graph=True)

    grad_1d = x_value_tensor.grad.clone()

    grad_2d = []
    for i in range(len(grad_1d)):
        grad_1d[i].backward(retain_graph=True)
        grad_2d.append(x_value_tensor.grad.clone().detach().numpy())
        x_value_tensor.grad.zero_()
    return np.array(grad_2d)


if __name__ == "__main__":

    def func_01(x: Sequence) -> Sequence:
        """
        __doc__
        """
        return (
            x[0]**3 + x[1]**2 + 2*x[2] + 5*x[3]
        )

    def func_02(x: Sequence) -> Sequence:
        """
        __doc__
        """
        return (
            x[0]**3 + x[0]*x[1] + 2*x[1]**2 + 4*x[2] + 5*x[3]
        )

    x_input_01 = np.ones(4)

    func_01_1d = get_func_1d(
        func=func_01,
        x_value=x_input_01
    )

    func_02_2d = get_func_2d(
        func=func_02,
        x_value=x_input_01
    )

    print(func_01_1d)
    print(func_02_2d)
