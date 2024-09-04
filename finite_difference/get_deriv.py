#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
https://docs.mantidproject.org/nightly/tutorials/extending_mantid_with_python/advanced_fit_behaviours/02_analytical_derivatives.html#

https://python.plainenglish.io/how-is-symbolic-differentiation-done-in-python-using-sympy-6484554f25b0
"""

from pprint import pprint
from typing import Callable, Union

import numpy as np
import sympy as sm


def get_partial_deriv_analytical(function: sm.Symbol,
                                 variables: Union[sm.Symbol, list],
                                 order: int = 1) -> Union[sm.Symbol, list]:
    """
    Get partial derivatives of the specified function.
    """
    if not isinstance(variables, list):
        variables = [variables]

    def nest_list(whole_list: list, depth: int) -> list:
        """
        Nest the list to the specified depth.
        """
        if depth == 1:
            return whole_list
        return [
            nest_list(whole_list[i:i+len(variables)**(depth-1)], depth-1)
            for i in range(0, len(whole_list), len(variables)**(depth-1))
        ]

    if order == 1:
        return [function.diff(var) for var in variables]

    partial_derivatives = [function]
    for _ in range(order):
        current_order_derivatives = []
        for deriv in partial_derivatives:
            current_order_derivatives.extend(
                [deriv.diff(var) for var in variables]
            )
        partial_derivatives = current_order_derivatives

    return nest_list(partial_derivatives, order)


def get_partial_deriv_numerical(function: Callable,
                                x: np.ndarray,
                                delta_x: float = 1e-5,
                                order: int = 1,
                                difference: str = "forward") -> np.ndarray:
    """
    __doc__
    """
    x = np.asarray(x)
    n = x.size

    if order == 1:
        grad = np.zeros(n)
        for i in range(n):
            x_forward = np.copy(x)
            x_backward = np.copy(x)
            x_forward[i] += delta_x
            x_backward[i] -= delta_x

            if difference == "central":
                grad[i] = (function(*x_forward) - function(*x_backward)) / (2 * delta_x)
            elif difference == "forward":
                grad[i] = (function(*x_forward) - function(*x)) / delta_x
            elif difference == "backward":
                grad[i] = (function(*x) - function(*x_backward)) / delta_x
        return grad

    elif order == 2:
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_forward_i = np.copy(x)
                x_backward_i = np.copy(x)
                x_forward_j = np.copy(x)
                x_backward_j = np.copy(x)

                x_forward_i[i] += delta_x
                x_backward_i[i] -= delta_x
                x_forward_j[j] += delta_x
                x_backward_j[j] -= delta_x

                if difference == "central":
                    hessian[i, j] = (function(*x_forward_i) - 2 * function(*x) + function(*x_backward_i)) / (delta_x ** 2)
                elif difference == "forward":
                    hessian[i, j] = (function(*x_forward_i) - function(*x)) / (delta_x ** 2)
                elif difference == "backward":
                    hessian[i, j] = (function(*x) - function(*x_backward_i)) / (delta_x ** 2)
        return hessian

    else:
        raise ValueError("Unsupported number of order.")


if __name__ == "__main__":

    x1, x2, x3 = sm.symbols("x1, x2, x3")
    formula_01 = x1**4 + x2**4 + x3**4

    pprint(get_partial_deriv_analytical(formula_01, [x1, x2, x3], order=1))
    pprint(get_partial_deriv_analytical(formula_01, [x1, x2, x3], order=2))
    pprint(get_partial_deriv_analytical(formula_01, [x1, x2, x3], order=3))

    deriv_2d_01 = get_partial_deriv_numerical(
        function=lambda a, b, c: a**3 + b**3 + c**3,
        x=np.array([1.0, 2.0, 3.0]),
        delta_x=1e-7,
        order=2,
        difference="central"
    )

    pprint(deriv_2d_01)
