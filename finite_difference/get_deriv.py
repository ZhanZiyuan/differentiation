#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
https://docs.mantidproject.org/nightly/tutorials/extending_mantid_with_python/advanced_fit_behaviours/02_analytical_derivatives.html#

https://python.plainenglish.io/how-is-symbolic-differentiation-done-in-python-using-sympy-6484554f25b0
"""

from pprint import pprint
from typing import Union

import sympy as sm


def get_partial_deriv(function: sm.Symbol,
                      variables: Union[list, tuple, sm.Symbol]) -> list:
    """
    Get first-order partial derivatives
    of the specified function.
    """
    if not isinstance(variables, (list, tuple)):
        variables = [variables]

    return [
        sm.diff(function, var)
        for var in variables
    ]


def split_into_sublist(whole_list: list, sub_list_length: int) -> list:
    """
    To split the whole list into sublists
    at a specific length.
    """
    split_list = []
    for i in range(0, len(whole_list), sub_list_length):
        sub_list = []
        for j in range(i, min(i + sub_list_length, len(whole_list))):
            sub_list.append(whole_list[j])
        split_list.append(sub_list)
    return split_list


def get_partial_deriv_2d(function: sm.Symbol,
                         variables: Union[list, tuple, sm.Symbol]) -> list:
    """
    Get second-order partial derivatives
    of the specified function.
    """
    if not isinstance(variables, (list, tuple)):
        variables = [variables]

    partial_derivatives_2d = []
    for var1 in variables:
        for var2 in variables:
            partial_derivatives_2d.append(sm.diff(function, var1, var2))
    return split_into_sublist(partial_derivatives_2d, len(variables))


if __name__ == "__main__":

    x1, x2, x3 = sm.symbols("x1, x2, x3")
    formula_01 = x1**4 + x2**4 + x3**4

    pprint(get_partial_deriv(formula_01, [x1, x2, x3]))
    pprint(get_partial_deriv_2d(formula_01, [x1, x2, x3]))
