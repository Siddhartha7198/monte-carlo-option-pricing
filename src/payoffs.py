#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:24:45 2026
Payoff class
@author: poddar
"""

import numpy as np


class Payoff:
    """
    Base class for option payoffs.
    """

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Payoff must implement __call__ method.")


class EuropeanCall(Payoff):
    """
    European Call Option Payoff
    max(ST - K, 0)
    """

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError("Strike must be positive.")
        self.strike = float(strike)

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        return np.maximum(ST - self.strike, 0.0)


class EuropeanPut(Payoff):
    """
    European Put Option Payoff
    max(K - ST, 0)
    """

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError("Strike must be positive.")
        self.strike = float(strike)

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - ST, 0.0)