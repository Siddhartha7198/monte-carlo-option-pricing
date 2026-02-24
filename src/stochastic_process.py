#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:21:55 2026
GeometricBrownianMotion class 
@author: poddar
"""

import numpy as np


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion under the risk-neutral measure.

    dS_t = r S_t dt + sigma S_t dW_t

    Terminal solution:
    S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    Parameters
    ----------
    S0 : float
        Initial asset price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity (in years)
    """

    def __init__(self, S0: float, r: float, sigma: float, T: float):
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)

        self._validate_parameters()

    def _validate_parameters(self):
        if self.S0 <= 0:
            raise ValueError("Initial price S0 must be positive.")
        if self.sigma < 0:
            raise ValueError("Volatility sigma must be non-negative.")
        if self.T <= 0:
            raise ValueError("Time to maturity T must be positive.")

    def simulate_terminal(self, n_simulations: int, random_state: int = None):
        """
        Simulate terminal asset prices S_T.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Simulated terminal prices (size = n_simulations)
        """
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        rng = np.random.default_rng(random_state)

        Z = rng.standard_normal(n_simulations)

        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z

        ST = self.S0 * np.exp(drift + diffusion)

        return ST