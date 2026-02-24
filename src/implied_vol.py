#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:15:24 2026
Implied Volatility Solver
@author: poddar
"""

import numpy as np
from src.black_scholes import call_price, call_vega


def implied_vol_call(S0, K, r, T, market_price,
                     initial_guess=0.2,
                     tol=1e-8,
                     max_iter=100):

    sigma = initial_guess

    for _ in range(max_iter):

        price = call_price(S0, K, r, sigma, T)
        vega = call_vega(S0, K, r, sigma, T)

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

        # safety guard
        sigma = max(sigma, 1e-6)

    raise RuntimeError("Implied volatility did not converge")