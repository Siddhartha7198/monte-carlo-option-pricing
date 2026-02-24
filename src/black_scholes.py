#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 16:22:45 2026
Benchmarking monte carlo with Black–Scholes
@author: poddar
"""

import numpy as np
from scipy.stats import norm


def d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, sigma, T):
    return d1(S0, K, r, sigma, T) - sigma * np.sqrt(T)


def call_price(S0, K, r, sigma, T):
    """
    Black-Scholes European call price.
    """
    _d1 = d1(S0, K, r, sigma, T)
    _d2 = d2(S0, K, r, sigma, T)

    return S0 * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S0, K, r, sigma, T):
    """
    Black-Scholes European put price.
    """
    _d1 = d1(S0, K, r, sigma, T)
    _d2 = d2(S0, K, r, sigma, T)

    return K * np.exp(-r * T) * norm.cdf(-_d2) - S0 * norm.cdf(-_d1)

def call_delta(S0, K, r, sigma, T):
    _d1 = d1(S0, K, r, sigma, T)
    return norm.cdf(_d1)

def call_vega(S0, K, r, sigma, T):
    _d1 = d1(S0, K, r, sigma, T)
    return S0 * np.sqrt(T) * norm.pdf(_d1)