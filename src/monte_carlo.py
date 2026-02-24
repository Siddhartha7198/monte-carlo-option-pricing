#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:17:30 2026
Running monte carlo simulations
@author: poddar
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

class MonteCarloPricer:
    """
    Monte Carlo pricing engine.

    Parameters
    ----------
    process : stochastic process object
        Must implement simulate_terminal()
    payoff : payoff object
        Must implement __call__(ST)
    """

    def __init__(self, process, payoff):
        self.process = process
        self.payoff = payoff

    def price(self, n_simulations: int, confidence_level: float = 0.95, random_state: int = None, 
              antithetic: bool = False, control_variate: bool = False):
        """
        Estimate option price using Monte Carlo simulation.

        Parameters
        ----------
        n_simulations : int
            Number of simulations
        confidence_level : float
            Confidence level for interval (default 95%)
        random_state : int, optional
            Random seed

        Returns
        -------
        dict
            Pricing results
        """

        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        # Simulate terminal prices
        ### ST = self.process.simulate_terminal(n_simulations, random_state)
        
        # Generate shocks
        rng = np.random.default_rng(random_state)
        
        if antithetic:
            half_n = n_simulations // 2
            Z = rng.standard_normal(half_n)
        
            drift = (self.process.r - 0.5 * self.process.sigma**2) * self.process.T
            diffusion = self.process.sigma * np.sqrt(self.process.T)
        
            ST_plus = self.process.S0 * np.exp(drift + diffusion * Z)
            ST_minus = self.process.S0 * np.exp(drift - diffusion * Z)
        
            payoff_plus = self.payoff(ST_plus)
            payoff_minus = self.payoff(ST_minus)
        
            discounted = np.exp(-self.process.r * self.process.T)
        
            paired_avg = discounted * 0.5 * (payoff_plus + payoff_minus)
        
            price_estimate = np.mean(paired_avg)
        
            sample_std = np.std(paired_avg, ddof=1)
            std_error = sample_std / np.sqrt(half_n)
            
            # Confidence interval
            z = norm.ppf(0.5 + confidence_level / 2.0)
            ci_lower = price_estimate - z * std_error
            ci_upper = price_estimate + z * std_error

            return {
                "price": price_estimate,
                "std_error": std_error,
                "confidence_interval": (ci_lower, ci_upper),
                "n_simulations": n_simulations
            }
        
        else:
            Z = rng.standard_normal(n_simulations)
        
            # Simulate terminal prices manually
            drift = (self.process.r - 0.5 * self.process.sigma**2) * self.process.T
            diffusion = self.process.sigma * np.sqrt(self.process.T) * Z
            
            ST = self.process.S0 * np.exp(drift + diffusion)
    
            # Compute payoffs
            payoffs = self.payoff(ST)
    
            # Discount factor
            discount_factor = np.exp(-self.process.r * self.process.T)
    
            discounted_payoffs = discount_factor * payoffs
            
            if control_variate:

                # Control variable: discounted stock price
                discounted_ST = np.exp(-self.process.r * self.process.T) * ST
            
                # Known expectation
                expected_Y = self.process.S0
            
                # Estimate beta
                cov = np.cov(discounted_payoffs, discounted_ST, ddof=1)
                beta = cov[0, 1] / cov[1, 1]
                
                # Apply control variate adjustment
                adjusted = discounted_payoffs - beta * (discounted_ST - expected_Y)
            
                discounted_payoffs = adjusted
    
            # Estimate price
            price_estimate = np.mean(discounted_payoffs)
    
            # Estimate standard error
            sample_std = np.std(discounted_payoffs, ddof=1)
            std_error = sample_std / np.sqrt(len(discounted_payoffs))
    
            # Confidence interval
            z = norm.ppf(0.5 + confidence_level / 2.0)
            ci_lower = price_estimate - z * std_error
            ci_upper = price_estimate + z * std_error
    
            return {
                "price": price_estimate,
                "std_error": std_error,
                "confidence_interval": (ci_lower, ci_upper),
                "n_simulations": n_simulations
            }
        
# %%
        

    def delta_pathwise(self, n_simulations: int,
               random_state: int = None):
        """
        Monte Carlo Delta via Pathwise Derivative method.
        """
    
        rng = np.random.default_rng(random_state)
    
        Z = rng.standard_normal(n_simulations)
    
        drift = (self.process.r - 0.5 * self.process.sigma**2) * self.process.T
        diffusion = self.process.sigma * np.sqrt(self.process.T)
    
        ST = self.process.S0 * np.exp(drift + diffusion * Z)
    
        discount_factor = np.exp(-self.process.r * self.process.T)
    
        indicator = (ST > self.payoff.strike).astype(float)
    
        delta_estimator = discount_factor * indicator * (ST / self.process.S0)
    
        delta = np.mean(delta_estimator)
    
        std_error = np.std(delta_estimator, ddof=1) / np.sqrt(n_simulations)
    
        return {
            "delta": delta,
            "std_error": std_error
        }
    
    
# %%
    
    def vega_pathwise(self, n_simulations: int,
                      random_state: int = None):
        """
        Monte Carlo Vega via Pathwise Derivative method.
        """
    
        rng = np.random.default_rng(random_state)
    
        Z = rng.standard_normal(n_simulations)
    
        S0 = self.process.S0
        r = self.process.r
        sigma = self.process.sigma
        T = self.process.T
    
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
    
        ST = S0 * np.exp(drift + diffusion * Z)
    
        discount_factor = np.exp(-r * T)
    
        indicator = (ST > self.payoff.strike).astype(float)
    
        dST_dsigma = ST * (-sigma * T + np.sqrt(T) * Z)
    
        vega_estimator = discount_factor * indicator * dST_dsigma
    
        vega = np.mean(vega_estimator)
    
        std_error = np.std(vega_estimator, ddof=1) / np.sqrt(n_simulations)
    
        return {
            "vega": vega,
            "std_error": std_error
        }
    
# %%
    
    def price_sobol(self, n_simulations: int):
        """
        Monte Carlo pricing using Sobol quasi-random sequences.
        """
    
        sampler = qmc.Sobol(d=1, scramble=True)
        U = sampler.random(n_simulations)
    
        # Transform uniform to standard normal
        Z = norm.ppf(U).flatten()
    
        S0 = self.process.S0
        r = self.process.r
        sigma = self.process.sigma
        T = self.process.T
    
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
    
        ST = S0 * np.exp(drift + diffusion * Z)
    
        discount_factor = np.exp(-r * T)
    
        payoffs = self.payoff(ST)
        discounted_payoffs = discount_factor * payoffs
    
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_simulations)
    
        return {
            "price": price,
            "std_error": std_error
        }