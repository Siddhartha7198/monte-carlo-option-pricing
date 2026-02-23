# Monte Carlo Simulation for European Option Pricing

## Executive Summary

This project implements a Monte Carlo simulation framework for pricing European options under the risk-neutral measure. The objective is to evaluate the accuracy, convergence behavior, and computational characteristics of Monte Carlo pricing relative to the closed-form Black–Scholes solution.

The study includes:
- Mathematical derivation of the pricing framework
- Modular Python implementation
- Convergence and confidence interval analysis
- Benchmark comparison against Black–Scholes
- Sensitivity analysis with respect to volatility and time to maturity

This project is structured as a reproducible quantitative research report.

---

## 1. Motivation and Practical Relevance

Monte Carlo methods are widely used in quantitative finance when analytical solutions are unavailable or impractical.

While European options admit closed-form solutions via the Black–Scholes model, many real-world derivatives do not, including:

- Asian options
- Barrier options
- Lookback options
- Multi-asset derivatives

Monte Carlo simulation is especially valuable in:

- Structured products desks
- Risk management teams
- XVA calculations
- Model validation frameworks

Its flexibility makes it a core computational tool in financial engineering and quantitative consulting.

---

## 2. Model Overview

We assume the underlying asset follows a Geometric Brownian Motion (GBM) under the risk-neutral probability measure:

dS_t = r S_t dt + σ S_t dW_t

Under this assumption, the terminal stock price distribution is:

S_T = S_0 * exp((r - 0.5σ²)T + σ√T Z)

where:
- Z ~ N(0,1)
- r is the risk-free rate
- σ is volatility
- T is time to maturity

The arbitrage-free price of a European call option is:

C = e^{-rT} E_Q[max(S_T - K, 0)]

This expectation is approximated numerically via simulation.

---

## 3. Mathematical Framework

### 3.1 Risk-Neutral Valuation

Under the risk-neutral measure Q, all assets earn the risk-free rate in expectation. This ensures absence of arbitrage.

Option price:

V_0 = e^{-rT} E_Q[Payoff]

### 3.2 Monte Carlo Estimator

For N simulations:

Ĉ = e^{-rT} * (1/N) Σ max(S_T^{(i)} - K, 0)

By the Law of Large Numbers:

Ĉ → C as N → ∞

The standard error is:

SE = σ̂ / √N

Convergence rate: O(1/√N)

### References

- Hull, J. *Options, Futures, and Other Derivatives*
- Shreve, S. *Stochastic Calculus for Finance II*
- Black, F., & Scholes, M. (1973)

---

## 4. Data Requirements

For European option pricing under GBM, no historical dataset is required.

Inputs:
- Initial price (S₀)
- Strike (K)
- Risk-free rate (r)
- Volatility (σ)
- Time to maturity (T)

For benchmarking against market data, volatility may be estimated from historical returns.

---

## 5. Implementation Architecture

Core pricing logic is implemented in:

- `src/monte_carlo.py`
- `src/black_scholes.py`

Research analysis and visualization are performed in:

- `notebooks/analysis.ipynb`

Key features:

- Vectorized NumPy implementation
- Confidence interval computation
- Reproducible random seed control
- Benchmark comparison against Black–Scholes

---

## 6. Empirical Analysis

The following analyses are performed:

- Convergence study as N increases
- Confidence interval width behavior
- Sensitivity to volatility
- Sensitivity to maturity
- Runtime scaling analysis

Figures are stored in `reports/figures/`.

---

## 7. Benchmarking Against Black–Scholes

Since European options admit closed-form pricing, Monte Carlo results are compared against:

Black–Scholes formula:

C = S_0 N(d1) - K e^{-rT} N(d2)

Metrics evaluated:

- Absolute error
- Relative error
- Convergence trajectory
- Computational efficiency

---

## 8. Assumptions

- Geometric Brownian Motion
- Constant volatility
- Constant risk-free rate
- No transaction costs
- Continuous trading
- No arbitrage

Limitations of these assumptions are discussed in the conclusion.

---

## 9. Computational Considerations

- Complexity: O(N)
- Convergence rate: O(1/√N)
- Fully vectorized implementation
- Random seed for reproducibility
- Memory-efficient simulation

---

## 10. Results and Interpretation

Key observations:

- Monte Carlo converges slowly relative to closed-form pricing.
- Variance decreases proportionally to 1/N.
- High volatility increases dispersion of outcomes.
- Monte Carlo becomes computationally intensive for high precision requirements.

---

## 11. Conclusion

Monte Carlo simulation provides a flexible and generalizable framework for derivative pricing.

While inefficient for plain vanilla European options, it becomes indispensable for:

- Path-dependent derivatives
- High-dimensional problems
- Complex payoff structures

Future extensions:

- Variance reduction (Antithetic variates, Control variates)
- Quasi-Monte Carlo methods
- Stochastic volatility models
- Multi-asset simulation

---

## Reproducibility

Install dependencies:

pip install -r requirements.txt

Run analysis notebook:

jupyter notebook notebooks/analysis.ipynb
