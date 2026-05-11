"""
Markov Switching Volatility Model

Implements regime-switching model for substitute player performance.
When core players are injured, substitute players enter a high-volatility
regime with increased performance variance and tail risk.

Mathematical Model
------------------
Two-state Markov model:
    - State 0 (Low Volatility): Normal operations, core players healthy
    - State 1 (High Volatility): Core player injured, substitutes playing

Performance distribution:
    y_t | s_t=0 ~ N(渭_0, 蟽_0虏)  # Low volatility
    y_t | s_t=1 ~ N(渭_1, 蟽_1虏)  # High volatility, 蟽_1 > 蟽_0

Transition probabilities:
    P(s_t=1 | s_{t-1}=0) = p_01  # Injury occurs
    P(s_t=0 | s_{t-1}=1) = p_10  # Recovery

Key Insight
-----------
Substitute players have higher performance variance, leading to:
- Increased downside risk (bad games more likely)
- CVaR increases (tail risk)
- Win probability becomes more uncertain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MarkovRegimeModel:
    """
    Two-state Markov switching model for player performance.

    Attributes
    ----------
    mu_low : float
        Mean performance in low-volatility regime (healthy)
    sigma_low : float
        Std dev in low-volatility regime
    mu_high : float
        Mean performance in high-volatility regime (injured)
    sigma_high : float
        Std dev in high-volatility regime (higher than sigma_low)
    p_01 : float
        Transition probability: healthy ->injured
    p_10 : float
        Transition probability: injured ->healthy
    """
    mu_low: float = 0.75      # Win probability when healthy
    sigma_low: float = 0.10   # Low variance
    mu_high: float = 0.55     # Win probability when injured (from paper)
    sigma_high: float = 0.18  # High variance (~80% increase)
    p_01: float = 0.05        # 5% chance of injury per game
    p_10: float = 0.10        # 10% chance of recovery per game


def fit_regime_model(
    performance_data: pd.DataFrame,
    injury_indicator: pd.Series,
) -> MarkovRegimeModel:
    """
    Fit Markov regime model from historical data.

    Parameters
    ----------
    performance_data : pd.DataFrame
        Historical performance data with column: win_probability or performance_metric
    injury_indicator : pd.Series
        Binary indicator: 1 if core player injured, 0 otherwise

    Returns
    -------
    MarkovRegimeModel
        Fitted model parameters

    Examples
    --------
    >>> data = pd.DataFrame({'performance': [0.75, 0.73, 0.55, 0.58, 0.76]})
    >>> injury = pd.Series([0, 0, 1, 1, 0])
    >>> model = fit_regime_model(data, injury)
    >>> model.mu_low > model.mu_high
    True
    """
    # Separate data by regime
    perf_col = 'win_probability' if 'win_probability' in performance_data.columns else performance_data.columns[0]
    perf = performance_data[perf_col].values

    low_regime = perf[injury_indicator == 0]
    high_regime = perf[injury_indicator == 1]

    # Estimate parameters for each regime
    if len(low_regime) > 0:
        mu_low = float(np.mean(low_regime))
        sigma_low = float(np.std(low_regime))
    else:
        mu_low = 0.75
        sigma_low = 0.10

    if len(high_regime) > 0:
        mu_high = float(np.mean(high_regime))
        sigma_high = float(np.std(high_regime))
    else:
        mu_high = 0.55
        sigma_high = 0.18

    # Estimate transition probabilities
    transitions = np.diff(injury_indicator.values)
    n_01 = np.sum(transitions == 1)  # 0 ->1 transitions
    n_10 = np.sum(transitions == -1)  # 1 ->0 transitions
    n_00 = np.sum((injury_indicator.values[:-1] == 0) & (injury_indicator.values[1:] == 0))
    n_11 = np.sum((injury_indicator.values[:-1] == 1) & (injury_indicator.values[1:] == 1))

    p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.05
    p_10 = n_10 / (n_11 + n_10) if (n_11 + n_10) > 0 else 0.10

    return MarkovRegimeModel(
        mu_low=mu_low,
        sigma_low=sigma_low,
        mu_high=mu_high,
        sigma_high=sigma_high,
        p_01=float(p_01),
        p_10=float(p_10),
    )


def predict_regime_probabilities(
    model: MarkovRegimeModel,
    current_state: int,
    n_steps: int,
) -> np.ndarray:
    """
    Predict regime probabilities over n_steps.

    Parameters
    ----------
    model : MarkovRegimeModel
        Fitted model
    current_state : int
        Current state (0=healthy, 1=injured)
    n_steps : int
        Number of steps to predict

    Returns
    -------
    np.ndarray
        Array of shape (n_steps, 2) with probabilities [P(state=0), P(state=1)]

    Examples
    --------
    >>> model = MarkovRegimeModel()
    >>> probs = predict_regime_probabilities(model, current_state=0, n_steps=5)
    >>> probs.shape
    (5, 2)
    >>> probs[0, 0]  # Initially healthy
    1.0
    """
    # Transition matrix
    P = np.array([
        [1 - model.p_01, model.p_01],
        [model.p_10, 1 - model.p_10],
    ])

    # Initial state distribution
    state_probs = np.zeros((n_steps, 2))
    current_prob = np.array([1.0, 0.0]) if current_state == 0 else np.array([0.0, 1.0])

    for t in range(n_steps):
        state_probs[t] = current_prob
        current_prob = current_prob @ P  # Matrix multiplication

    return state_probs


def simulate_regime_path(
    model: MarkovRegimeModel,
    initial_state: int,
    n_games: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate regime path and performance over n_games.

    Parameters
    ----------
    model : MarkovRegimeModel
        Model parameters
    initial_state : int
        Initial state (0=healthy, 1=injured)
    n_games : int
        Number of games to simulate
    random_state : int, optional
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - states: Array of states (0 or 1) for each game
        - performance: Array of performance values for each game

    Examples
    --------
    >>> model = MarkovRegimeModel()
    >>> states, perf = simulate_regime_path(model, 0, 10, random_state=42)
    >>> len(states)
    10
    >>> len(perf)
    10
    """
    rng = np.random.RandomState(random_state)

    states = np.zeros(n_games, dtype=int)
    performance = np.zeros(n_games)

    states[0] = initial_state

    for t in range(n_games):
        current_state = states[t]

        # Sample performance from current regime
        if current_state == 0:  # Low volatility
            performance[t] = rng.normal(model.mu_low, model.sigma_low)
        else:  # High volatility
            performance[t] = rng.normal(model.mu_high, model.sigma_high)

        # Transition to next state (if not last game)
        if t < n_games - 1:
            if current_state == 0:
                # Healthy ->Injured with probability p_01
                states[t + 1] = 1 if rng.random() < model.p_01 else 0
            else:
                # Injured ->Healthy with probability p_10
                states[t + 1] = 0 if rng.random() < model.p_10 else 1

    return states, performance


def calculate_cvar(
    performance: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """
    Calculate Conditional Value-at-Risk (CVaR) at confidence level alpha.

    CVaR measures the expected loss in the worst (1-alpha)% of cases.

    Parameters
    ----------
    performance : np.ndarray
        Performance samples (e.g., win probabilities)
    alpha : float, default=0.95
        Confidence level (0.95 = worst 5%)

    Returns
    -------
    float
        CVaR (expected loss in tail)

    Examples
    --------
    >>> perf = np.array([0.75, 0.73, 0.55, 0.58, 0.76, 0.50, 0.48])
    >>> cvar = calculate_cvar(perf, alpha=0.95)
    >>> cvar < np.mean(perf)  # CVaR is lower than mean
    True
    """
    # Sort performance (ascending)
    sorted_perf = np.sort(performance)

    # Find VaR (Value-at-Risk) at alpha
    var_idx = int(np.floor((1 - alpha) * len(sorted_perf)))
    var_idx = max(0, min(var_idx, len(sorted_perf) - 1))

    # CVaR is the mean of values below VaR
    cvar = float(np.mean(sorted_perf[:var_idx + 1]))

    return cvar


def compare_regimes(
    model: MarkovRegimeModel,
    n_simulations: int = 10000,
    n_games: int = 10,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance distributions between regimes.

    Parameters
    ----------
    model : MarkovRegimeModel
        Model parameters
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    n_games : int, default=10
        Number of games per simulation
    random_state : int, optional
        Random seed

    Returns
    -------
    Dict[str, Dict[str, float]]
        Statistics for each regime:
        - mean: Mean performance
        - std: Standard deviation
        - cvar_95: CVaR at 95% confidence
        - prob_below_50: Probability of performance < 0.50

    Examples
    --------
    >>> model = MarkovRegimeModel()
    >>> stats = compare_regimes(model, n_simulations=1000, random_state=42)
    >>> stats['healthy']['mean'] > stats['injured']['mean']
    True
    >>> stats['injured']['std'] > stats['healthy']['std']
    True
    """
    rng = np.random.RandomState(random_state)

    # Simulate healthy regime
    healthy_perf = rng.normal(model.mu_low, model.sigma_low, size=(n_simulations, n_games))
    healthy_mean = healthy_perf.mean(axis=1)

    # Simulate injured regime
    injured_perf = rng.normal(model.mu_high, model.sigma_high, size=(n_simulations, n_games))
    injured_mean = injured_perf.mean(axis=1)

    # Calculate statistics
    stats = {
        'healthy': {
            'mean': float(np.mean(healthy_mean)),
            'std': float(np.std(healthy_mean)),
            'cvar_95': calculate_cvar(healthy_mean, alpha=0.95),
            'prob_below_50': float(np.mean(healthy_mean < 0.50)),
        },
        'injured': {
            'mean': float(np.mean(injured_mean)),
            'std': float(np.std(injured_mean)),
            'cvar_95': calculate_cvar(injured_mean, alpha=0.95),
            'prob_below_50': float(np.mean(injured_mean < 0.50)),
        },
    }

    return stats


def estimate_volatility_increase(
    baseline_performance: np.ndarray,
    injured_performance: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate volatility increase from baseline to injured state.

    Parameters
    ----------
    baseline_performance : np.ndarray
        Performance when healthy
    injured_performance : np.ndarray
        Performance when injured

    Returns
    -------
    Dict[str, float]
        Volatility metrics:
        - baseline_std: Baseline standard deviation
        - injured_std: Injured standard deviation
        - volatility_increase_pct: Percentage increase
        - variance_ratio: Ratio of variances

    Examples
    --------
    >>> baseline = np.array([0.75, 0.73, 0.76, 0.74, 0.77])
    >>> injured = np.array([0.55, 0.48, 0.62, 0.50, 0.58])
    >>> metrics = estimate_volatility_increase(baseline, injured)
    >>> metrics['volatility_increase_pct'] > 0
    True
    """
    baseline_std = float(np.std(baseline_performance))
    injured_std = float(np.std(injured_performance))

    volatility_increase_pct = ((injured_std - baseline_std) / baseline_std) * 100 if baseline_std > 0 else 0.0
    variance_ratio = (injured_std ** 2) / (baseline_std ** 2) if baseline_std > 0 else 1.0

    return {
        'baseline_std': baseline_std,
        'injured_std': injured_std,
        'volatility_increase_pct': float(volatility_increase_pct),
        'variance_ratio': float(variance_ratio),
    }


def create_default_model() -> MarkovRegimeModel:
    """
    Create default model with paper-specified parameters.

    Default scenario:
    - Win probability: 75% ->55% when injured
    - Network entropy decreases by 25%
    - Volatility increases

    Returns
    -------
    MarkovRegimeModel
        Default model
    """
    return MarkovRegimeModel(
        mu_low=0.75,      # Healthy win probability
        sigma_low=0.10,   # Low variance
        mu_high=0.55,     # Injured win probability (from paper)
        sigma_high=0.18,  # High variance (~80% increase)
        p_01=0.05,        # 5% injury risk per game
        p_10=0.10,        # 10% recovery rate per game
    )

