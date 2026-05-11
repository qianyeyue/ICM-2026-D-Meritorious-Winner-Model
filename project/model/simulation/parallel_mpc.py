"""
Parallel MPC optimization utilities.

This module provides parallel Monte Carlo sampling for MPC action selection,
faster than sequential evaluation for larger runs.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def parallel_mc_evaluation(
    sequences: Sequence[Sequence],
    rollout_fn: Callable,
    n_mc: int,
    seed: int,
    max_workers: Optional[int] = None,
    **rollout_kwargs,
) -> Dict[int, List[float]]:
    """
    Evaluate action sequences in parallel using Monte Carlo sampling.

    Parameters
    ----------
    sequences : Sequence[Sequence]
        Action sequences to evaluate
    rollout_fn : Callable
        Function to evaluate a single rollout
        Signature: rollout_fn(actions, seed, **kwargs) -> float
    n_mc : int
        Number of Monte Carlo samples per sequence
    seed : int
        Base random seed
    max_workers : int, optional
        Maximum number of parallel workers (defaults to CPU count)
    **rollout_kwargs
        Additional arguments passed to rollout_fn

    Returns
    -------
    Dict[int, List[float]]
        Mapping from sequence index to list of scores

    Examples
    --------
    >>> def my_rollout(actions, seed, **kwargs):
    ...     rng = np.random.RandomState(seed)
    ...     return rng.randn()
    >>> sequences = [[(0.9, 200k, 0)], [(1.0, 500k, 0)]]
    >>> results = parallel_mc_evaluation(sequences, my_rollout, n_mc=10, seed=42)
    >>> len(results[0])
    10
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)

    # Create tasks: (seq_idx, mc_idx, seed, actions)
    tasks = []
    for seq_i, seq in enumerate(sequences):
        for mc_i in range(n_mc):
            task_seed = seed + 100000 * rollout_kwargs.get('t', 0) + 1000 * seq_i + mc_i
            tasks.append((seq_i, mc_i, task_seed, seq))

    # Execute in parallel
    results: Dict[int, List[float]] = {i: [] for i in range(len(sequences))}

    # Windows (and some sandboxed environments) can have issues with ProcessPool pipes
    # and also requires all callables to be picklable. Threads are more robust here.
    use_threads = os.name == "nt"
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    try:
        with Executor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_evaluate_single_rollout, rollout_fn, actions, task_seed, rollout_kwargs): (seq_i, mc_i)
                for seq_i, mc_i, task_seed, actions in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                seq_i, mc_i = future_to_task[future]
                try:
                    score = future.result()
                    results[seq_i].append(score)
                except Exception as e:
                    print(f"Warning: Rollout failed for sequence {seq_i}, MC {mc_i}: {e}")
                    results[seq_i].append(float('-inf'))
    except PermissionError:
        # Fallback: if process pool creation fails, retry with threads.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_evaluate_single_rollout, rollout_fn, actions, task_seed, rollout_kwargs): (seq_i, mc_i)
                for seq_i, mc_i, task_seed, actions in tasks
            }
            for future in as_completed(future_to_task):
                seq_i, mc_i = future_to_task[future]
                try:
                    score = future.result()
                    results[seq_i].append(score)
                except Exception as e:
                    print(f"Warning: Rollout failed for sequence {seq_i}, MC {mc_i}: {e}")
                    results[seq_i].append(float('-inf'))

    return results


def _evaluate_single_rollout(
    rollout_fn: Callable,
    actions: Sequence,
    seed: int,
    kwargs: Dict,
) -> float:
    """
    Evaluate a single rollout (helper for parallel execution).

    Parameters
    ----------
    rollout_fn : Callable
        Rollout evaluation function
    actions : Sequence
        Action sequence
    seed : int
        Random seed
    kwargs : Dict
        Additional arguments

    Returns
    -------
    float
        Rollout score
    """
    return rollout_fn(actions=actions, seed=seed, **kwargs)


def choose_action_with_early_stopping(
    sequences: Sequence[Sequence],
    rollout_fn: Callable,
    n_mc: int,
    seed: int,
    patience: int = 5,
    threshold: float = 0.01,
    max_workers: Optional[int] = None,
    **rollout_kwargs,
) -> Tuple[int, float]:
    """
    Choose best action with early stopping when no improvement is found.

    Parameters
    ----------
    sequences : Sequence[Sequence]
        Action sequences to evaluate
    rollout_fn : Callable
        Rollout evaluation function
    n_mc : int
        Number of MC samples per sequence
    seed : int
        Base random seed
    patience : int, default=5
        Number of sequences without improvement before stopping
    threshold : float, default=0.01
        Minimum improvement to reset patience counter
    max_workers : int, optional
        Maximum parallel workers
    **rollout_kwargs
        Additional rollout arguments

    Returns
    -------
    Tuple[int, float]
        (best_sequence_index, best_score)

    Examples
    --------
    >>> best_idx, best_score = choose_action_with_early_stopping(
    ...     sequences, rollout_fn, n_mc=10, seed=42, patience=5
    ... )
    """
    best_score = float('-inf')
    best_idx = 0
    no_improvement_count = 0

    # Evaluate sequences with early stopping
    for seq_i, seq in enumerate(sequences):
        # Evaluate this sequence with MC sampling
        scores = []
        for mc_i in range(n_mc):
            task_seed = seed + 100000 * rollout_kwargs.get('t', 0) + 1000 * seq_i + mc_i
            score = rollout_fn(actions=seq, seed=task_seed, **rollout_kwargs)
            scores.append(score)

        avg_score = float(np.mean(scores))

        # Check for improvement
        if avg_score > best_score + threshold:
            best_score = avg_score
            best_idx = seq_i
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping at sequence {seq_i + 1}/{len(sequences)} (no improvement for {patience} sequences)")
            break

    return best_idx, best_score


def adaptive_mc_sampling(
    sequence: Sequence,
    rollout_fn: Callable,
    seed: int,
    min_samples: int = 5,
    max_samples: int = 20,
    convergence_threshold: float = 0.05,
    **rollout_kwargs,
) -> Tuple[float, int]:
    """
    Adaptive Monte Carlo sampling that stops when score converges.

    Parameters
    ----------
    sequence : Sequence
        Action sequence to evaluate
    rollout_fn : Callable
        Rollout evaluation function
    seed : int
        Base random seed
    min_samples : int, default=5
        Minimum number of samples before checking convergence
    max_samples : int, default=20
        Maximum number of samples
    convergence_threshold : float, default=0.05
        Relative standard error threshold for convergence
    **rollout_kwargs
        Additional rollout arguments

    Returns
    -------
    Tuple[float, int]
        (mean_score, num_samples_used)

    Examples
    --------
    >>> score, n_samples = adaptive_mc_sampling(
    ...     sequence, rollout_fn, seed=42, min_samples=5, max_samples=20
    ... )
    """
    scores = []

    for mc_i in range(max_samples):
        task_seed = seed + mc_i
        score = rollout_fn(actions=sequence, seed=task_seed, **rollout_kwargs)
        scores.append(score)

        # Check convergence after min_samples
        if len(scores) >= min_samples:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            # Relative standard error
            if mean_score != 0:
                relative_se = std_score / (np.sqrt(len(scores)) * abs(mean_score))
                if relative_se < convergence_threshold:
                    break

    return float(np.mean(scores)), len(scores)


def batch_parallel_evaluation(
    sequences: Sequence[Sequence],
    rollout_fn: Callable,
    n_mc: int,
    seed: int,
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    **rollout_kwargs,
) -> List[float]:
    """
    Evaluate sequences in batches for better memory management.

    Useful when evaluating a large number of sequences (e.g., 729 sequences).

    Parameters
    ----------
    sequences : Sequence[Sequence]
        Action sequences to evaluate
    rollout_fn : Callable
        Rollout evaluation function
    n_mc : int
        Number of MC samples per sequence
    seed : int
        Base random seed
    batch_size : int, default=100
        Number of sequences per batch
    max_workers : int, optional
        Maximum parallel workers
    **rollout_kwargs
        Additional rollout arguments

    Returns
    -------
    List[float]
        Mean scores for each sequence

    Examples
    --------
    >>> scores = batch_parallel_evaluation(
    ...     sequences, rollout_fn, n_mc=10, seed=42, batch_size=100
    ... )
    """
    all_scores = []

    # Process in batches
    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]

        # Evaluate batch
        batch_results = parallel_mc_evaluation(
            batch_sequences,
            rollout_fn,
            n_mc,
            seed,
            max_workers=max_workers,
            **rollout_kwargs,
        )

        # Compute mean scores
        for i in range(len(batch_sequences)):
            if batch_results[i]:
                all_scores.append(float(np.mean(batch_results[i])))
            else:
                all_scores.append(float('-inf'))

    return all_scores


# Convenience function for backward compatibility
def parallel_choose_action(
    sequences: Sequence[Sequence],
    rollout_fn: Callable,
    n_mc: int,
    seed: int,
    use_early_stopping: bool = False,
    patience: int = 5,
    max_workers: Optional[int] = None,
    **rollout_kwargs,
) -> Tuple[int, float]:
    """
    Choose best action using parallel evaluation.

    Parameters
    ----------
    sequences : Sequence[Sequence]
        Action sequences to evaluate
    rollout_fn : Callable
        Rollout evaluation function
    n_mc : int
        Number of MC samples per sequence
    seed : int
        Base random seed
    use_early_stopping : bool, default=False
        Whether to use early stopping
    patience : int, default=5
        Patience for early stopping
    max_workers : int, optional
        Maximum parallel workers
    **rollout_kwargs
        Additional rollout arguments

    Returns
    -------
    Tuple[int, float]
        (best_sequence_index, best_score)
    """
    if use_early_stopping:
        return choose_action_with_early_stopping(
            sequences,
            rollout_fn,
            n_mc,
            seed,
            patience=patience,
            max_workers=max_workers,
            **rollout_kwargs,
        )
    else:
        # Full parallel evaluation
        results = parallel_mc_evaluation(
            sequences,
            rollout_fn,
            n_mc,
            seed,
            max_workers=max_workers,
            **rollout_kwargs,
        )

        # Find best sequence
        best_idx = 0
        best_score = float('-inf')

        for seq_i, scores in results.items():
            if scores:
                avg_score = float(np.mean(scores))
                if avg_score > best_score:
                    best_score = avg_score
                    best_idx = seq_i

        return best_idx, best_score

