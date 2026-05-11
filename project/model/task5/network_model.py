"""
Network Structure Disruption Model

Implements graph-based analysis of team network structure and the impact
of key player injuries on network connectivity and entropy.

Mathematical Model
------------------
Network entropy:
    H_team = -Σ p_ij log(p_ij)

Network-adjusted strength:
    S_t^net = Σ h_{i,t} · μ_i - γ_net · ΔH_team(h_t)

Where:
    - p_ij: probability of pass from player i to player j
    - h_{i,t}: health status (1=healthy, 0=injured)
    - μ_i: player performance contribution
    - γ_net: network disruption coefficient (~0.25)
    - ΔH_team: change in network entropy

Key Insight
-----------
Core players (high flow centrality) act as network hubs. Their absence
causes disproportionate entropy loss, making the team's offense more
predictable and easier to defend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PlayerNetwork:
    """
    Represents team network structure.

    Attributes
    ----------
    players : List[str]
        Player names/IDs
    adjacency_matrix : np.ndarray
        Pass frequency matrix (i,j) = passes from i to j
    player_performance : Dict[str, float]
        Performance contribution μ_i for each player
    """
    players: List[str]
    adjacency_matrix: np.ndarray
    player_performance: Dict[str, float]

    def __post_init__(self):
        n = len(self.players)
        if self.adjacency_matrix.shape != (n, n):
            raise ValueError(f"Adjacency matrix shape {self.adjacency_matrix.shape} != ({n}, {n})")


def calculate_flow_centrality(network: PlayerNetwork) -> Dict[str, float]:
    """
    Calculate flow centrality for each player.

    Flow centrality measures how much a player acts as a hub in the
    passing network. High flow centrality = high network importance.

    Formula:
        FC_i = (in_degree_i + out_degree_i) / (2 * max_possible_degree)

    Parameters
    ----------
    network : PlayerNetwork
        Team network structure

    Returns
    -------
    Dict[str, float]
        Flow centrality for each player (normalized to [0, 1])

    Examples
    --------
    >>> players = ["A", "B", "C"]
    >>> adj = np.array([[0, 10, 5], [8, 0, 12], [3, 7, 0]])
    >>> perf = {"A": 0.8, "B": 1.2, "C": 0.9}
    >>> network = PlayerNetwork(players, adj, perf)
    >>> fc = calculate_flow_centrality(network)
    >>> fc["B"] > fc["A"]  # B is more central
    True
    """
    adj = network.adjacency_matrix
    n = len(network.players)

    # Calculate in-degree and out-degree (weighted by pass frequency)
    out_degree = adj.sum(axis=1)  # Sum of passes from player i
    in_degree = adj.sum(axis=0)   # Sum of passes to player i

    # Total flow through each player
    total_flow = out_degree + in_degree

    # Normalize by maximum possible flow
    max_flow = total_flow.max() if total_flow.max() > 0 else 1.0

    centrality = {}
    for i, player in enumerate(network.players):
        centrality[player] = float(total_flow[i] / max_flow)

    return centrality


def calculate_network_entropy(
    network: PlayerNetwork,
    health_status: Optional[Dict[str, bool]] = None,
) -> float:
    """
    Calculate network entropy H_team.

    Higher entropy = more unpredictable offense = harder to defend.
    Lower entropy = more predictable = easier to defend.

    Formula:
        H_team = -Σ p_ij log(p_ij)

    Where p_ij is the probability of a pass from i to j.

    Parameters
    ----------
    network : PlayerNetwork
        Team network structure
    health_status : Dict[str, bool], optional
        Health status for each player (True=healthy, False=injured)
        If None, assumes all players healthy

    Returns
    -------
    float
        Network entropy (higher = more diverse passing options)

    Examples
    --------
    >>> players = ["A", "B", "C"]
    >>> adj = np.array([[0, 10, 5], [8, 0, 12], [3, 7, 0]])
    >>> perf = {"A": 0.8, "B": 1.2, "C": 0.9}
    >>> network = PlayerNetwork(players, adj, perf)
    >>> H_healthy = calculate_network_entropy(network)
    >>> H_injured = calculate_network_entropy(network, {"A": True, "B": False, "C": True})
    >>> H_injured < H_healthy  # Entropy decreases when key player injured
    True
    """
    if health_status is None:
        health_status = {p: True for p in network.players}

    # Filter adjacency matrix to only healthy players
    adj = network.adjacency_matrix.copy()
    for i, player in enumerate(network.players):
        if not health_status.get(player, True):
            # Remove injured player from network
            adj[i, :] = 0
            adj[:, i] = 0

    # Convert to probability distribution
    total_passes = adj.sum()
    if total_passes == 0:
        return 0.0

    p_ij = adj / total_passes

    # Calculate entropy: H = -Σ p_ij log(p_ij)
    # Use log base 2 for bits of information
    entropy = 0.0
    for i in range(len(network.players)):
        for j in range(len(network.players)):
            if p_ij[i, j] > 0:
                entropy -= p_ij[i, j] * np.log2(p_ij[i, j])

    return float(entropy)


def simulate_network_disruption(
    network: PlayerNetwork,
    injured_player: str,
    gamma_net: float = 0.25,
) -> Tuple[float, float, float]:
    """
    Simulate impact of player injury on network structure.

    Returns the change in network entropy and the corresponding
    Elo strength penalty.

    Parameters
    ----------
    network : PlayerNetwork
        Team network structure
    injured_player : str
        Player who is injured
    gamma_net : float, default=0.25
        Network disruption coefficient

    Returns
    -------
    Tuple[float, float, float]
        - ΔH_team: Change in network entropy (negative)
        - Elo_penalty: Elo strength penalty from network disruption
        - centrality_loss: Flow centrality of injured player

    Examples
    --------
    >>> players = ["Wilson", "Young", "Plum"]
    >>> adj = np.array([[0, 20, 10], [15, 0, 25], [8, 12, 0]])
    >>> perf = {"Wilson": 1.5, "Young": 1.2, "Plum": 1.0}
    >>> network = PlayerNetwork(players, adj, perf)
    >>> dH, penalty, cent = simulate_network_disruption(network, "Wilson")
    >>> penalty < 0  # Negative impact on strength
    True
    """
    if injured_player not in network.players:
        raise ValueError(f"Player {injured_player} not in network")

    # Calculate baseline entropy (all healthy)
    H_baseline = calculate_network_entropy(network)

    # Calculate entropy with injured player
    health_status = {p: True for p in network.players}
    health_status[injured_player] = False
    H_injured = calculate_network_entropy(network, health_status)

    # Change in entropy (negative)
    delta_H = H_injured - H_baseline

    # Calculate flow centrality of injured player
    centrality = calculate_flow_centrality(network)
    centrality_loss = centrality[injured_player]

    # Elo penalty from network disruption
    # Higher centrality loss = larger penalty
    # Typical range: -10 to -50 Elo points
    elo_penalty = gamma_net * delta_H * 100  # Scale to Elo units

    return float(delta_H), float(elo_penalty), float(centrality_loss)


def create_network_from_data(
    player_stats: pd.DataFrame,
    pass_data: Optional[pd.DataFrame] = None,
) -> PlayerNetwork:
    """
    Create PlayerNetwork from player statistics and passing data.

    Parameters
    ----------
    player_stats : pd.DataFrame
        Player statistics with columns: player_name, ppg, apg, mpg
    pass_data : pd.DataFrame, optional
        Passing data with columns: from_player, to_player, count
        If None, estimates from assist data

    Returns
    -------
    PlayerNetwork
        Constructed network

    Notes
    -----
    If pass_data is not available, we estimate the adjacency matrix
    from assist data using a heuristic:
    - High-assist players have more outgoing edges
    - High-scoring players have more incoming edges
    """
    players = player_stats['player_name'].tolist()
    n = len(players)

    # Build performance dictionary (normalized by mean)
    perf_values = player_stats['ppg'].values
    mean_perf = perf_values.mean() if len(perf_values) > 0 else 1.0
    player_performance = {
        players[i]: float(perf_values[i] / mean_perf)
        for i in range(n)
    }

    # Build adjacency matrix
    if pass_data is not None:
        # Use actual passing data
        adj = np.zeros((n, n))
        player_to_idx = {p: i for i, p in enumerate(players)}

        for _, row in pass_data.iterrows():
            from_p = row['from_player']
            to_p = row['to_player']
            count = row['count']

            if from_p in player_to_idx and to_p in player_to_idx:
                i = player_to_idx[from_p]
                j = player_to_idx[to_p]
                adj[i, j] = count
    else:
        # Estimate from assist and scoring data
        adj = np.zeros((n, n))

        apg = player_stats['apg'].fillna(0).values
        ppg = player_stats['ppg'].fillna(0).values

        # Heuristic: player i passes to player j proportional to
        # assists_i * points_j
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Passes from i to j ~ assists_i * scoring_j
                    adj[i, j] = apg[i] * ppg[j]

        # Normalize to reasonable scale (total ~100 passes per game)
        total = adj.sum()
        if total > 0:
            adj = adj * (100.0 / total)

    return PlayerNetwork(
        players=players,
        adjacency_matrix=adj,
        player_performance=player_performance,
    )


def analyze_key_players(network: PlayerNetwork, top_k: int = 5) -> pd.DataFrame:
    """
    Identify key players by network centrality.

    Parameters
    ----------
    network : PlayerNetwork
        Team network
    top_k : int, default=5
        Number of top players to return

    Returns
    -------
    pd.DataFrame
        Top players with columns: player, flow_centrality, performance,
        disruption_impact (estimated Elo penalty if injured)
    """
    centrality = calculate_flow_centrality(network)

    results = []
    for player in network.players:
        # Simulate injury to estimate impact
        _, elo_penalty, cent = simulate_network_disruption(network, player)

        results.append({
            'player': player,
            'flow_centrality': centrality[player],
            'performance': network.player_performance[player],
            'disruption_impact': elo_penalty,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('flow_centrality', ascending=False)

    return df.head(top_k)
