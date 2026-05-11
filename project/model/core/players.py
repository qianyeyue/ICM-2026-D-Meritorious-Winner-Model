"""
Player Commercial Value (PCV) estimation.

Goal
----
Use only:
- project/data/wnba_salaries.csv (salary + box-score style stats)
- project/data/players_fans.csv (social media fans proxy)

to estimate each player's single-player value PCV_i via a small ML regression.

Approach (pragmatic)
--------------------
We treat salary as the observable proxy target and learn:

    log(salary) ~ f(performance features, fans)

Then define:
    PCV_i = predicted_salary_i / mean(predicted_salary)

So the league-average PCV is ~1.0 (dimensionless index).

Outputs
-------
This module can be imported, or run as a script to write:
    project/data/processed/players_pcv.csv
"""

from __future__ import annotations

import warnings

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

def _silence_sklearn_warnings() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^sklearn\.")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"^sklearn\.")


_silence_sklearn_warnings()

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "project" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_OUTPUT_DIR = PROCESSED_DIR / "players"
MODEL_DIR = ROOT_DIR / "project" / "model" / "core"
WEIGHTS_FILE = MODEL_DIR / "court_impact_weights.json"
BRAND_WEIGHTS_FILE = MODEL_DIR / "brand_impact_weights.json"


def _resolve_path(p: str | Path, *, fallback_dir: Optional[Path] = None) -> Path:
    p = Path(p)
    if p.is_file():
        return p
    if (ROOT_DIR / p).is_file():
        return ROOT_DIR / p
    if fallback_dir is not None and (fallback_dir / p.name).is_file():
        return fallback_dir / p.name
    return p


_MONEY_RE = re.compile(r"[^\d.]")


def parse_money(value: object) -> float:
    """
    Parse strings like '$223,000' into float 223000.0.
    Returns NaN if value is missing/unparseable.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    s = str(value)
    s = s.replace("\u00a0", " ").strip()  # nbsp
    if not s:
        return float("nan")

    s = _MONEY_RE.sub("", s)
    if not s:
        return float("nan")

    try:
        return float(s)
    except Exception:
        return float("nan")


def load_salaries(
    csv_path: str | Path = None,
    *,
    first_block_only: bool = True,
) -> pd.DataFrame:
    # Default path: try new location first, then legacy
    if csv_path is None:
        csv_path = RAW_DIR / "wnba_salaries.csv"
        if not csv_path.is_file():
            csv_path = DATA_DIR / "wnba_salaries.csv"
    p = _resolve_path(csv_path, fallback_dir=RAW_DIR)
    df = pd.read_csv(p)

    required = {"Player", "2025 Salary", "G", "MIN", "PTS", "TRB", "AST"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("wnba_salaries.csv missing columns: %s" % sorted(missing))

    out = df.copy()

    # The raw file may contain multiple year blocks separated by repeated header rows
    # where Player == "Player". By default we keep only the first block (typically latest year).
    if first_block_only:
        marker_idx = out.index[out["Player"].astype(str).str.strip().str.lower() == "player"]
        if len(marker_idx) > 0:
            out = out.loc[: int(marker_idx[0]) - 1].copy()

    out["player_name"] = out["Player"].astype(str).str.strip()
    out = out[out["player_name"].notna() & (out["player_name"] != "")].copy()
    out = out[out["player_name"].str.lower() != "player"].copy()

    out["salary"] = out["2025 Salary"].map(parse_money)

    # Convert numeric fields
    numeric_cols = [
        "G",
        "MIN",
        "PTS",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG%",
        "3P%",
        "FT%",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Per-game features (robust to G=0)
    g = out["G"].replace(0, np.nan)
    out["ppg"] = out["PTS"] / g
    out["mpg"] = out["MIN"] / g
    out["apg"] = out["AST"] / g
    out["rpg"] = out["TRB"] / g

    if "STL" in out.columns:
        out["spg"] = out["STL"] / g
    if "BLK" in out.columns:
        out["bpg"] = out["BLK"] / g
    if "TOV" in out.columns:
        out["tovpg"] = out["TOV"] / g

    # Drop obvious broken rows (e.g. line-wrapped continuations) and keep one row per player.
    out = out[out["salary"].notna() & (out["salary"] > 0)].copy()
    out = out[out["G"].notna() & (out["G"] > 0)].copy()

    sort_cols = [c for c in ["salary", "MIN", "G"] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    out = out.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)

    return out


def load_players_fans(csv_path: str | Path = None) -> pd.DataFrame:
    # Default path: try new location first, then legacy
    if csv_path is None:
        csv_path = RAW_DIR / "players_fans.csv"
        if not csv_path.is_file():
            csv_path = DATA_DIR / "players_fans.csv"
    p = _resolve_path(csv_path, fallback_dir=RAW_DIR)
    df = pd.read_csv(p)

    # Some exports are actually TSV but named *.csv (seen in this repo).
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(p, sep="\t")

    # In your file: ID, number(w)
    if "ID" not in df.columns:
        # Sometimes BOM / whitespace sneaks into header
        id_like = None
        for c in df.columns:
            if str(c).strip().lstrip("\ufeff") == "ID":
                id_like = c
                break
        if id_like is None:
            raise ValueError("players_fans.csv missing column: ID")
        df = df.rename(columns={id_like: "ID"})

    fans_col = None
    for c in df.columns:
        if c.lower().startswith("number"):
            fans_col = c
            break
    if fans_col is None:
        raise ValueError("players_fans.csv missing fans column (expected something like 'number(w)')")

    out = df.copy()
    out["player_name"] = out["ID"].astype(str).str.strip()
    out["fans_w"] = pd.to_numeric(out[fans_col], errors="coerce")  # 'w' as in 10k
    return out[["player_name", "fans_w"]]


def load_players_age(csv_path: str | Path = None) -> pd.DataFrame:
    """
    Load player age data from years.csv.

    Note: Injury risk is calculated later in build_pcv_dataset() after merging
    with performance data (mpg, games) to incorporate workload/fatigue factors.
    """
    # Default path: try new location first, then legacy
    if csv_path is None:
        csv_path = RAW_DIR / "years.csv"
        if not csv_path.is_file():
            csv_path = DATA_DIR / "years.csv"
    p = _resolve_path(csv_path, fallback_dir=RAW_DIR)

    # Try reading with tab separator first
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception:
        df = pd.read_csv(p)

    if "Player" not in df.columns or "Age" not in df.columns:
        raise ValueError("years.csv missing required columns: Player, Age")

    out = df.copy()

    # Clean player names - remove extra whitespace and standardize
    out["player_name"] = out["Player"].astype(str).str.strip()

    # Remove any HTML tags or special formatting
    out["player_name"] = out["player_name"].str.replace(r'<[^>]+>', '', regex=True)

    # Parse age string (format: "XX y, XX d" or "XX y")
    def parse_age(age_str):
        if pd.isna(age_str):
            return float("nan")
        age_str = str(age_str)
        # Remove HTML tags
        age_str = re.sub(r'<[^>]+>', '', age_str)
        # Extract years from format like "25 y, 120 d" or "25 y"
        match = re.search(r'(\d+)\s*y', age_str)
        if match:
            return float(match.group(1))
        return float("nan")

    out["age"] = out["Age"].apply(parse_age)

    return out[["player_name", "age"]]


def build_pcv_dataset(
    salaries: pd.DataFrame,
    fans: pd.DataFrame,
    age_data: pd.DataFrame = None,
) -> pd.DataFrame:
    df = salaries.merge(fans, on="player_name", how="left")

    # Merge age data if provided
    if age_data is not None:
        df = df.merge(age_data, on="player_name", how="left")

    # Missing fans -> 0 (or could use median; 0 keeps "unknown" neutral after scaling)
    if "fans_w" in df.columns:
        df["fans_w"] = df["fans_w"].fillna(0.0)
    else:
        df["fans_w"] = 0.0

    # Calculate integrated injury risk based on age + workload + cumulative fatigue
    # This is done after merging so we have access to age, mpg, and G (games)
    def calculate_injury_risk(row):
        age = row.get("age")
        mpg = row.get("mpg")
        games = row.get("G")

        if pd.isna(age):
            return float("nan")

        # Component 1: Age risk (θ3*Age_i in the formula)
        # Base risk increases with age, especially after 30
        if age < 23:
            age_risk = 0.15 + (age - 20) * 0.02  # Young: 0.15-0.21
        elif age < 28:
            age_risk = 0.10 + (age - 23) * 0.01  # Prime: 0.10-0.15
        elif age < 32:
            age_risk = 0.15 + (age - 28) * 0.04  # Late prime: 0.15-0.31
        else:
            age_risk = 0.30 + (age - 32) * 0.05  # Veteran: 0.30+

        # Component 2: Workload/fatigue risk (θ1*Minutes_{i,t} in the formula)
        # Higher minutes per game = higher acute fatigue
        workload_risk = 0.0
        if mpg is not None and not pd.isna(mpg):
            # Normalize: 30+ min/game is high workload
            if mpg > 30:
                workload_risk = 0.15 + (mpg - 30) * 0.01
            elif mpg > 20:
                workload_risk = 0.05 + (mpg - 20) * 0.01
            else:
                workload_risk = max(0.0, mpg * 0.0025)

        # Component 3: Cumulative fatigue risk (related to total games played)
        # More games = more cumulative wear and tear
        cumulative_risk = 0.0
        if games is not None and not pd.isna(games):
            # Normalize: 30+ games is high cumulative load
            if games > 30:
                cumulative_risk = 0.10 + (games - 30) * 0.005
            elif games > 20:
                cumulative_risk = 0.05 + (games - 20) * 0.005
            else:
                cumulative_risk = max(0.0, games * 0.0025)

        # Integrated injury risk (weighted combination)
        # Weights: 50% age, 30% workload, 20% cumulative
        total_risk = 0.50 * age_risk + 0.30 * workload_risk + 0.20 * cumulative_risk

        # Cap at 1.0
        return min(total_risk, 1.0)

    # Calculate injury risk for all players
    df["injury_risk"] = df.apply(calculate_injury_risk, axis=1)

    # Missing injury risk -> median (neutral)
    median_risk = df["injury_risk"].median()
    if pd.isna(median_risk):
        median_risk = 0.5  # default moderate risk
    df["injury_risk"] = df["injury_risk"].fillna(median_risk)

    # Filter rows usable for training
    df = df[df["salary"].notna() & (df["salary"] > 0)].copy()
    df = df[df["ppg"].notna()].copy()

    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "ppg",
        "mpg",
        "apg",
        "rpg",
        "spg",
        "bpg",
        "tovpg",
        "FG%",
        "3P%",
        "FT%",
        "fans_w",
        "injury_risk",
    ]
    return [c for c in candidates if c in df.columns]


def _add_derived_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for non-linear relationships.

    Args:
        X: Feature dataframe to add features to
        df: Original dataframe with raw data

    Returns:
        X with derived features added
    """
    X = X.copy()

    # Career stage indicators (helps with rookie contracts, veteran minimums)
    if "age" in df.columns:
        X["is_rookie"] = (df["age"] <= 24).astype(float)
        X["is_prime"] = ((df["age"] > 24) & (df["age"] <= 30)).astype(float)
        X["is_veteran"] = (df["age"] > 30).astype(float)
        X["age_squared"] = df["age"] ** 2

    # Performance interaction terms
    if "ppg" in X.columns and "mpg" in X.columns:
        X["scoring_efficiency"] = X["ppg"] / (X["mpg"] + 1)

    if "apg" in X.columns and "tovpg" in X.columns:
        X["assist_to_turnover"] = X["apg"] / (X["tovpg"] + 0.1)

    # Fan engagement boost (log scale for diminishing returns)
    if "fans_w" in X.columns:
        X["fan_boost"] = np.log1p(X["fans_w"])

    return X


def save_court_impact_weights(
    weights: Dict[str, float],
    metrics: Dict[str, float],
    original_weights: Dict[str, float] = None,
) -> None:
    """
    保存学习到的court_impact权重到JSON配置文件

    Parameters:
    - weights: 学习到的权重
    - metrics: 模型评估指标
    - original_weights: 原始公式权重（用于对比）
    """
    if original_weights is None:
        original_weights = {
            'ppg': 0.45,
            'apg': 0.20,
            'rpg': 0.20,
            'FG%': 0.10,
            '3P%': 0.05,
        }

    # 计算权重变化
    weight_comparison = {}
    for feat in weights.keys():
        learned = weights[feat]
        original = original_weights.get(feat, 0.0)
        change_pct = ((learned - original) / original * 100) if original > 0 else 0.0

        weight_comparison[feat] = {
            "learned": float(learned),
            "original": float(original),
            "change": f"{change_pct:+.1f}%",
        }

    # 构建配置数据
    config = {
        "model_version": "1.0",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "description": "On-court Impact weights learned from WNBA player data using linear regression",
        "target_variable": "mpg * ppg (playing time × scoring ability)",
        "model_performance": {
            "r_squared": float(metrics.get('r2', 0.0)),
            "mae": float(metrics.get('mae', 0.0)),
            "n_samples": int(metrics.get('n_samples', 0)),
        },
        "learned_weights": {k: float(v) for k, v in weights.items()},
        "original_formula_weights": {k: float(v) for k, v in original_weights.items()},
        "weight_comparison": weight_comparison,
        "recommended_formula": {
            "full": f"court_impact = {weights['ppg']:.4f} × ppg + {weights['apg']:.4f} × apg + {weights['rpg']:.4f} × rpg + {weights['FG%']:.4f} × FG% + {weights['3P%']:.4f} × 3P%",
            "simplified": f"court_impact = {weights['ppg']:.2f} × ppg + {weights['apg']:.2f} × apg",
        },
    }

    # 保存到文件
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n权重已保存到: {WEIGHTS_FILE}")


def load_court_impact_weights() -> Dict[str, float]:
    """
    从JSON配置文件加载court_impact权重

    Returns:
    - weights: 权重字典
    """
    if not WEIGHTS_FILE.exists():
        print(f"警告: 权重文件不存在 {WEIGHTS_FILE}，使用默认权重")
        return {
            'ppg': 0.45,
            'apg': 0.20,
            'rpg': 0.20,
            'FG%': 0.10,
            '3P%': 0.05,
        }

    try:
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        weights = config.get('learned_weights', {})
        print(f"\n从配置文件加载权重: {WEIGHTS_FILE}")
        print(f"训练日期: {config.get('training_date', 'unknown')}")
        print(f"模型R-squared: {config.get('model_performance', {}).get('r_squared', 0.0):.4f}")

        return weights

    except Exception as e:
        print(f"警告: 加载权重文件失败 {e}，使用默认权重")
        return {
            'ppg': 0.45,
            'apg': 0.20,
            'rpg': 0.20,
            'FG%': 0.10,
            '3P%': 0.05,
        }


def save_brand_impact_weights(weights: Dict[str, float], metrics: Dict[str, float], *, model_name: str) -> None:
    config = {
        "model_version": "2.0",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "description": "Brand Impact (明星效应) weights learned from WNBA player data",
        "target_variable": "log1p(fans_w) - fan engagement proxy",
        "model": str(model_name),
        "model_performance": {
            "r_squared": float(metrics.get("r2", 0.0)),
            "mae": float(metrics.get("mae", 0.0)),
            "n_samples": int(metrics.get("n_samples", 0)),
        },
        "learned_weights": {k: float(v) for k, v in weights.items()},
        "recommended_formula": {
            "full": (
                "brand_impact = "
                f"{weights.get('efficiency', 0.0):.4f} * (FG% * ppg) + "
                f"{weights.get('ppg', 0.0):.4f} * ppg + "
                f"{weights.get('versatility', 0.0):.4f} * (apg + rpg) + "
                f"{weights.get('mpg', 0.0):.4f} * mpg"
            )
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BRAND_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nBrand Impact weights saved to: {BRAND_WEIGHTS_FILE}")


def learn_brand_impact_weights(
    df: pd.DataFrame,
    *,
    model_name: str = "auto",
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    学习 Brand Impact (明星效应) 的权重。

    公式: brand_impact = w_ppg*ppg + w_mpg*mpg + w_efficiency*(FG%*ppg) + w_versatility*(apg+rpg)

    策略：使用fans_w作为目标变量，学习哪些表现因素驱动粉丝增长

    Returns:
    - weights: 学习到的权重字典
    - metrics: 回归评估指标
    """
    required_cols = ['ppg', 'mpg', 'fans_w']
    available_cols = [c for c in required_cols if c in df.columns]

    if len(available_cols) < 3 or 'fans_w' not in df.columns:
        return {
            'ppg': 0.50,
            'mpg': 0.30,
            'efficiency': 0.20,
        }, {'r2': 0.0, 'note': 'insufficient data, using default weights'}

    # 构建特征：影响明星效应的因素
    feature_names = []
    X_brand = []

    # 1. 得分能力 (ppg) - 明星的核心吸引力
    if 'ppg' in df.columns:
        feature_names.append('ppg')
        X_brand.append(df['ppg'].fillna(0).values)

    # 2. 上场时间 (mpg) - 曝光度
    if 'mpg' in df.columns:
        feature_names.append('mpg')
        X_brand.append(df['mpg'].fillna(0).values)

    # 3. 效率指标 (FG% * ppg) - 高效得分更吸引人
    if 'FG%' in df.columns and 'ppg' in df.columns:
        feature_names.append('efficiency')
        efficiency = df['FG%'].fillna(0) * df['ppg'].fillna(0)
        X_brand.append(efficiency.values)

    # 4. 全能表现 (apg + rpg) - 全面球员更受欢迎
    if 'apg' in df.columns and 'rpg' in df.columns:
        feature_names.append('versatility')
        versatility = df['apg'].fillna(0) + df['rpg'].fillna(0)
        X_brand.append(versatility.values)

    if len(feature_names) < 2:
        return {
            'ppg': 0.50,
            'mpg': 0.30,
            'efficiency': 0.20,
        }, {'r2': 0.0, 'note': 'insufficient features'}

    X_brand = np.column_stack(X_brand)

    # 目标变量：粉丝数 (log scale，因为粉丝增长通常是指数型的)
    y_brand = np.log1p(df['fans_w'].fillna(0).values)

    # 过滤有效样本
    valid_mask = (y_brand > 0) & (X_brand.sum(axis=1) > 0)
    X_brand = X_brand[valid_mask]
    y_brand = y_brand[valid_mask]

    if len(X_brand) < 10:
        return {
            'ppg': 0.50,
            'mpg': 0.30,
            'efficiency': 0.20,
        }, {'r2': 0.0, 'note': 'insufficient valid samples'}

    def _normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
        raw = np.abs(np.asarray(raw_weights, dtype=float))
        s = float(np.sum(raw))
        if s <= 0:
            return np.ones(len(feature_names), dtype=float) / float(len(feature_names))
        return raw / s

    def _r2_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return float(r2), float(mae)

    kind = (model_name or "").strip().lower()
    if kind in ("best", ""):
        kind = "auto"

    if kind in ("rf", "random_forest"):
        kind = "rf"
    elif kind in ("gbm", "gbrt", "gradient_boosting"):
        kind = "gbm"
    elif kind in ("linear", "lin", "lr"):
        kind = "linear"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import GradientBoostingRegressor

    X_train, X_test, y_train, y_test = train_test_split(
        X_brand, y_brand, test_size=0.2, random_state=random_state
    )

    def _fit_once(model_kind: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        if model_kind == "linear":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)

            raw = lr.coef_ / scaler.scale_
            normalized = _normalize_weights(raw)

        elif model_kind == "rf":
            model = RandomForestRegressor(
                n_estimators=600,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            normalized = _normalize_weights(getattr(model, "feature_importances_", np.zeros(len(feature_names))))

        elif model_kind == "gbm":
            model = GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            normalized = _normalize_weights(getattr(model, "feature_importances_", np.zeros(len(feature_names))))

        else:
            raise ValueError("Unknown brand impact model: %r" % model_kind)

        weights_out = {feature_names[i]: float(normalized[i]) for i in range(len(feature_names))}
        r2, mae = _r2_mae(y_test, y_pred)
        metrics_out = {
            "r2": float(r2),
            "mae": float(mae),
            "n_samples": int(len(X_brand)),
            "model": str(model_kind),
        }
        return weights_out, metrics_out

    if kind == "auto":
        best_weights: Dict[str, float] | None = None
        best_metrics: Dict[str, float] | None = None
        best_r2 = float("-inf")

        for candidate in ("linear", "rf", "gbm"):
            w, m = _fit_once(candidate)
            r2 = float(m.get("r2", float("-inf")))
            if np.isfinite(r2) and r2 > best_r2:
                best_weights, best_metrics, best_r2 = w, m, r2

        if best_weights is None or best_metrics is None:
            return _fit_once("linear")
        return best_weights, best_metrics

    return _fit_once(kind if kind else "linear")


def learn_court_impact_weights(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    使用线性回归学习On-court Impact的权重

    公式: court_impact = w1*ppg + w2*apg + w3*rpg + w4*FG% + w5*3P%

    策略：使用综合表现指标（mpg * ppg）作为目标，因为上场时间和得分反映了教练对球员的信任

    Returns:
    - weights: 学习到的权重字典
    - metrics: 回归评估指标
    """
    # 准备特征
    required_cols = ['ppg', 'apg', 'rpg', 'FG%', '3P%', 'mpg']
    available_cols = [c for c in required_cols if c in df.columns]

    if len(available_cols) < 4:
        # 如果数据不足，返回默认权重
        return {
            'ppg': 0.45,
            'apg': 0.20,
            'rpg': 0.20,
            'FG%': 0.10,
            '3P%': 0.05,
        }, {'r2': 0.0, 'note': 'insufficient data, using default weights'}

    # 构建特征矩阵（标准化到相同量纲）
    feature_names = ['ppg', 'apg', 'rpg', 'FG%', '3P%']
    X_impact = []

    for fname in feature_names:
        if fname in df.columns:
            col_data = df[fname].fillna(0).values
            # 标准化：百分比特征乘以100，使其与计数特征在同一量级
            if '%' in fname:
                col_data = col_data * 100
            X_impact.append(col_data)
        else:
            X_impact.append(np.zeros(len(df)))

    X_impact = np.column_stack(X_impact)

    # 目标变量：综合表现 = mpg * ppg（上场时间 × 得分能力）
    # 这反映了球员的实际贡献（既要能得分，又要获得上场机会）
    if 'mpg' in df.columns and 'ppg' in df.columns:
        y_impact = (df['mpg'] * df['ppg']).values
    else:
        # 备选：直接用ppg
        y_impact = df['ppg'].fillna(0).values

    # 过滤掉全0的样本
    valid_mask = (y_impact > 0) & (X_impact.sum(axis=1) > 0)
    X_impact = X_impact[valid_mask]
    y_impact = y_impact[valid_mask]

    if len(X_impact) < 10:
        return {
            'ppg': 0.45,
            'apg': 0.20,
            'rpg': 0.20,
            'FG%': 0.10,
            '3P%': 0.05,
        }, {'r2': 0.0, 'note': 'insufficient valid samples'}

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_impact, y_impact, test_size=0.2, random_state=42
    )

    # 标准化特征（使不同量纲的特征可比）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练线性回归（带截距）
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train_scaled, y_train)

    # 获取权重并转换回原始尺度
    # 权重需要除以标准差来还原
    raw_weights = lr.coef_ / scaler.scale_

    # 取绝对值并归一化
    raw_weights = np.abs(raw_weights)

    # 如果百分比特征的权重过大，进行调整
    # FG%和3P%的权重不应超过总权重的30%
    if raw_weights[3] + raw_weights[4] > 0.5 * raw_weights.sum():
        # 压缩百分比特征的权重
        raw_weights[3] *= 0.3
        raw_weights[4] *= 0.3

    normalized_weights = raw_weights / np.sum(raw_weights)

    # 构建权重字典
    weights = {
        feature_names[i]: float(normalized_weights[i])
        for i in range(len(feature_names))
    }

    # 评估
    y_pred = lr.predict(X_test_scaled)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    mae = np.mean(np.abs(y_test - y_pred))

    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'n_samples': len(X_impact),
    }

    return weights, metrics


def train_pcv_model(
    df: pd.DataFrame,
    *,
    model_name: str = "ridge",
    random_state: int = 42,
) -> Tuple[object, List[str], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Train an improved regression model with non-linear salary prediction.

    改进：
    1. 先用线性回归学习court_impact权重
    2. 学习brand_impact权重 (明星效应)
    3. 使用随机森林预测pred_salary
    4. 添加court_impact和brand_impact作为特征

    Returns:
    - model: fitted estimator or pipeline
    - features: list of feature column names
    - metrics: holdout metrics (R2, MAE, MAPE)
    - court_impact_weights: 学习到的court_impact权重
    - brand_impact_weights: 学习到的brand_impact权重
    """
    # 步骤1: 学习court_impact权重
    print("\n学习On-court Impact权重...")
    court_weights, impact_metrics = learn_court_impact_weights(df)

    print("\n学习到的Court Impact权重:")
    for feat, weight in court_weights.items():
        print(f"  {feat}: {weight:.4f}")
    print(f"\nCourt Impact权重学习R2: {impact_metrics['r2']:.4f}")

    # 保存权重到配置文件
    save_court_impact_weights(court_weights, impact_metrics)

    # 步骤2: 学习brand_impact权重 (明星效应)
    print("\n学习Brand Impact权重 (明星效应)...")
    brand_weights, brand_metrics = learn_brand_impact_weights(df)

    print("\n学习到的Brand Impact权重:")
    for feat, weight in brand_weights.items():
        print(f"  {feat}: {weight:.4f}")
    brand_model_used = str(brand_metrics.get("model", "linear"))
    print(f"\nBrand Impact权重学习R2 ({brand_model_used}): {brand_metrics['r2']:.4f}")

    save_brand_impact_weights(brand_weights, brand_metrics, model_name=brand_model_used)

    # 步骤3: 计算court_impact特征
    df = df.copy()
    df['court_impact'] = 0.0
    for feat, weight in court_weights.items():
        if feat in df.columns:
            df['court_impact'] += weight * df[feat].fillna(0)

    # 步骤4: 计算brand_impact特征
    df['brand_impact'] = 0.0
    for feat, weight in brand_weights.items():
        if feat in df.columns:
            if feat == 'efficiency' and 'FG%' in df.columns and 'ppg' in df.columns:
                df['brand_impact'] += weight * (df['FG%'].fillna(0) * df['ppg'].fillna(0))
            elif feat == 'versatility' and 'apg' in df.columns and 'rpg' in df.columns:
                df['brand_impact'] += weight * (df['apg'].fillna(0) + df['rpg'].fillna(0))
            else:
                df['brand_impact'] += weight * df[feat].fillna(0)

    # 步骤5: 准备特征用于salary预测
    features = _feature_columns(df)
    if not features:
        raise ValueError("No usable feature columns found.")

    X = df[features].copy()

    # 添加court_impact和brand_impact特征
    X['court_impact'] = df['court_impact'].values
    X['brand_impact'] = df['brand_impact'].values

    # Add derived features
    X = _add_derived_features(X, df)

    # Fill remaining NaNs with median
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(float(X[c].median()))

    # Target: Use log transformation for salary
    y = np.log1p(df["salary"].astype(float).values)

    model_name = (model_name or "").strip().lower()

    # scikit-learn imports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.model_selection import train_test_split  # pylint: disable=import-error
        from sklearn.ensemble import GradientBoostingRegressor  # pylint: disable=import-error
        from sklearn.ensemble import RandomForestRegressor  # pylint: disable=import-error
        from sklearn.linear_model import Ridge  # pylint: disable=import-error
        from sklearn.pipeline import Pipeline  # pylint: disable=import-error
        from sklearn.preprocessing import StandardScaler  # pylint: disable=import-error

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=random_state)

    # 默认使用随机森林
    print(f"\n使用{model_name}模型预测salary...")

    if model_name == "rf" or model_name == "gbm":
        # Random Forest: Better for non-linear relationships
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    else:
        # Default: Ridge (more stable + interpretable)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate metrics
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - float(np.mean(y_test))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae_log = float(np.mean(np.abs(y_test - y_pred)))

    # MAPE on original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    mape = float(np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1))) * 100)

    metrics = {
        "r2": r2,
        "mae_log_salary": mae_log,
        "mape": mape,
        "court_impact_r2": impact_metrics['r2'],
        "brand_impact_r2": brand_metrics['r2'],
    }

    return model, list(X.columns), metrics, court_weights, brand_weights


def estimate_pcv(
    *,
    salaries_csv: str | Path = None,
    fans_csv: str | Path = None,
    age_csv: str | Path = None,
    model_name: str = "ridge",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    salaries = load_salaries(salaries_csv)
    fans = load_players_fans(fans_csv)

    # Load age data and calculate injury risk
    try:
        age_data = load_players_age(age_csv)
    except Exception:
        age_data = None

    df = build_pcv_dataset(salaries, fans, age_data)

    model, _, metrics, court_weights, brand_weights = train_pcv_model(df, model_name=model_name, random_state=random_state)

    # Get base features
    base_features = _feature_columns(df)
    X_all = df[base_features].copy()

    # 计算court_impact（使用学习到的权重）
    df['court_impact'] = 0.0
    for feat, weight in court_weights.items():
        if feat in df.columns:
            df['court_impact'] += weight * df[feat].fillna(0)

    # 计算brand_impact（使用学习到的权重）
    df['brand_impact'] = 0.0
    for feat, weight in brand_weights.items():
        if feat in df.columns:
            if feat == 'efficiency' and 'FG%' in df.columns and 'ppg' in df.columns:
                df['brand_impact'] += weight * (df['FG%'].fillna(0) * df['ppg'].fillna(0))
            elif feat == 'versatility' and 'apg' in df.columns and 'rpg' in df.columns:
                df['brand_impact'] += weight * (df['apg'].fillna(0) + df['rpg'].fillna(0))
            else:
                df['brand_impact'] += weight * df[feat].fillna(0)

    X_all['court_impact'] = df['court_impact'].values
    X_all['brand_impact'] = df['brand_impact'].values

    # Add derived features (same as in training)
    X_all = _add_derived_features(X_all, df)

    # Fill NaNs
    for c in X_all.columns:
        if X_all[c].isna().any():
            X_all[c] = X_all[c].fillna(float(X_all[c].median()))

    pred_log_salary = model.predict(X_all.values)
    pred_salary = np.expm1(pred_log_salary)

    df_out = df[["player_name", "salary", "fans_w"]].copy()
    df_out["pred_salary"] = pred_salary
    df_out["pcv"] = df_out["pred_salary"] / float(np.mean(df_out["pred_salary"]))
    df_out["pcv"] = df_out["pcv"].clip(lower=0.1, upper=3.0)

    # Simple signing economics: compare model-implied "fair value" (pred_salary)
    # versus the observed salary (cost).
    df_out["value_gap"] = df_out["pred_salary"] - df_out["salary"]  # >0 => underpaid / good buy
    df_out["value_ratio"] = df_out["pred_salary"] / df_out["salary"]  # >1 => underpaid / good buy

    # Keep some key performance features for inspection
    for c in ["ppg", "mpg", "apg", "rpg", "FG%", "3P%", "FT%", "injury_risk", "age", "court_impact", "brand_impact"]:
        if c in df.columns:
            df_out[c] = df[c].values

    df_out = df_out.sort_values("pcv", ascending=False).reset_index(drop=True)
    return df_out, metrics, court_weights, brand_weights


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate player PCV from salary + fans + age data.")
    parser.add_argument("--salaries", type=str, default=str(RAW_DIR / "wnba_salaries.csv"))
    parser.add_argument("--fans", type=str, default=str(RAW_DIR / "players_fans.csv"))
    parser.add_argument("--age", type=str, default=str(RAW_DIR / "years.csv"))
    parser.add_argument("--model", type=str, default="gbm", choices=["ridge", "rf", "gbm"])
    parser.add_argument("--out", type=str, default=str(PLAYERS_OUTPUT_DIR / "players_pcv.csv"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    df_out, metrics, court_weights, brand_weights = estimate_pcv(
        salaries_csv=args.salaries,
        fans_csv=args.fans,
        age_csv=args.age,
        model_name=args.model,
    )

    if not args.no_write:
        PLAYERS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out, index=False)

    print("\n" + "=" * 80)
    print("PCV模型评估指标:")
    print("=" * 80)
    print(f"  Salary预测R2: {metrics['r2']:.4f}")
    print(f"  Salary预测MAE (log): {metrics['mae_log_salary']:.4f}")
    print(f"  Salary预测MAPE: {metrics['mape']:.2f}%")
    print(f"  Court Impact权重学习R2: {metrics['court_impact_r2']:.4f}")
    print(f"  Brand Impact权重学习R2: {metrics['brand_impact_r2']:.4f}")

    print("\n" + "=" * 80)
    print("学习到的On-court Impact权重:")
    print("=" * 80)
    total_weight = sum(court_weights.values())
    for feat, weight in sorted(court_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:8s}: {weight:.4f} ({weight/total_weight*100:.1f}%)")

    print("\n" + "=" * 80)
    print("学习到的Brand Impact权重 (明星效应):")
    print("=" * 80)
    total_brand_weight = sum(brand_weights.values())
    for feat, weight in sorted(brand_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:12s}: {weight:.4f} ({weight/total_brand_weight*100:.1f}%)")

    print("\n" + "=" * 80)
    print("Top 12球员 (按PCV排序):")
    print("=" * 80)
    print(df_out.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
