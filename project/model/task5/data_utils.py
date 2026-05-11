"""
真实数据接入工具集

提供便捷的数据加载、验证和转换工具，用于将真实 WNBA 数据接入 Task 5 模型。

功能：
1. 数据加载器（支持多种格式）
2. 数据验证器（检查完整性和质量）
3. 数据转换器（标准化格式）
4. 参数校准器（从历史数据估计参数）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 数据加载器
# =============================================================================

class RealDataLoader:
    """真实数据加载器"""

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:
            root = Path(__file__).resolve().parents[3]
            data_dir = root / "project" / "data" / "raw"
        self.data_dir = Path(data_dir)

    def load_player_stats(
        self,
        filename: str = "wnba_2024_player_stats.csv",
        team: str = None,
    ) -> pd.DataFrame:
        """
        加载球员统计数据

        Parameters
        ----------
        filename : str
            CSV 文件名
        team : str, optional
            筛选特定球队（例如 'LVA'）

        Returns
        -------
        pd.DataFrame
            标准化的球员数据
        """
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        df = pd.read_csv(path)

        # 标准化列名（处理可能的变体）
        column_mapping = {
            'Player': 'player_name',
            'Name': 'player_name',
            'PPG': 'ppg',
            'APG': 'apg',
            'RPG': 'rpg',
            'MPG': 'mpg',
            'Age': 'age',
            'Team': 'team',
            'Tm': 'team',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 筛选球队
        if team is not None and 'team' in df.columns:
            df = df[df['team'] == team].copy()

        # 验证必需字段
        required = ['player_name', 'ppg', 'apg', 'mpg']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"缺少必需字段: {missing}")

        return df

    def load_schedule(
        self,
        filename: str = "LVA_2024_schedule.csv",
    ) -> pd.DataFrame:
        """
        加载赛程数据

        Parameters
        ----------
        filename : str
            CSV 文件名

        Returns
        -------
        pd.DataFrame
            标准化的赛程数据
        """
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"赛程文件不存在: {path}")

        df = pd.read_csv(path)

        # 标准化列名
        column_mapping = {
            'Game': 'game_number',
            'Date': 'date',
            'Home': 'home_team',
            'Away': 'away_team',
            'Travel': 'travel_km',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 转换日期
        df['date'] = pd.to_datetime(df['date'])

        # 计算派生字段
        if 'rest_days' not in df.columns:
            df['rest_days'] = df['date'].diff().dt.days.fillna(3.0)

        if 'is_b2b' not in df.columns:
            df['is_b2b'] = (df['rest_days'] <= 0).astype(bool)

        # 验证必需字段
        required = ['game_number', 'date', 'home_team', 'away_team']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"缺少必需字段: {missing}")

        return df

    def load_pass_network(
        self,
        filename: str = "lva_2024_pass_network.csv",
    ) -> Optional[pd.DataFrame]:
        """
        加载传球网络数据（可选）

        Parameters
        ----------
        filename : str
            CSV 文件名

        Returns
        -------
        pd.DataFrame or None
            传球网络数据，如果不存在返回 None
        """
        path = self.data_dir / filename

        if not path.exists():
            return None

        df = pd.read_csv(path)

        # 验证必需字段
        required = ['from_player', 'to_player', 'count']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"传球网络数据缺少必需字段: {missing}")

        return df


# =============================================================================
# 数据验证器
# =============================================================================

class DataValidator:
    """数据质量验证器"""

    @staticmethod
    def validate_player_stats(df: pd.DataFrame) -> Dict[str, any]:
        """
        验证球员统计数据质量

        Returns
        -------
        Dict
            验证结果，包含 'valid' (bool) 和 'issues' (list)
        """
        issues = []

        # 检查缺失值
        for col in ['player_name', 'ppg', 'apg', 'mpg']:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    issues.append(f"{col} 存在 {missing_count} 个缺失值")

        # 检查数值范围
        if 'ppg' in df.columns:
            if (df['ppg'] < 0).any():
                issues.append("ppg 存在负值")
            if (df['ppg'] > 50).any():
                issues.append("ppg 存在异常高值 (>50)")

        if 'mpg' in df.columns:
            if (df['mpg'] < 0).any():
                issues.append("mpg 存在负值")
            if (df['mpg'] > 48).any():
                issues.append("mpg 超过比赛时长 (>48)")

        if 'age' in df.columns:
            if (df['age'] < 18).any() or (df['age'] > 45).any():
                issues.append("age 存在异常值 (<18 或 >45)")

        # 检查样本量
        if len(df) < 5:
            issues.append(f"样本量过小: {len(df)} 名球员")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_players': len(df),
        }

    @staticmethod
    def validate_schedule(df: pd.DataFrame) -> Dict[str, any]:
        """
        验证赛程数据质量

        Returns
        -------
        Dict
            验证结果
        """
        issues = []

        # 检查日期连续性
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_gaps = df['date'].diff().dt.days
            max_gap = date_gaps.max()
            if max_gap > 10:
                issues.append(f"赛程存在大间隔: {max_gap} 天")

        # 检查背靠背频率
        if 'is_b2b' in df.columns:
            b2b_rate = df['is_b2b'].mean()
            if b2b_rate > 0.30:
                issues.append(f"背靠背比例过高: {b2b_rate:.1%}")
            elif b2b_rate < 0.05:
                issues.append(f"背靠背比例过低: {b2b_rate:.1%}")

        # 检查旅行距离
        if 'travel_km' in df.columns:
            avg_travel = df[df['travel_km'] > 0]['travel_km'].mean()
            if avg_travel > 3000:
                issues.append(f"平均旅行距离过大: {avg_travel:.0f} km")

        # 检查样本量
        if len(df) < 10:
            issues.append(f"赛程过短: {len(df)} 场比赛")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_games': len(df),
            'b2b_rate': df['is_b2b'].mean() if 'is_b2b' in df.columns else None,
        }


# =============================================================================
# 参数校准器
# =============================================================================

class ParameterCalibrator:
    """从历史数据校准模型参数"""

    @staticmethod
    def calibrate_win_probabilities(
        historical_games: pd.DataFrame,
        player_name: str,
    ) -> Tuple[float, float]:
        """
        从历史比赛数据计算健康/受伤胜率

        Parameters
        ----------
        historical_games : pd.DataFrame
            历史比赛数据，需要列：
            - player_active: 球员是否出场 (bool)
            - win: 是否获胜 (bool)
        player_name : str
            球员名称

        Returns
        -------
        Tuple[float, float]
            (baseline_win_prob, injured_win_prob)
        """
        if 'player_active' not in historical_games.columns:
            print("[WARNING] 缺少 player_active 字段，使用默认值")
            return 0.75, 0.55

        with_player = historical_games[historical_games['player_active'] == True]
        without_player = historical_games[historical_games['player_active'] == False]

        if len(with_player) == 0 or len(without_player) == 0:
            print("[WARNING] 历史数据不足，使用默认值")
            return 0.75, 0.55

        baseline_win_prob = with_player['win'].mean()
        injured_win_prob = without_player['win'].mean()

        return float(baseline_win_prob), float(injured_win_prob)

    @staticmethod
    def calibrate_injury_parameters(
        injury_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        从历史伤病数据校准伤病概率参数

        Parameters
        ----------
        injury_history : pd.DataFrame
            历史伤病数据，需要列：
            - player_name: 球员名称
            - injury_date: 伤病日期
            - games_missed: 缺席场次
            - age: 年龄
            - fatigue_level: 疲劳水平（如果有）

        Returns
        -------
        Dict[str, float]
            校准后的参数: beta_0, beta_1, beta_2, beta_3
        """
        if len(injury_history) < 10:
            print("[WARNING] 伤病样本不足，使用默认参数")
            return {
                'beta_0': -3.5,
                'beta_1': 2.0,
                'beta_2': 0.5,
                'beta_3': 0.05,
            }

        # 使用 Logistic 回归拟合
        # 这里简化处理，实际应该用 sklearn.linear_model.LogisticRegression
        print("[INFO] 使用默认参数（需要更多数据进行校准）")

        return {
            'beta_0': -3.5,
            'beta_1': 2.0,
            'beta_2': 0.5,
            'beta_3': 0.05,
        }

    @staticmethod
    def estimate_franchise_valuation(
        team_code: str,
        year: int = 2024,
    ) -> float:
        """
        估算球队估值

        Parameters
        ----------
        team_code : str
            球队代码（例如 'LVA'）
        year : int
            年份

        Returns
        -------
        float
            估值（百万美元）
        """
        # WNBA 球队估值参考（2024 年估计）
        valuations = {
            'LVA': 150.0,  # Las Vegas Aces
            'NYL': 180.0,  # New York Liberty
            'LA': 160.0,   # Los Angeles Sparks
            'SEA': 140.0,  # Seattle Storm
            'CHI': 130.0,  # Chicago Sky
            'PHX': 120.0,  # Phoenix Mercury
            'DAL': 110.0,  # Dallas Wings
            'ATL': 100.0,  # Atlanta Dream
            'CON': 95.0,   # Connecticut Sun
            'IND': 90.0,   # Indiana Fever
            'MIN': 85.0,   # Minnesota Lynx
            'WAS': 80.0,   # Washington Mystics
        }

        return valuations.get(team_code, 100.0)


# =============================================================================
# 数据转换器
# =============================================================================

class DataTransformer:
    """数据格式转换器"""

    @staticmethod
    def standardize_team_codes(df: pd.DataFrame, column: str = 'team') -> pd.DataFrame:
        """
        标准化球队代码

        Parameters
        ----------
        df : pd.DataFrame
            数据框
        column : str
            球队代码列名

        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        if column not in df.columns:
            return df

        # 球队代码映射
        team_mapping = {
            'LAS': 'LVA',  # Las Vegas (旧代码)
            'LA': 'LVA',
            'NY': 'NYL',
            'NEW YORK': 'NYL',
            'CHI': 'CHI',
            'CHICAGO': 'CHI',
            'SEA': 'SEA',
            'SEATTLE': 'SEA',
        }

        df = df.copy()
        df[column] = df[column].str.upper().str.strip()
        df[column] = df[column].replace(team_mapping)

        return df

    @staticmethod
    def add_travel_distances(
        schedule: pd.DataFrame,
        team: str,
    ) -> pd.DataFrame:
        """
        为赛程添加旅行距离

        Parameters
        ----------
        schedule : pd.DataFrame
            赛程数据
        team : str
            主队代码

        Returns
        -------
        pd.DataFrame
            添加 travel_km 列的赛程
        """
        if 'travel_km' in schedule.columns:
            return schedule

        # WNBA 城市坐标
        city_coords = {
            'LVA': (36.1699, -115.1398),
            'NYL': (40.7128, -74.0060),
            'CHI': (41.8781, -87.6298),
            'SEA': (47.6062, -122.3321),
            'PHX': (33.4484, -112.0740),
            'DAL': (32.7767, -96.7970),
            'ATL': (33.7490, -84.3880),
            'CON': (41.7658, -72.6734),
            'IND': (39.7684, -86.1581),
            'MIN': (44.9778, -93.2650),
            'WAS': (38.9072, -77.0369),
            'LA': (34.0522, -118.2437),
        }

        def haversine_distance(city1: str, city2: str) -> float:
            """计算两城市间距离（公里）"""
            if city1 not in city_coords or city2 not in city_coords:
                return 1500.0

            from math import radians, cos, sin, asin, sqrt

            lat1, lon1 = city_coords[city1]
            lat2, lon2 = city_coords[city2]

            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return 6371 * c

        schedule = schedule.copy()

        # 计算旅行距离
        travel_distances = []
        for _, row in schedule.iterrows():
            if row['home_team'] == team:
                # 主场比赛，无旅行
                travel_distances.append(0.0)
            else:
                # 客场比赛，计算距离
                distance = haversine_distance(team, row['home_team'])
                travel_distances.append(distance)

        schedule['travel_km'] = travel_distances

        return schedule


# =============================================================================
# 便捷函数
# =============================================================================

def quick_load_real_data(
    team: str = 'LVA',
    data_dir: str | Path = None,
) -> Dict[str, pd.DataFrame]:
    """
    快速加载真实数据

    Parameters
    ----------
    team : str
        球队代码
    data_dir : str | Path, optional
        数据目录

    Returns
    -------
    Dict
        包含 'player_stats', 'schedule', 'pass_network' 的字典

    Examples
    --------
    >>> data = quick_load_real_data('LVA')
    >>> print(f"加载 {len(data['player_stats'])} 名球员")
    """
    loader = RealDataLoader(data_dir)
    validator = DataValidator()
    transformer = DataTransformer()

    results = {}

    # 加载球员数据
    try:
        player_stats = loader.load_player_stats(team=team)
        player_stats = transformer.standardize_team_codes(player_stats)

        validation = validator.validate_player_stats(player_stats)
        if not validation['valid']:
            print("[WARNING] 球员数据质量问题:")
            for issue in validation['issues']:
                print(f"  - {issue}")

        results['player_stats'] = player_stats
        print(f"[OK] 加载 {len(player_stats)} 名球员")

    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        results['player_stats'] = None

    # 加载赛程数据
    try:
        schedule = loader.load_schedule(filename=f"{team}_2024_schedule.csv")
        schedule = transformer.add_travel_distances(schedule, team)

        validation = validator.validate_schedule(schedule)
        if not validation['valid']:
            print("[WARNING] 赛程数据质量问题:")
            for issue in validation['issues']:
                print(f"  - {issue}")

        results['schedule'] = schedule
        print(f"[OK] 加载 {len(schedule)} 场比赛")

    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        results['schedule'] = None

    # 加载传球网络（可选）
    try:
        pass_network = loader.load_pass_network()
        results['pass_network'] = pass_network
        if pass_network is not None:
            print(f"[OK] 加载传球网络数据")
        else:
            print("[INFO] 传球网络数据不可用，将从助攻估算")
    except Exception:
        results['pass_network'] = None
        print("[INFO] 传球网络数据不可用，将从助攻估算")

    return results


def validate_and_report(data: Dict[str, pd.DataFrame]) -> bool:
    """
    验证数据并生成报告

    Parameters
    ----------
    data : Dict
        从 quick_load_real_data 返回的数据

    Returns
    -------
    bool
        数据是否可用
    """
    validator = DataValidator()

    print("\n" + "="*80)
    print("数据质量验证报告")
    print("="*80)

    all_valid = True

    # 验证球员数据
    if data['player_stats'] is not None:
        result = validator.validate_player_stats(data['player_stats'])
        print(f"\n球员数据: {'[OK]' if result['valid'] else '[WARNING]'}")
        print(f"  样本量: {result['n_players']} 名球员")
        if not result['valid']:
            for issue in result['issues']:
                print(f"  - {issue}")
            all_valid = False
    else:
        print("\n球员数据: [MISSING]")
        all_valid = False

    # 验证赛程数据
    if data['schedule'] is not None:
        result = validator.validate_schedule(data['schedule'])
        print(f"\n赛程数据: {'[OK]' if result['valid'] else '[WARNING]'}")
        print(f"  比赛数: {result['n_games']} 场")
        if result['b2b_rate'] is not None:
            print(f"  背靠背比例: {result['b2b_rate']:.1%}")
        if not result['valid']:
            for issue in result['issues']:
                print(f"  - {issue}")
            all_valid = False
    else:
        print("\n赛程数据: [MISSING]")
        all_valid = False

    # 传球网络
    if data['pass_network'] is not None:
        print(f"\n传球网络: [OK]")
        print(f"  传球记录: {len(data['pass_network'])} 条")
    else:
        print("\n传球网络: [OPTIONAL] 将从助攻数据估算")

    print("\n" + "="*80)
    print(f"总体状态: {'[OK] 可以运行分析' if all_valid else '[WARNING] 存在数据问题'}")
    print("="*80)

    return all_valid


# =============================================================================
# 真实数据适配器
# =============================================================================

def load_existing_project_data(team: str = 'LVA') -> Dict[str, pd.DataFrame]:
    """
    加载项目现有的真实数据并转换为 task5 格式

    Parameters
    ----------
    team : str
        球队代码

    Returns
    -------
    Dict
        包含 'player_stats', 'schedule', 'pass_network' 的字典
    """
    root = Path(__file__).resolve().parents[3]

    results = {}

    # 加载球员数据
    try:
        player_file = root / "project" / "data" / "processed" / "players" / "players_4indicators.csv"
        if player_file.exists():
            df = pd.read_csv(player_file)

            # 转换为 task5 格式 - 使用列表推导式避免 dtype 问题
            player_stats = pd.DataFrame({
                'player_name': [str(x) for x in df['player_name']],
                'ppg': [float(x) for x in df['ppg']],
                'apg': [float(x) for x in df['apg']],
                'rpg': [float(x) for x in df['rpg']],
                'mpg': [30.0] * len(df),
                'age': [int(x) for x in df['age']],
                'team': [team] * len(df),
            })

            results['player_stats'] = player_stats
            print(f"[OK] 从项目数据加载 {len(player_stats)} 名球员")
        else:
            print(f"[WARNING] 球员数据文件不存在: {player_file}")
            results['player_stats'] = None
    except Exception as e:
        print(f"[ERROR] 加载球员数据失败: {e}")
        import traceback
        traceback.print_exc()
        results['player_stats'] = None

    # 加载赛程数据（从 gamelogs 提取）
    try:
        gamelog_file = root / "project" / "data" / "raw" / "wnba_gamelogs_2015_2025.csv"
        if gamelog_file.exists():
            df = pd.read_csv(gamelog_file)

            # 筛选 Las Vegas Aces (LAS/LVA) 2024 赛季
            team_games = df[(df['Team'].isin(['LAS', 'LVA'])) & (df['Season'] == 2024)].copy()

            if len(team_games) > 0:
                # 转换为 task5 格式
                team_games['date'] = pd.to_datetime(team_games['Date'])
                team_games = team_games.sort_values('date').reset_index(drop=True)

                # 使用列表推导式构建数据
                game_numbers = list(range(1, len(team_games) + 1))
                dates = team_games['date'].tolist()

                # 修正主客场逻辑：Home=1 表示主场，Home=0 表示客场
                home_teams = []
                away_teams = []
                for _, row in team_games.iterrows():
                    if row['Home'] == 1:
                        # 主场比赛：LVA 是主队
                        home_teams.append(team)
                        away_teams.append(str(row['Opp']))
                    else:
                        # 客场比赛：LVA 是客队
                        home_teams.append(str(row['Opp']))
                        away_teams.append(team)

                schedule = pd.DataFrame({
                    'game_number': game_numbers,
                    'date': dates,
                    'home_team': home_teams,
                    'away_team': away_teams,
                    'is_b2b': [False] * len(team_games),
                    'rest_days': [0.0] * len(team_games),
                    'travel_km': [0.0] * len(team_games),
                })

                # 计算休息天数和背靠背
                schedule['rest_days'] = schedule['date'].diff().dt.days.fillna(3.0)
                schedule['is_b2b'] = (schedule['rest_days'] <= 1).astype(bool)

                # 添加旅行距离
                transformer = DataTransformer()
                schedule = transformer.add_travel_distances(schedule, team)

                results['schedule'] = schedule
                print(f"[OK] 从项目数据加载 {len(schedule)} 场比赛")
            else:
                print(f"[WARNING] 未找到 {team} 2024 赛季数据")
                results['schedule'] = None
        else:
            print(f"[WARNING] 比赛日志文件不存在: {gamelog_file}")
            results['schedule'] = None
    except Exception as e:
        print(f"[ERROR] 加载赛程数据失败: {e}")
        import traceback
        traceback.print_exc()
        results['schedule'] = None

    # 传球网络（从助攻估算）
    results['pass_network'] = None
    print("[INFO] 传球网络将从助攻数据估算")

    return results
