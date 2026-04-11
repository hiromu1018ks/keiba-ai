"""データクラス定義（Race, Entry, Bet, OddsSnapshot, DDState 等）"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from domain.types import BetType, RecoveryState, Surface

if TYPE_CHECKING:
    from models.ev_correction_model import EVCorrectionModel
    from models.market_model import MarketModel
    from models.place_ability_model import PlaceAbilityModel
    from models.race_quality_screener import RaceQualityScreener
    from models.regime_detector import RegimeDetector
    from models.robust_confidence_estimator import RobustConfidenceEstimator
    from models.stage1_ability_model import AbilityModel
    from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
    from models.wide_two_stage_model import WideTwoStageModel


def _surface_from_track_cd(track_cd: int) -> Surface:
    """TrackCD から Surface を判定（設計書 everydb2-data-reference §3.1）"""
    if 10 <= track_cd <= 22:
        return Surface.TURF
    elif 23 <= track_cd <= 29:
        return Surface.DIRT
    else:
        raise ValueError(f"未対応の TrackCD: {track_cd} (障害51-59は除外前提)")


def _distance_band(surface: Surface, distance: int) -> str:
    """サーフェスと距離から距離帯を返す（設計書 everydb2-data-reference §3.1）"""
    if surface == Surface.TURF:
        if distance <= 1400:
            return "sprint"
        elif distance <= 1700:
            return "mile"
        elif distance <= 2100:
            return "intermediate"
        else:
            return "long"
    else:  # DIRT
        if distance <= 1400:
            return "sprint"
        elif distance <= 1700:
            return "mile"
        else:
            return "intermediate"


@dataclass(frozen=True)
class Race:
    """レース情報（n_race テーブル対応）

    複合主キー: (year, month_day, jyo_cd, kaiji, nichiji, race_num)
    """

    year: int
    month_day: str  # MMDD
    jyo_cd: str  # 場所コード 01-10
    kaiji: str  # 回次
    nichiji: str  # 日次
    race_num: str  # レース番号
    track_cd: int  # トラックコード
    distance: int  # 距離(m)
    tenko_cd: int  # 天候コード
    baba_cd: int  # 馬場状態コード
    syubetu_cd: str  # 種別コード
    jyoken_cd: str  # 条件コード
    grade_cd: str  # グレードコード
    field_size: int  # 頭数

    # --- 計算プロパティ ---
    @property
    def surface(self) -> Surface:
        return _surface_from_track_cd(self.track_cd)

    @property
    def distance_band(self) -> str:
        return _distance_band(self.surface, self.distance)

    @property
    def race_id(self) -> str:
        """複合主キーを文字列化: YYYYMMDDJyoKaiNiRace"""
        return f"{self.year}{self.month_day}{self.jyo_cd}{self.kaiji}{self.nichiji}{self.race_num}"

    @property
    def is_good_track(self) -> bool:
        """良 or 稍重"""
        return self.baba_cd in (1, 2)

    @property
    def is_soft_track(self) -> bool:
        """重 or 不良"""
        return self.baba_cd in (3, 4)

    @property
    def is_steeple(self) -> bool:
        """障害レース"""
        return self.track_cd >= 51

    @property
    def grade_name(self) -> str:
        grade_map = {"A": "G1", "B": "G2", "C": "G3", "D": "重賞", "E": "特別"}
        return grade_map.get(self.grade_cd, "一般")


@dataclass
class Entry:
    """出走馬情報（n_uma_race テーブル対応）"""

    race_id: str
    umaban: int  # 馬番
    ketto_num: str  # 血統番号
    finish_pos: int  # 確定着順 (1=1着, 0=取消等)
    win_odds_actual: float  # 確定単勝オッズ
    popularity_rank: int  # 人気順位
    running_style: int  # 脚質 (1=逃げ, 2=先行, 3=差し, 4=追込, 0=不明)
    ba_taijyu: float  # 馬体重
    zogen_fugo: int  # 体重増減符号 (1=増, 2=減, 3=不变)
    zogen_sa: float  # 体重増減幅
    kisyu_code: str  # 騎手コード
    chokyosi_code: str  # 調教師コード

    @property
    def is_winner(self) -> bool:
        return self.finish_pos == 1

    @property
    def is_place(self) -> bool:
        return 1 <= self.finish_pos <= 3

    @property
    def is_cancelled(self) -> bool:
        return self.finish_pos == 0

    @property
    def running_style_name(self) -> str:
        style_map = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}
        return style_map.get(self.running_style, "不明")


@dataclass
class Bet:
    """投票情報"""

    race_id: str
    umaban: int
    bet_type: BetType
    odds: float  # ベット判定に使用したオッズ（発走前 or 確定）
    ev_lower_corrected: float  # EV下限値（補正済み）
    stake: float  # 投票金額
    final_odds: float = 0.0  # 精算用オッズ（確定オッズ）
    result: Optional[float] = None  # 払戻金（確定後）

    @property
    def is_valid(self) -> bool:
        """最低投票額 100円以上"""
        return self.stake >= 100

    @property
    def profit(self) -> float:
        """利益（払戻 - 投票額）"""
        if self.result is None:
            return 0.0
        return self.result - self.stake


@dataclass
class OddsSnapshot:
    """時系列オッズスナップショット（n_jodds_tanpuku テーブル対応）"""

    race_id: str
    happyo_time: str  # 発表時刻 MMDDHHmm
    umaban: int
    tan_odds: float  # 単勝オッズ
    fuku_odds: float  # 複勝オッズ


@dataclass
class DDState:
    """ドローダウン状態（§9 DDコントローラー用）"""

    current_dd: float
    rolling_roi: float
    n_bets_eval: int
    recovery_state: RecoveryState = RecoveryState.NORMAL


@dataclass
class RegimeConfig:
    """レジーム検知設定（§9.5）"""

    window: int = 200
    min_samples: int = 100
    fav_rate_aggressive: float = 0.28
    fav_rate_collapsed: float = 0.18
    overround_base: float = 0.20
    retrain_trigger: int = 100


@dataclass
class TwoStageConfig:
    """2段階モデルハイパーパラメータ（§2）"""

    hit_metric: str = "auc"
    hit_leaves: int = 31
    hit_lr: float = 0.03
    hit_rounds: int = 500
    return_metric: str = "mae"
    return_leaves: int = 15
    return_lr: float = 0.03
    return_rounds: int = 300
    min_hit_samples: int = 200


@dataclass
class SubmodelSet:
    """サブモデル（芝/ダート）のセット

    TrainingPipelineV5 が各 surface ごとに生成する。
    """

    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False


@dataclass
class TrainedModelsV5:
    """学習済みモデルのコンテナ (§11)

    全サブモデル + RaceQualityScreener + RegimeDetector を保持。
    TrainingPipelineV5.run() の戻り値。
    """

    submodels: dict[str, SubmodelSet]
    quality_screener: RaceQualityScreener
    regime_detector: RegimeDetector
    train_period: tuple[str, str] = field(default=("2020-01-01", "2023-12-31"))


@dataclass
class SafetyConfig:
    """SafetyGuard 設定 (§12 ステップ ⑪)"""

    min_bankroll: float = 10000.0
    max_daily_loss: float = 10000.0
    max_weekly_loss: float = 30000.0
    max_consecutive_losses: int = 10


@dataclass(frozen=True)
class SafetyCheckResult:
    """SafetyGuard チェック結果"""

    can_bet: bool
    reason: str = ""
