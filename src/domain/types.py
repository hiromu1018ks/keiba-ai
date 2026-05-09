"""Enum 定義と型エイリアス"""

from enum import Enum


class Surface(str, Enum):
    """芝/ダートのサーフェス"""

    TURF = "turf"
    DIRT = "dirt"


class BetType(str, Enum):
    """投票タイプ"""

    WIN = "win"
    PLACE = "place"
    WIDE = "wide"


class RecoveryState(str, Enum):
    """ドローダウン回復状態（DDコントローラー用）"""

    NORMAL = "normal"
    REDUCED = "reduced"
    STOP = "stop"


class RegimeState(str, Enum):
    """市場レジーム状態"""

    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    COLLAPSED = "collapsed"


# レース後のみ入手可能な列 — ML特徴量として使用禁止
POST_RACE_COLS: list[str] = [
    "kakuteijyuni",
    "confirmed_odds",
    "ninki",
    "kyakusitukubun",
    "time",
    "timediff",
    "harontimel3",
    "harontimel4",
    "jyuni1c",
    "jyuni2c",
    "jyuni3c",
    "jyuni4c",
    "honsyokin",
    "chakusacd",
    "dmjyuni",
    "dmtime",
]
