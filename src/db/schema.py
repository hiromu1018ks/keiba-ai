"""PostgreSQL スキーマ定義（EveryDB2 対応）

5つのスキーマ:
- raw: EveryDB2 生データのローカルコピー
- odds_history: 時系列オッズ（JODDS_TANPUKU対応）
- feature: 特徴量エンジン出力
- prediction: モデル予測結果
- betting: 投票記録

EveryDB2の外部テーブル(n_race, n_uma_race 等)は読み取り専用。
本スキーマは特徴量・予測・投票の保存用。
"""

SCHEMA_RAW = """
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.races (
    year          INTEGER NOT NULL,
    month_day     VARCHAR(4) NOT NULL,
    jyo_cd        VARCHAR(2) NOT NULL,
    kaiji         VARCHAR(2) NOT NULL,
    nichiji       VARCHAR(2) NOT NULL,
    race_num      VARCHAR(2) NOT NULL,
    track_cd      INTEGER NOT NULL,
    distance      INTEGER NOT NULL,
    tenko_cd      INTEGER NOT NULL,
    baba_cd       INTEGER,
    syubetu_cd    VARCHAR(4) NOT NULL,
    jyoken_cd     VARCHAR(4) NOT NULL,
    grade_cd      VARCHAR(1) NOT NULL DEFAULT '_',
    field_size    INTEGER NOT NULL,
    -- 複合PKを文字列化したrace_id（子テーブルからのFK参照用）
    race_id       VARCHAR(16) GENERATED ALWAYS AS (
        year::text || month_day || jyo_cd || kaiji || nichiji || race_num
    ) STORED UNIQUE,
    surface       VARCHAR(10) GENERATED ALWAYS AS (
        CASE
            WHEN track_cd BETWEEN 10 AND 22 THEN 'turf'
            WHEN track_cd BETWEEN 23 AND 29 THEN 'dirt'
            ELSE 'exclude'
        END
    ) STORED,
    distance_band VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN track_cd BETWEEN 10 AND 22 THEN
                CASE
                    WHEN distance <= 1400 THEN 'sprint'
                    WHEN distance <= 1700 THEN 'mile'
                    WHEN distance <= 2100 THEN 'intermediate'
                    ELSE 'long'
                END
            WHEN track_cd BETWEEN 23 AND 29 THEN
                CASE
                    WHEN distance <= 1400 THEN 'sprint'
                    WHEN distance <= 1700 THEN 'mile'
                    ELSE 'intermediate'
                END
            ELSE NULL
        END
    ) STORED,
    PRIMARY KEY (year, month_day, jyo_cd, kaiji, nichiji, race_num)
);

CREATE TABLE IF NOT EXISTS raw.entries (
    race_id       VARCHAR(16) NOT NULL REFERENCES raw.races(race_id) ON DELETE CASCADE,
    umaban        INTEGER NOT NULL,
    ketto_num     VARCHAR(10) NOT NULL,
    finish_pos    INTEGER NOT NULL DEFAULT 0,
    finish_time   FLOAT,
    haron_time_l3 FLOAT,
    ninki         INTEGER,
    win_odds      FLOAT,
    ba_taijyu     FLOAT,
    zogen_fugo    VARCHAR(1),
    zogen_sa      FLOAT,
    kisyu_code    VARCHAR(10),
    chokyosi_code VARCHAR(10),
    kyakusitu     INTEGER,
    honsyokin     INTEGER,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS raw.payouts (
    race_id       VARCHAR(16) NOT NULL REFERENCES raw.races(race_id) ON DELETE CASCADE,
    tan_umaban    INTEGER,
    tan_pay       FLOAT,
    fuku_umaban1  INTEGER,  fuku_pay1  FLOAT,
    fuku_umaban2  INTEGER,  fuku_pay2  FLOAT,
    fuku_umaban3  INTEGER,  fuku_pay3  FLOAT,
    fuku_umaban4  INTEGER,  fuku_pay4  FLOAT,
    fuku_umaban5  INTEGER,  fuku_pay5  FLOAT,
    PRIMARY KEY (race_id)
);
"""

SCHEMA_ODDS_HISTORY = """
CREATE SCHEMA IF NOT EXISTS odds_history;

CREATE TABLE IF NOT EXISTS odds_history.odds_snapshots (
    race_id    VARCHAR(16) NOT NULL,
    umaban     INTEGER NOT NULL,
    tan_odds   FLOAT,
    fuku_odds  FLOAT,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS odds_history.odds_time_series (
    race_id     VARCHAR(16) NOT NULL,
    happyo_time VARCHAR(8) NOT NULL,  -- MMDDHHmm
    umaban      INTEGER NOT NULL,
    tan_odds    FLOAT,
    fuku_odds   FLOAT,
    ninki       INTEGER,
    PRIMARY KEY (race_id, happyo_time, umaban)
);

CREATE TABLE IF NOT EXISTS odds_history.wide_odds (
    race_id     VARCHAR(16) NOT NULL,
    kumi        VARCHAR(10) NOT NULL,  -- "3-7" 形式
    odds_low    FLOAT,
    odds_high   FLOAT,
    PRIMARY KEY (race_id, kumi)
);
"""

SCHEMA_FEATURE = """
CREATE SCHEMA IF NOT EXISTS feature;

CREATE TABLE IF NOT EXISTS feature.features (
    race_id     VARCHAR(16) NOT NULL,
    umaban      INTEGER NOT NULL,
    surface     VARCHAR(5) NOT NULL,
    feature_data JSONB NOT NULL,  -- 特徴量の辞書をJSONで保存
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, umaban)
);
"""

SCHEMA_PREDICTION = """
CREATE SCHEMA IF NOT EXISTS prediction;

CREATE TABLE IF NOT EXISTS prediction.predictions (
    race_id               VARCHAR(16) NOT NULL,
    umaban                INTEGER NOT NULL,
    surface               VARCHAR(5) NOT NULL,
    p_ability_win         FLOAT,
    p_ability_place       FLOAT,
    signed_log_error_win  FLOAT,
    abs_log_error_win     FLOAT,
    p_win_pred            FLOAT,
    ev_win                FLOAT,
    p_win_corrected       FLOAT,
    ev_win_corrected      FLOAT,
    ev_lower_win_corrected FLOAT,
    p_place_pred          FLOAT,
    ev_place              FLOAT,
    ev_lower_place        FLOAT,
    wide_score_adj        FLOAT,
    predicted_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, umaban)
);
"""

SCHEMA_BETTING = """
CREATE SCHEMA IF NOT EXISTS betting;

CREATE TABLE IF NOT EXISTS betting.bets (
    id                    SERIAL PRIMARY KEY,
    race_id               VARCHAR(16) NOT NULL,
    umaban                INTEGER NOT NULL,
    bet_type              VARCHAR(5) NOT NULL,
    odds                  FLOAT NOT NULL,
    ev_lower_corrected    FLOAT NOT NULL,
    stake                 INTEGER NOT NULL,
    result_payout         FLOAT,
    profit                FLOAT,
    regime_state          VARCHAR(15),
    recovery_state        VARCHAR(15),
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at            TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bets_race_id ON betting.bets (race_id);
CREATE INDEX IF NOT EXISTS idx_bets_created_at ON betting.bets (created_at);
CREATE INDEX IF NOT EXISTS idx_bets_bet_type ON betting.bets (bet_type);
"""

ALL_CREATE_STATEMENTS = [
    SCHEMA_RAW,
    SCHEMA_ODDS_HISTORY,
    SCHEMA_FEATURE,
    SCHEMA_PREDICTION,
    SCHEMA_BETTING,
]
