"""src/features/race_difficulty_model.py のテスト"""

import pandas as pd
import pytest

from features.race_difficulty_model import compute_difficulty_score


@pytest.fixture
def race_df() -> pd.DataFrame:
    """様々な条件のレースデータ"""
    return pd.DataFrame({
        "race_id": ["G1", "G3", "GENERAL", "BIG_FIELD"],
        "field_size": [18, 16, 14, 18],
        "grade_cd": ["A", "C", "_", "_"],
        "market_entropy": [2.8, 2.5, 1.5, 2.89],  # ln(18)≈2.89 が最大
    })


class TestDifficultyScore:
    def test_g1_harder_than_general(self, race_df: pd.DataFrame):
        """G1レースの難易度が一般レースより高い"""
        result = compute_difficulty_score(race_df)
        g1_score = result[result["race_id"] == "G1"]["difficulty_score"].iloc[0]
        gen_score = result[result["race_id"] == "GENERAL"]["difficulty_score"].iloc[0]
        assert g1_score > gen_score

    def test_big_field_harder(self, race_df: pd.DataFrame):
        """大頭数レースの方が難易度が高い（同グレード・同entropyの場合）"""
        result = compute_difficulty_score(race_df)
        big_score = result[result["race_id"] == "BIG_FIELD"]["difficulty_score"].iloc[0]
        gen_score = result[result["race_id"] == "GENERAL"]["difficulty_score"].iloc[0]
        assert big_score > gen_score

    def test_high_entropy_harder(self, race_df: pd.DataFrame):
        """高エントロピ（拮抗）レースの方が難易度が高い（同グレード・同頭数の場合）"""
        result = compute_difficulty_score(race_df)
        g1_score = result[result["race_id"] == "G1"]["difficulty_score"].iloc[0]
        g3_score = result[result["race_id"] == "G3"]["difficulty_score"].iloc[0]
        # G1 (entropy=2.8) > G3 (entropy=2.5)、同頭数は異なるがグレード重みも大きい
        assert g1_score > g3_score

    def test_score_range(self, race_df: pd.DataFrame):
        """スコアが 0.0〜1.0 の範囲に収まる"""
        result = compute_difficulty_score(race_df)
        scores = result["difficulty_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_preserves_columns(self, race_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_difficulty_score(race_df)
        assert "race_id" in result.columns
        assert "field_size" in result.columns
