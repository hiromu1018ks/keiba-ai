"""ハイパーパラメータチューニング CLI (B2)

Usage:
    python scripts/run_tuning.py --model win_hit --start 20200101 --end 20231231 --trials 50
"""

import argparse
import json
import sys
from pathlib import Path

# src/ を pythonpath に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.parquet_store import ParquetStore
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races
from tuning.optuna_tuner import OptunaTuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model", required=True,
                        choices=["win_hit", "win_return", "place_hit", "place_return", "ability"],
                        help="チューニング対象モデル")
    parser.add_argument("--start", required=True, help="学習開始日 YYYYMMDD")
    parser.add_argument("--end", required=True, help="学習終了日 YYYYMMDD")
    parser.add_argument("--trials", type=int, default=50, help="Optuna試行数")
    args = parser.parse_args()

    store = ParquetStore()
    print(f"Loading data: {args.start} ~ {args.end}")
    race_df = load_races(store, args.start, args.end)
    entry_df = load_entries(store, args.start, args.end)
    odds_df = load_odds_snapshots(store, args.start, args.end)

    # 時系列オッズもロード (odds dynamics 特徴量に必要)
    odds_ts_df = load_odds_time_series_range(store, args.start, args.end)

    # 特徴量生成
    from features.feature_engine import FeatureEngine
    engine = FeatureEngine()
    df = engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=store)

    # レース日ソート (時系列評価の前提)
    df = df.sort_values("race_date").reset_index(drop=True)
    print(f"Data loaded: {len(df)} rows")

    print(f"Tuning {args.model} with {args.trials} trials...")
    tuner = OptunaTuner(model_type=args.model)
    result = tuner.tune(df, n_trials=args.trials)

    print(f"\nBest value: {result['best_value']:.4f}")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")

    # 結果保存
    out_path = Path(f"data/tuning/{args.model}_best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
