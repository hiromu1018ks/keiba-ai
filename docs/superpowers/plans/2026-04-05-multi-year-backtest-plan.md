# マルチ年度バックテスト + HTMLレポート 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2023/2024/2025年を前3年学習でローリングバックテストし、タブ切り替えHTMLレポートを生成する

**Architecture:** 既存 `BacktestEngine` の `bet_history.append()` を拡張して詳細フィールドを追加。新規 `MultiYearReportGenerator` が委譲パターンで既存ヘルパーを再利用し、Jinja2テンプレートで1ファイル・タブ切り替えHTMLを生成。

**Tech Stack:** Python 3.11, LightGBM, pandas, Jinja2, Chart.js, DataTables (CDN)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/backtest/engine.py:143-253` | `bet_history.append()` に12個の拡張フィールド追加 + `top3_finishers` 計算 |
| Modify | `src/backtest/report.py` | `MultiYearReportGenerator` クラス追加（委譲パターン） |
| Create | `src/backtest/templates/multi_year_report.html` | タブ切り替えHTMLテンプレート（全体サマリー + 年度別 + ベット明細） |
| Create | `scripts/run_multi_year_backtest.py` | 年度ループ + コンソール出力 + レポート生成 |
| Modify | `tests/test_backtest_engine.py` | 拡張フィールドのテスト更新 |
| Create | `tests/test_multi_year_report.py` | `MultiYearReportGenerator` のテスト |

---

### Task 1: BacktestEngine の bet_history 拡張

**Files:**
- Modify: `src/backtest/engine.py:143-253`
- Test: `tests/test_backtest_engine.py` (既存テスト確認)

- [ ] **Step 1: engine.py の bet_history.append() を拡張**

`engine.py` のレースループ内（約 L143-257）で、各レースの先頭でレースメタデータを抽出し、`bet_history.append()` に以下の拡張フィールドを追加する。

レースループの先頭（`race_df_single` の作成後、`result_df` の推論前）に以下を追加:

```python
# --- レースメタデータ抽出 (bet_history拡張用) ---
race_row = race_df_single.iloc[0]
race_date_str = (
    f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
    if len(race_id) >= 8 else ""
)
_jyocd = str(race_row.get("jyocd", "")).zfill(2) if pd.notna(race_row.get("jyocd")) else ""
_racenum = int(race_row.get("racenum", 0)) if pd.notna(race_row.get("racenum")) else 0
_grade_code = str(race_row.get("grade_code", "_")) if pd.notna(race_row.get("grade_code")) else "_"
_race_name = str(race_row.get("hondai", "")) if pd.notna(race_row.get("hondai")) else ""
_track_condition = (
    int(race_row.get("track_condition_code", 0))
    if pd.notna(race_row.get("track_condition_code")) else 0
)

# top3_finishers: kakuteijyuni でソートした上位3頭
_valid = race_df_single[
    race_df_single["kakuteijyuni"].notna()
    & (race_df_single["kakuteijyuni"] > 0)
].nsmallest(3, "kakuteijyuni")
_top3: list[dict[str, Any]] = []
for _, r in _valid.iterrows():
    _top3.append({
        "umaban": int(r["umaban"]),
        "bamei": str(r.get("bamei", "")) if pd.notna(r.get("bamei")) else "",
        "kisyuryakusyo": str(r.get("kisyuryakusyo", "")) if pd.notna(r.get("kisyuryakusyo")) else "",
        "kakuteijyuni": int(r["kakuteijyuni"]),
    })
```

次に `bet_history.append()` の dict に以下を追加:

```python
# --- 拡張フィールド ---
"race_date": race_date_str,
"jyocd": _jyocd,
"racenum": _racenum,
"grade_code": _grade_code,
"race_name": _race_name,
"bamei": (
    str(horse_rows.iloc[0].get("bamei", ""))
    if not horse_rows.empty and pd.notna(horse_rows.iloc[0].get("bamei"))
    else ""
),
"kisyu": (
    str(horse_rows.iloc[0].get("kisyuryakusyo", ""))
    if not horse_rows.empty and pd.notna(horse_rows.iloc[0].get("kisyuryakusyo"))
    else ""
),
"kakuteijyuni": (
    int(horse_rows.iloc[0]["kakuteijyuni"])
    if not horse_rows.empty and pd.notna(horse_rows.iloc[0].get("kakuteijyuni"))
    else 0
),
"track_condition_code": _track_condition,
"p_place_pred": float(horse_rows.iloc[0].get("p_place_pred", 0)) if not horse_rows.empty else 0.0,
"e_return_place_pred": float(horse_rows.iloc[0].get("e_return_place_pred", 0)) if not horse_rows.empty else 0.0,
"top3_finishers": _top3,
```

注意: `horse_rows` は既存コードで `result_df[result_df["umaban"] == bet.umaban]` として定義済み。しかし拡張フィールド（`bamei`, `kisyuryakusyo`, `kakuteijyuni`, `p_place_pred`, `e_return_place_pred`）は `result_df` に含まれていることを前提とする。`result_df` は `RacePredictor.predict()` の戻り値であり、元の `feat_df` の列を引き継いでいる。

- [ ] **Step 2: 既存テストを実行して影響確認**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: 全テスト PASS（テストは `bet["surface"]`, `bet["kyori"]` などのキーを `in` でチェックしているため、新しいキーの追加は影響しないはず）

- [ ] **Step 3: テストの feat_df モックに新列を追加**

`tests/test_backtest_engine.py` の `test_engine_populates_enriched_fields` 内の `feat_df` DataFrame に以下の列を追加:

```python
feat_df = pd.DataFrame(
    {
        "race_id": ["20240101010101"],
        "umaban": [1],
        "surface": ["turf"],
        "kyori": [1200],
        "distance_bin": ["sprint"],
        "popularity_rank": [3],
        "ninki": [3],
        "ev_place": [1.5],
        "fukuoddslow": [2.4],
        "kakuteijyuni": [2],
        "kettonum": [1234],
        "odds": [5.0],
        "bataijyu": [480],
        # --- 拡張フィールド用の追加列 ---
        "jyocd": [6],                   # 中山
        "racenum": [11],                # 11R
        "grade_code": ["E"],            # 特別
        "hondai": ["テスト特別"],        # レース名
        "bamei": ["テスト馬"],           # 馬名
        "kisyuryakusyo": ["テスト騎手"], # 騎手名
        "track_condition_code": [1],    # 良
        "p_place_pred": [0.65],         # 複勝確率予測
        "e_return_place_pred": [1.80],  # 期待払戻予測
    }
)
```

- [ ] **Step 4: テストに拡張フィールドの検証を追加**

`test_engine_populates_enriched_fields` の最後にアサーションを追加:

```python
# --- 拡張フィールドの検証 ---
assert "race_date" in bet
assert bet["race_date"] == "2024-01-01"
assert "jyocd" in bet
assert "racenum" in bet
assert bet["racenum"] == 11
assert "grade_code" in bet
assert "bamei" in bet
assert bet["bamei"] == "テスト馬"
assert "kisyu" in bet
assert bet["kisyu"] == "テスト騎手"
assert "kakuteijyuni" in bet
assert bet["kakuteijyuni"] == 2
assert "track_condition_code" in bet
assert "top3_finishers" in bet
assert isinstance(bet["top3_finishers"], list)
assert len(bet["top3_finishers"]) >= 1  # feat_df に1頭のみ
assert bet["top3_finishers"][0]["umaban"] == 1
```

- [ ] **Step 5: テストを実行**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: 全テスト PASS

- [ ] **Step 6: コミット**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: bet_history に詳細フィールド追加 (馬名/騎手/着順/競馬場等)"
```

---

### Task 2: MultiYearReportGenerator クラス追加

**Files:**
- Modify: `src/backtest/report.py`
- Create: `tests/test_multi_year_report.py`

- [ ] **Step 1: テストファイル作成**

`tests/test_multi_year_report.py`:

```python
"""MultiYearReportGenerator のテスト"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backtest.engine import BacktestResult


def _make_result(
    bets: int = 3,
    stake: float = 300.0,
    ret: float = 420.0,
) -> BacktestResult:
    return BacktestResult(
        total_bets=bets,
        total_stake=stake,
        total_return=ret,
        winning_bets=1,
        total_roi=ret / stake if stake > 0 else 0.0,
        max_drawdown=0.05,
        final_bankroll=100000 + ret - stake,
        bet_history=[
            {
                "race_id": "20240106061101",
                "bet_type": "place",
                "umaban": 5,
                "stake": 100.0,
                "odds": 2.3,
                "result": 230.0,
                "surface": "turf",
                "kyori": 1600,
                "ev": 1.25,
                "popularity": 3,
                "bankroll_after": 100130.0,
                "race_date": "2024-01-06",
                "jyocd": "06",
                "racenum": 11,
                "grade_code": "C",
                "race_name": "ポプリS",
                "bamei": "テスト馬",
                "kisyu": "テスト騎手",
                "kakuteijyuni": 2,
                "track_condition_code": 1,
                "p_place_pred": 0.65,
                "e_return_place_pred": 1.80,
                "top3_finishers": [
                    {"umaban": 8, "bamei": "1着馬", "kisyuryakusyo": "川田", "kakuteijyuni": 1},
                    {"umaban": 5, "bamei": "テスト馬", "kisyuryakusyo": "テスト騎手", "kakuteijyuni": 2},
                ],
            },
        ],
    )


class TestMultiYearHtmlGeneration:
    """マルチ年度HTML生成のテスト"""

    def test_html_contains_all_year_tabs(self, tmp_path: Path) -> None:
        """全年度タブが含まれる"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2023: _make_result(), 2024: _make_result()}
        metadata = {
            2023: {"train_start": "2020-01-01", "train_end": "2022-12-31",
                   "test_start": "2023-01-01", "test_end": "2023-12-31"},
            2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                   "test_start": "2024-01-01", "test_end": "2024-12-31"},
        }
        path = gen.generate(results, metadata)
        html = path.read_text(encoding="utf-8")

        assert "2023" in html
        assert "2024" in html
        assert "全体サマリー" in html

    def test_html_contains_bet_detail_tab(self, tmp_path: Path) -> None:
        """ベット明細タブが含まれる"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2024: _make_result()}
        metadata = {2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                           "test_start": "2024-01-01", "test_end": "2024-12-31"}}
        path = gen.generate(results, metadata)
        html = path.read_text(encoding="utf-8")

        assert "ベット明細" in html
        assert "テスト馬" in html
        assert "テスト騎手" in html

    def test_output_path(self, tmp_path: Path) -> None:
        """出力パスが multi_year_report.html"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2024: _make_result()}
        metadata = {2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                           "test_start": "2024-01-01", "test_end": "2024-12-31"}}
        path = gen.generate(results, metadata)

        assert path.name == "multi_year_report.html"
        assert path.parent == tmp_path

    def test_empty_results(self, tmp_path: Path) -> None:
        """空結果でもHTMLが生成される"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        path = gen.generate({}, {})
        html = path.read_text(encoding="utf-8")

        assert path.exists()
        assert "データなし" in html
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_multi_year_report.py -v`
Expected: FAIL (ModuleNotFoundError or ImportError — `MultiYearReportGenerator` がまだ存在しない)

- [ ] **Step 3: MultiYearReportGenerator を実装**

`src/backtest/report.py` の末尾に `MultiYearReportGenerator` クラスを追加:

```python
class MultiYearReportGenerator:
    """マルチ年度バックテスト結果から自己完結型HTMLレポートを生成"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._single_gen = BacktestReportGenerator(output_dir)

    def generate(
        self,
        results: dict[int, BacktestResult],
        metadata: dict[int, dict[str, str]],
    ) -> Path:
        """マルチ年度HTMLレポートを生成"""
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        env.filters["format_number"] = lambda x: f"{x:,.0f}"
        template = env.get_template("multi_year_report.html")

        # 年度別データを組み立て
        year_data: dict[int, dict[str, Any]] = {}
        for year, result in sorted(results.items()):
            enriched = self._single_gen._derive_fields(result.bet_history)
            meta = metadata.get(year, {})
            year_data[year] = {
                "summary": {
                    "roi": result.total_roi,
                    "win_rate": result.winning_bets / result.total_bets if result.total_bets > 0 else 0.0,
                    "profit": result.profit,
                    "max_dd": result.max_drawdown,
                    "final_bankroll": result.final_bankroll,
                    "total_bets": result.total_bets,
                    "total_stake": result.total_stake,
                    "total_return": result.total_return,
                    "total_wins": result.winning_bets,
                    "train_period": f"{meta.get('train_start', '')} ~ {meta.get('train_end', '')}",
                    "test_period": f"{meta.get('test_start', '')} ~ {meta.get('test_end', '')}",
                    "train_seconds": int(float(meta.get("train_seconds", "0"))),
                    "test_seconds": int(float(meta.get("test_seconds", "0"))),
                },
                "monthly_stats": self._single_gen._compute_monthly_stats(enriched),
                "condition_stats": self._single_gen._compute_condition_stats(enriched),
                "bankroll_series": self._single_gen._compute_bankroll_series(enriched),
                "bet_details": enriched,
            }

        # 全体サマリー計算
        all_bets = sum(r.total_bets for r in results.values()) if results else 0
        all_stake = sum(r.total_stake for r in results.values()) if results else 0.0
        all_return = sum(r.total_return for r in results.values()) if results else 0.0
        overall = {
            "total_bets": all_bets,
            "total_stake": all_stake,
            "total_return": all_return,
            "profit": all_return - all_stake,
            "roi": all_return / all_stake if all_stake > 0 else 0.0,
            "best_year": max(results, key=lambda y: results[y].total_roi) if results else 0,
            "worst_year": min(results, key=lambda y: results[y].total_roi) if results else 0,
        }
        if results:
            overall["best_roi"] = results[overall["best_year"]].total_roi
            overall["worst_roi"] = results[overall["worst_year"]].total_roi
        else:
            overall["best_roi"] = 0.0
            overall["worst_roi"] = 0.0

        html = template.render(
            year_data=year_data,
            overall=overall,
            years=sorted(results.keys()),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        outpath = self.output_dir / "multi_year_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
```

- [ ] **Step 4: テストを実行**

Run: `python -m pytest tests/test_multi_year_report.py -v`
Expected: テンプレートがまだないので FAIL（TemplateNotFound）

- [ ] **Step 5: コミット**

```bash
git add src/backtest/report.py tests/test_multi_year_report.py
git commit -m "feat: MultiYearReportGenerator クラス追加 (委譲パターン)"
```

---

### Task 3: HTMLテンプレート作成

**Files:**
- Create: `src/backtest/templates/multi_year_report.html`

- [ ] **Step 1: テンプレート作成**

`src/backtest/templates/multi_year_report.html` を作成。既存 `report.html` をベースに、タブ切り替え構造を追加。

以下はテンプレートの骨格コード（完全版）。実装者はこれをそのままファイルに書き込む:

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>マルチ年度バックテストレポート</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <style>
        :root { --primary: #2563eb; --success: #16a34a; --danger: #dc2626;
                --bg: #f8fafc; --card-bg: #ffffff; --text: #1e293b;
                --text-muted: #64748b; --border: #e2e8f0; }
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: var(--text-muted); margin-bottom: 30px; }
        /* --- タブ --- */
        .tab-bar { display: flex; gap: 2px; border-bottom: 2px solid var(--border); margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; border: none; background: #e2e8f0; color: var(--text-muted);
                   cursor: pointer; font-size: 14px; font-weight: 600; border-radius: 6px 6px 0 0; }
        .tab-btn.active { background: var(--primary); color: white; }
        .tab-btn:hover:not(.active) { background: #cbd5e1; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        /* --- セクション/カード --- */
        .section { background: var(--card-bg); border-radius: 8px; padding: 20px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .section h2 { margin-top: 0; border-bottom: 2px solid var(--primary); padding-bottom: 10px; }
        .section h3 { font-size: 1.1em; margin-top: 20px; margin-bottom: 10px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }
        .kpi-card { background: var(--bg); border-radius: 8px; padding: 15px; text-align: center; }
        .kpi-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
        .kpi-value { font-size: 28px; font-weight: 700; margin-top: 5px; }
        .kpi-value.positive { color: var(--success); }
        .kpi-value.negative { color: var(--danger); }
        .chart-container { position: relative; height: 300px; }
        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .heatmap-positive { background-color: #dcfce7; }
        .heatmap-negative { background-color: #fee2e2; }
        .win-row { background-color: #f0fdf4 !important; }
        .footer { text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 30px; }
        .no-data { color: var(--text-muted); font-style: italic; }
        table { border-collapse: collapse; }
        th, td { padding: 8px 12px; text-align: center; border-bottom: 1px solid var(--border); }
        th { background: #f1f5f9; font-weight: 600; }
        @media (max-width: 768px) { .charts-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <h1>マルチ年度バックテストレポート</h1>
    <p class="subtitle">テスト年度: {{ years | join(' / ') }}</p>
    <p style="text-align:center;color:#94a3b8;font-size:12px;">生成日時: {{ generated_at }}</p>

    <!-- タブバー -->
    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('overview')">全体サマリー</button>
        {% for year in years %}
        <button class="tab-btn" onclick="switchTab('year-{{ year }}')">{{ year }}年</button>
        {% endfor %}
        <button class="tab-btn" onclick="switchTab('bets')">ベット明細</button>
    </div>

    <!-- ===== タブ: 全体サマリー ===== -->
    <div id="tab-overview" class="tab-content active">
        <div class="section">
            <h2>年度比較</h2>
            <div class="kpi-grid">
                {% for year in years %}
                <div class="kpi-card">
                    <div class="kpi-label">{{ year }}年 ROI</div>
                    <div class="kpi-value {{ 'positive' if year_data[year].summary.roi >= 1 else 'negative' }}">
                        {{ "%.1f%%" | format(year_data[year].summary.roi * 100) }}
                    </div>
                </div>
                {% endfor %}
                <div class="kpi-card">
                    <div class="kpi-label">合計 ROI</div>
                    <div class="kpi-value {{ 'positive' if overall.roi >= 1 else 'negative' }}">
                        {{ "%.1f%%" | format(overall.roi * 100) }}
                    </div>
                </div>
            </div>
        </div>
        <div class="section">
            <h2>年度比較表</h2>
            <table style="width:100%;">
                <thead>
                    <tr>
                        <th>年度</th><th>学習期間</th><th>ベット数</th><th>投資額</th>
                        <th>払戻額</th><th>利益</th><th>ROI</th><th>最大DD</th>
                    </tr>
                </thead>
                <tbody>
                {% for year in years %}
                    {% set s = year_data[year].summary %}
                    <tr class="{{ 'heatmap-positive' if s.roi >= 1 else 'heatmap-negative' }}">
                        <td>{{ year }}</td>
                        <td style="font-size:12px;">{{ s.train_period }}</td>
                        <td>{{ s.total_bets }}</td>
                        <td>&yen;{{ s.total_stake | format_number }}</td>
                        <td>&yen;{{ s.total_return | format_number }}</td>
                        <td style="color:{{ '#16a34a' if s.profit >= 0 else '#dc2626' }}">
                            &yen;{{ s.profit | format_number }}
                        </td>
                        <td>{{ "%.1f%%" | format(s.roi * 100) }}</td>
                        <td>{{ "%.1f%%" | format(s.max_dd * 100) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        <div class="section">
            <h2>年度別チャート</h2>
            <div class="charts-row">
                <div class="chart-container"><canvas id="yearly-roi-chart"></canvas></div>
                <div class="chart-container"><canvas id="yearly-bankroll-chart"></canvas></div>
            </div>
        </div>
    </div>

    <!-- ===== タブ: 年度別 (各年) ===== -->
    {% for year in years %}
    <div id="tab-year-{{ year }}" class="tab-content">
        {% set yd = year_data[year] %}
        <div class="section">
            <h2>{{ year }}年 サマリー</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">ROI</div>
                    <div class="kpi-value {{ 'positive' if yd.summary.roi >= 1 else 'negative' }}">
                        {{ "%.1f%%" | format(yd.summary.roi * 100) }}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">的中率</div>
                    <div class="kpi-value">{{ "%.1f%%" | format(yd.summary.win_rate * 100) }}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">利益</div>
                    <div class="kpi-value {{ 'positive' if yd.summary.profit >= 0 else 'negative' }}">
                        &yen;{{ yd.summary.profit | format_number }}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">最大DD</div>
                    <div class="kpi-value negative">{{ "%.1f%%" | format(yd.summary.max_dd * 100) }}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">最終資金</div>
                    <div class="kpi-value">&yen;{{ yd.summary.final_bankroll | format_number }}</div>
                </div>
            </div>
        </div>
        <div class="section">
            <h2>資金推移</h2>
            {% if yd.bankroll_series %}
            <div class="chart-container"><canvas id="bankroll-chart-{{ year }}"></canvas></div>
            {% else %}
            <p class="no-data">データなし</p>
            {% endif %}
        </div>
        <div class="section">
            <h2>月次ダッシュボード</h2>
            {% if yd.monthly_stats %}
            <table style="width:100%;margin-top:10px;">
                <thead>
                    <tr><th>月</th><th>ベット数</th><th>的中</th><th>投資額</th><th>払戻額</th><th>ROI</th></tr>
                </thead>
                <tbody>
                {% for m in yd.monthly_stats %}
                    <tr class="{{ 'heatmap-positive' if m.roi >= 1 else 'heatmap-negative' }}">
                        <td>{{ m.month }}</td><td>{{ m.bets }}</td><td>{{ m.wins }}</td>
                        <td>&yen;{{ m.stake | format_number }}</td>
                        <td>&yen;{{ m.total_return | format_number }}</td>
                        <td>{{ "%.1f%%" | format(m.roi * 100) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p class="no-data">データなし</p>
            {% endif %}
        </div>
        <div class="section">
            <h2>条件分析</h2>
            {% if yd.condition_stats.surface_distance %}
            <table style="width:100%;">
                <thead>
                    <tr><th>路面</th><th>距離帯</th><th>ベット数</th><th>的中率</th><th>ROI</th></tr>
                </thead>
                <tbody>
                {% for row in yd.condition_stats.surface_distance %}
                    <tr class="{{ 'heatmap-positive' if row.roi >= 1 else 'heatmap-negative' }}">
                        <td>{{ "芝" if row.surface == "turf" else "ダート" }}</td>
                        <td>{{ row.distance_band }}</td><td>{{ row.bets }}</td>
                        <td>{{ "%.1f%%" | format(row.win_rate * 100) }}</td>
                        <td>{{ "%.1f%%" | format(row.roi * 100) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p class="no-data">データなし</p>
            {% endif %}
        </div>
    </div>
    {% endfor %}

    <!-- ===== タブ: ベット明細 ===== -->
    <div id="tab-bets" class="tab-content">
        <div class="section">
            <h2>ベット明細 (全年度)</h2>
            <label style="margin-bottom:10px;display:block;">年度フィルタ:
                <select id="year-filter">
                    <option value="all">全年度</option>
                    {% for year in years %}
                    <option value="{{ year }}">{{ year }}年</option>
                    {% endfor %}
                </select>
            </label>
            <table id="all-bets-table" style="width:100%;">
                <thead>
                    <tr>
                        <th>年度</th><th>日付</th><th>競馬場</th><th>R</th><th>レース名</th>
                        <th>馬名</th><th>騎手</th><th>人気</th><th>EV</th><th>オッズ</th>
                        <th>ベット額</th><th>着順</th><th>払戻</th><th>利益</th><th>結果</th>
                    </tr>
                </thead>
                <tbody>
                {% for year in years %}
                    {% for bet in year_data[year].bet_details %}
                    <tr data-year="{{ year }}" class="{{ 'win-row' if bet.is_win }}">
                        <td>{{ year }}</td>
                        <td>{{ bet.race_date }}</td>
                        <td data-jyocd="{{ bet.jyocd }}"></td>
                        <td>{{ bet.racenum }}</td>
                        <td data-grade="{{ bet.grade_code }}">{{ bet.race_name }}</td>
                        <td>{{ bet.bamei }}</td>
                        <td>{{ bet.kisyu }}</td>
                        <td>{{ bet.popularity }}</td>
                        <td>{{ "%.2f" | format(bet.ev) }}</td>
                        <td>{{ "%.1f" | format(bet.odds) }}</td>
                        <td>&yen;{{ bet.stake | format_number }}</td>
                        <td>{{ bet.kakuteijyuni }}</td>
                        <td>&yen;{{ bet.result | format_number }}</td>
                        <td style="color:{{ '#16a34a' if bet.profit > 0 else '#dc2626' }}">
                            &yen;{{ bet.profit | format_number }}
                        </td>
                        <td>{{ "的中" if bet.is_win else "外れ" }}</td>
                    </tr>
                    {% endfor %}
                {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>Generated by keiba-ai multi-year backtest report generator</p>
    </div>
</div>

<script>
// --- タブ切り替え ---
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    event.target.classList.add('active');
}

// --- コード変換マップ ---
const KEIBAJO_MAP = {
    '01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
    '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'
};
const GRADE_MAP = {
    'A':'G1','B':'G2','C':'G3','D':'重賞','E':'特別',
    'F':'J·G1','G':'J·G2','H':'J·G3','_':'一般'
};
const TRACK_CONDITION_MAP = {1:'良',2:'稍重',3:'重',4:'不良'};

// 競馬場セルを変換
document.querySelectorAll('td[data-jyocd]').forEach(td => {
    const code = String(td.dataset.jyocd).padStart(2, '0');
    td.textContent = KEIBAJO_MAP[code] || code;
});
// グレードセルにツールチップ
document.querySelectorAll('td[data-grade]').forEach(td => {
    const g = td.dataset.grade;
    td.title = GRADE_MAP[g] || '';
});

// --- 年度別ROIチャート ---
{% if years %}
(function() {
    const years = {{ years | tojson }};
    const roiData = years.map(y => ({{ year_data | tojson }})[y].summary.roi * 100 - 100);
    new Chart(document.getElementById('yearly-roi-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: years.map(y => y + '年'),
            datasets: [{
                label: 'ROI (%)',
                data: roiData,
                backgroundColor: roiData.map(v => v >= 0 ? '#16a34a' : '#dc2626'),
            }]
        },
        options: { responsive: true, maintainAspectRatio: false,
            scales: { y: { title: { display: true, text: 'ROI (超過%)' } } } },
    });
})();
{% endif %}

// --- 各年度の資金推移チャート ---
{% for year in years %}
{% if year_data[year].bankroll_series %}
(function() {
    const data = {{ year_data[year].bankroll_series | tojson }};
    new Chart(document.getElementById('bankroll-chart-{{ year }}').getContext('2d'), {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: '資金 (¥)',
                data: data.map(d => d.bankroll),
                borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.1)',
                fill: true, tension: 0.1, pointRadius: 0,
            }]
        },
        options: { responsive: true, maintainAspectRatio: false },
    });
})();
{% endif %}
{% endfor %}

// --- DataTables (ベット明細) ---
$(document).ready(function() {
    $('#all-bets-table').DataTable({
        pageLength: 25, order: [[1, 'desc']],
        language: { url: '//cdn.datatables.net/plug-ins/1.13.7/i18n/ja.json' }
    });
    // 年度フィルタ
    $('#year-filter').on('change', function() {
        const val = $(this).val();
        const table = $('#all-bets-table').DataTable();
        if (val === 'all') { table.column(0).search('').draw(); }
        else { table.column(0).search(val).draw(); }
    });
});
</script>
</body>
</html>
```

- [ ] **Step 2: テストを実行してHTML生成確認**

Run: `python -m pytest tests/test_multi_year_report.py -v`
Expected: 全テスト PASS

- [ ] **Step 3: コミット**

```bash
git add src/backtest/templates/multi_year_report.html
git commit -m "feat: マルチ年度バックテストHTMLテンプレート追加"
```

---

### Task 4: run_multi_year_backtest.py スクリプト作成

**Files:**
- Create: `scripts/run_multi_year_backtest.py`

- [ ] **Step 1: スクリプト作成**

```python
"""マルチ年度バックテストスクリプト

使い方:
  python scripts/run_multi_year_backtest.py
  python scripts/run_multi_year_backtest.py --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="マルチ年度バックテスト")
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2023, 2024, 2025],
        help="テスト年度 (デフォルト: 2023 2024 2025)",
    )
    args = parser.parse_args()

    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    logger.info("ParquetStore OK")

    all_results: dict[int, Any] = {}
    all_metadata: dict[int, dict[str, str]] = {}

    for test_year in args.years:
        train_start = f"{test_year - 3}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print()
        print("=" * 50)
        print(f"  {test_year}年 (学習: {train_start[:4]}-{train_end[:4]})")
        print("=" * 50)

        # 学習
        t0 = time.time()
        try:
            from pipelines.training_pipeline import TrainingPipelineV5

            pipeline = TrainingPipelineV5(store=store)
            models = pipeline.run(train_start, train_end)
        except KeyboardInterrupt:
            logger.warning("中断されました")
            sys.exit(1)
        except Exception as e:
            logger.error("%d年 学習失敗: %s — スキップ", test_year, e)
            continue
        elapsed_train = time.time() - t0

        # バックテスト
        t1 = time.time()
        try:
            from backtest.engine import BacktestEngine

            engine = BacktestEngine(models=models, store=store)
            result = engine.run(test_start, test_end)
        except Exception as e:
            logger.error("%d年 テスト失敗: %s — スキップ", test_year, e)
            continue
        elapsed_test = time.time() - t1

        all_results[test_year] = result
        all_metadata[test_year] = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_seconds": str(round(elapsed_train)),
            "test_seconds": str(round(elapsed_test)),
        }

        # 年度サマリー
        profit = result.profit
        print(f"  学習完了 ({elapsed_train:.0f}秒)")
        print(f"  テスト完了 ({elapsed_test:.0f}秒)")
        print(
            f"  ベット数: {result.total_bets:>8,} | "
            f"投資額: ¥{result.total_stake:>10,.0f} | "
            f"払戻: ¥{result.total_return:>10,.0f}"
        )
        print(
            f"  ROI: {result.total_roi:>8.1%} | "
            f"利益: ¥{profit:>+10,.0f} | "
            f"最大DD: {result.max_drawdown:>6.1%}"
        )

    # 全体サマリー
    if not all_results:
        logger.error("全年度失敗。レポートは生成しません。")
        sys.exit(1)

    print()
    print("=" * 50)
    print("  全体サマリー")
    print("=" * 50)
    total_bets = sum(r.total_bets for r in all_results.values())
    total_stake = sum(r.total_stake for r in all_results.values())
    total_return = sum(r.total_return for r in all_results.values())
    total_profit = total_return - total_stake
    total_roi = total_return / total_stake if total_stake > 0 else 0.0
    best_year = max(all_results, key=lambda y: all_results[y].total_roi)
    worst_year = min(all_results, key=lambda y: all_results[y].total_roi)

    print(f"  総ベット数:  {total_bets:>10,}")
    print(f"  総投資額:  ¥{total_stake:>12,.0f}")
    print(f"  総払戻額:  ¥{total_return:>12,.0f}")
    print(f"  総利益:    ¥{total_profit:>+12,.0f}")
    print(f"  合計 ROI:   {total_roi:>10.1%}")
    print(f"  最良年度:  {best_year} ({all_results[best_year].total_roi:.1%})")
    print(f"  最悪年度:  {worst_year} ({all_results[worst_year].total_roi:.1%})")

    # レポート生成
    output_dir = Path(os.path.join(ROOT, "data", "backtest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from backtest.report import MultiYearReportGenerator

    gen = MultiYearReportGenerator(output_dir=output_dir)
    report_path = gen.generate(all_results, all_metadata)
    print(f"\n  レポート生成: {report_path}")

    # JSON保存
    json_data = {
        "overall": {
            "total_bets": total_bets,
            "total_stake": total_stake,
            "total_return": total_return,
            "profit": total_profit,
            "roi": total_roi,
            "best_year": best_year,
            "worst_year": worst_year,
        },
        "years": {},
    }
    for year, result in all_results.items():
        json_data["years"][str(year)] = {
            "total_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "roi": result.total_roi,
            "profit": result.profit,
            "max_drawdown": result.max_drawdown,
            "metadata": all_metadata[year],
        }
    json_path = output_dir / "multi_year_result.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON保存: {json_path}")

    # bet_history 保存
    all_bets: list[dict] = []
    for year, result in all_results.items():
        for bet in result.bet_history:
            bet["_test_year"] = year
            all_bets.append(bet)
    bets_path = output_dir / "multi_year_bet_history.json"
    bets_path.write_text(json.dumps(all_bets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  bet_history保存: {bets_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: mypy チェック**

Run: `python -m mypy scripts/run_multi_year_backtest.py --ignore-missing-imports`
Expected: エラーなし

- [ ] **Step 3: ruff チェック**

Run: `ruff check scripts/run_multi_year_backtest.py`
Expected: エラーなし

- [ ] **Step 4: コミット**

```bash
git add scripts/run_multi_year_backtest.py
git commit -m "feat: マルチ年度バックテストスクリプト追加"
```

---

### Task 5: 全テスト実行 + 統合確認

**Files:**
- なし（確認のみ）

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: 全テスト PASS

- [ ] **Step 2: ruff + mypy**

Run: `ruff check src/ tests/ scripts/run_multi_year_backtest.py && python -m mypy src/`
Expected: エラーなし

- [ ] **Step 3: 最終コミット（必要があれば）**

変更があればコミット。なければスキップ。
