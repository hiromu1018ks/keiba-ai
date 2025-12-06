#!/usr/bin/env python
"""Generate HTML report from simulation results."""
import pandas as pd
import os
import datetime

# Course code mapping
COURSE_NAMES = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉',
    # 地方競馬
    '30': '門別', '31': '盛岡', '32': '水沢', '33': '浦和',
    '34': '船橋', '35': '大井', '36': '川崎', '42': '金沢',
    '43': '笠松', '44': '名古屋', '45': '園田', '46': '姫路',
    '47': '高知', '48': '佐賀', '50': '帯広', '51': 'ばんえい',
    '54': '名古屋', '55': '園田', '56': '姫路'
}

def parse_race_id(race_id: str) -> dict:
    """Parse race ID into components."""
    rid = str(race_id)
    course_code = rid[4:6]
    return {
        'year': rid[0:4],
        'course_code': course_code,
        'kai': int(rid[6:8]),
        'day': int(rid[8:10]),
        'race_num': int(rid[10:12]),
        'course_name': COURSE_NAMES.get(course_code, f'不明({course_code})')
    }

def format_race_title(race_id: str) -> str:
    """Format race ID into readable title."""
    info = parse_race_id(race_id)
    return f"{info['course_name']} 第{info['race_num']}R"

def generate_report():
    """Generate HTML report from simulation results."""
    
    # Load predictions data
    print("Loading simulation_predictions.csv (this may take a moment)...")
    df = pd.read_csv('simulation_predictions.csv', usecols=[
        'race_id', 'date_dt', 'horse_name', 'horse_num', 'jockey', 'trainer',
        'odds', 'pred_prob', 'rank', 'surface', 'distance', 'weather', 'place'
    ])
    
    # Calculate EV
    df['ev'] = df['pred_prob'] * df['odds']
    
    # Filter for 2025 and EV > 3.0
    df['date'] = pd.to_datetime(df['date_dt'])
    df = df[df['date'].dt.year == 2025].copy()
    df = df[df['ev'] > 3.0].copy()
    
    # Filter for JRA only (place codes 01-10)
    jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    df['place_code'] = df['race_id'].astype(str).str[4:6]
    df = df[df['place_code'].isin(jra_places)].copy()
    
    df = df.sort_values('date', ascending=False)
    
    if df.empty:
        print("No 2025 data with EV > 3.0 found.")
        return
    
    print(f"Found {len(df)} bets with EV > 3.0 in 2025")
    
    # Calculate stats
    total_bets = len(df)
    total_hits = (df['rank'] == 1).sum()
    hit_rate = total_hits / total_bets * 100 if total_bets > 0 else 0
    total_invested = total_bets * 100  # 100円/回
    total_return = (df[df['rank'] == 1]['odds'] * 100).sum()
    recovery_rate = total_return / total_invested * 100 if total_invested > 0 else 0

    
    # Monthly stats
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly_stats = df.groupby('month').agg({
        'race_id': 'count',
        'rank': lambda x: (x == 1).sum(),
        'odds': lambda x: (x[df.loc[x.index, 'rank'] == 1] * 100).sum()
    }).reset_index()
    monthly_stats.columns = ['month', 'bets', 'hits', 'return']
    monthly_stats['invested'] = monthly_stats['bets'] * 100
    monthly_stats['recovery'] = monthly_stats['return'] / monthly_stats['invested'] * 100
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>シミュレーション結果レポート - 2025年</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Hiragino Kaku Gothic ProN', 'Meiryo', sans-serif; 
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{ 
            text-align: center; 
            color: #ffd700; 
            margin-bottom: 30px;
            font-size: 2rem;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-label {{
            color: #aaa;
            margin-top: 5px;
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        
        h2 {{
            color: #ffd700;
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-bottom: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
        }}
        th {{ 
            background: rgba(0, 212, 255, 0.2); 
            padding: 15px 10px; 
            text-align: left;
            font-weight: 600;
        }}
        td {{ 
            padding: 12px 10px; 
            border-bottom: 1px solid rgba(255,255,255,0.1); 
        }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        
        .hit {{ 
            background: linear-gradient(90deg, rgba(0,255,100,0.2), transparent) !important;
        }}
        .hit td:first-child {{ 
            border-left: 4px solid #00ff88;
        }}
        
        .miss {{ opacity: 0.6; }}
        
        .filter-bar {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-bar select, .filter-bar input {{
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
        }}
        .filter-bar label {{ color: #aaa; }}
        
        .ev-cell {{ font-weight: bold; }}
        .ev-high {{ color: #00ff88; }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <h1>🏇 シミュレーション結果レポート - 2025年</h1>
    
    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">{total_bets}</div>
            <div class="stat-label">総賭け回数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_hits}</div>
            <div class="stat-label">的中回数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{hit_rate:.1f}%</div>
            <div class="stat-label">的中率</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">¥{total_invested:,.0f}</div>
            <div class="stat-label">投資額 (100円/回)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">¥{total_return:,.0f}</div>
            <div class="stat-label">払戻額</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'positive' if recovery_rate >= 100 else 'negative'}">{recovery_rate:.1f}%</div>
            <div class="stat-label">回収率</div>
        </div>
    </div>
    
    <h2>📊 月別成績</h2>
    <table>
        <thead>
            <tr><th>月</th><th>賭け回数</th><th>的中</th><th>的中率</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr>
        </thead>
        <tbody>
"""
    
    for _, row in monthly_stats.iterrows():
        rate_class = 'positive' if row['recovery'] >= 100 else 'negative'
        hit_rate_m = row['hits'] / row['bets'] * 100 if row['bets'] > 0 else 0
        html += f"""
            <tr>
                <td>{row['month']}</td>
                <td>{row['bets']}</td>
                <td>{row['hits']}</td>
                <td>{hit_rate_m:.1f}%</td>
                <td>¥{row['invested']:,.0f}</td>
                <td>¥{row['return']:,.0f}</td>
                <td class="{rate_class}">{row['recovery']:.1f}%</td>
            </tr>"""
    
    html += """
        </tbody>
    </table>
    
    <h2>📋 賭け履歴 (EV > 3.0)</h2>
    
    <div class="filter-bar">
        <label>結果:</label>
        <select id="resultFilter" onchange="filterTable()">
            <option value="all">全て</option>
            <option value="hit">的中のみ</option>
            <option value="miss">不的中のみ</option>
        </select>
        <label>競馬場:</label>
        <select id="courseFilter" onchange="filterTable()">
            <option value="all">全て</option>
        </select>
    </div>
    
    <table id="betsTable">
        <thead>
            <tr>
                <th>日付</th>
                <th>競馬場</th>
                <th>レース</th>
                <th>馬名</th>
                <th>騎手</th>
                <th>オッズ</th>
                <th>予測確率</th>
                <th>EV</th>
                <th>結果</th>
                <th>払戻</th>
            </tr>
        </thead>
        <tbody>
"""
    
    courses_set = set()
    for _, row in df.iterrows():
        race_title = format_race_title(str(row['race_id']))
        info = parse_race_id(str(row['race_id']))
        courses_set.add(info['course_name'])
        
        is_hit = row['rank'] == 1
        row_class = 'hit' if is_hit else 'miss'
        result = '◎的中' if is_hit else f'{int(row["rank"])}着'
        payout = f"¥{row['odds'] * 100:,.0f}" if is_hit else '-'
        ev_class = 'ev-high' if row['ev'] > 5 else ''
        
        html += f"""
            <tr class="{row_class}" data-result="{'hit' if is_hit else 'miss'}" data-course="{info['course_name']}">
                <td>{row['date'].strftime('%Y-%m-%d')}</td>
                <td>{info['course_name']}</td>
                <td>第{info['race_num']}R</td>
                <td>{row['horse_name']}</td>
                <td>{row['jockey']}</td>
                <td>{row['odds']:.1f}</td>
                <td>{row['pred_prob']:.2%}</td>
                <td class="ev-cell {ev_class}">{row['ev']:.2f}</td>
                <td>{result}</td>
                <td>{payout}</td>
            </tr>"""
    
    # Add course options
    course_options = ''.join([f'<option value="{c}">{c}</option>' for c in sorted(courses_set)])
    
    html += f"""
        </tbody>
    </table>
    
    <div class="footer">
        Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | EV Threshold: 3.0
    </div>
    
    <script>
        // Populate course filter
        document.getElementById('courseFilter').innerHTML += `{course_options}`;
        
        function filterTable() {{
            const resultFilter = document.getElementById('resultFilter').value;
            const courseFilter = document.getElementById('courseFilter').value;
            const rows = document.querySelectorAll('#betsTable tbody tr');
            
            rows.forEach(row => {{
                const result = row.dataset.result;
                const course = row.dataset.course;
                
                let show = true;
                if (resultFilter !== 'all' && result !== resultFilter) show = false;
                if (courseFilter !== 'all' && course !== courseFilter) show = false;
                
                row.style.display = show ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    
    # Save
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'simulation_report_2025.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Total bets: {total_bets}")
    print(f"   Hits: {total_hits} ({hit_rate:.1f}%)")
    print(f"   Recovery: {recovery_rate:.1f}%")
    
    # Open in browser
    import webbrowser
    webbrowser.open(f'file://{os.path.abspath(output_path)}')

if __name__ == "__main__":
    generate_report()
