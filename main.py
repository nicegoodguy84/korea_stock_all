from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import io
import os
import time
import matplotlib
import base64
import platform
import urllib.request
import re
import warnings

# 💡 HTML 생성 및 차트 렌더링 중 프로그램 튕김(Crash) 현상 방지
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# OS별 폰트 동적 설정 (크로스플랫폼 대응 및 한글 깨짐 방지)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 날짜 설정 (최근 1년)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')


def step1_download_official_krx_to_csv(filename='krx_tickers.csv'):
    """단계 1: 한국거래소(KIND) 공식 상장법인 목록 수집 및 yfinance 호환 티커 기호 부여"""
    print('📊 [단계 1] 한국거래소(KIND) 공식 상장법인 전 종목 목록 수집 중...')

    try:
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        dfs = pd.read_html(io.StringIO(response.text), header=0)
        if not dfs:
            raise ValueError('HTML 테이블을 찾지 못했습니다.')

        df_list = dfs[0]
        df_list = df_list.rename(
            columns={
                '회사명': 'Name',
                '종목코드': 'Symbol',
                '업종': 'Sector',
                '주요제품': 'Industry',
                '상장일': 'ListingDate',
            }
        )
        df_list['Symbol'] = df_list['Symbol'].astype(str).str.zfill(6)

        formatted_data = []
        for _, row in df_list.iterrows():
            sym = row['Symbol']
            name = row['Name']
            sector = row.get('Sector', '기타')
            market = row.get('시장구분', '')

            if not sym.isdigit():
                continue

            if '코스닥' in str(market) or 'KOSDAQ' in str(market):
                market_type = 'KOSDAQ'
                ticker = f'{sym}.KQ'
            else:
                market_type = 'KOSPI'
                ticker = f'{sym}.KS'

            formatted_data.append({
                'Code': ticker,
                'Symbol': sym,
                'Name': name,
                'Market': market_type,
                'Sector': sector if pd.notna(sector) else '기타',
            })

        df_final = pd.DataFrame(formatted_data)
        df_final.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f'✅ 전 종목 총 {len(df_final)}개 마스터 데이터가 "{filename}"에 저장되었습니다.')
        return df_final

    except Exception as e:
        print(f'⚠️ 전 종목 자동 다운로드 중 오류 발생: {e}')
        return pd.DataFrame()


def download_chunk(chunk, retries=3, delay=1.5):
    """실패율을 줄이기 위해 재시도(Retry) 및 지연(Sleep) 기능이 포함된 청크 다운로드"""
    for attempt in range(retries):
        try:
            data = yf.download(
                chunk, start=start_date, end=end_date, progress=False, threads=True
            )['Close']
            
            if not data.empty:
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                return data
        except Exception:
            pass
        
        time.sleep(delay * (attempt + 1))
    return None


def step2_fetch_historical_data(df_meta, benchmark_ticker='^KS11', cache_file='market_closes_cache.pkl'):
    """단계 2: 캐시 즉시 로드 + 멀티스레딩 전 종목 다운로드"""
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if file_mod_time.date() == datetime.today().date():
            print(f'⚡ [단계 2] 오늘의 캐시 데이터("{cache_file}") 로드 완료! (네트워크 다운로드 생략)')
            combined = pd.read_pickle(cache_file)
            benchmark = combined[benchmark_ticker].dropna() if benchmark_ticker in combined.columns else None
            stocks = combined.drop(columns=[benchmark_ticker], errors='ignore')
            return stocks, benchmark

    print(f'📥 [단계 2] 안정화된 멀티스레딩으로 전 종목({len(df_meta)}개) 1년치 시세 일괄 다운로드 중...')

    tickers = df_meta['Code'].tolist() + [benchmark_ticker]
    chunk_size = 150
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    all_closes = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_chunk, chunk): chunk for chunk in chunks}
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            print(f' - 다운로드 진행률: {completed_count}/{len(chunks)} 묶음 완료')
            res = future.result()
            if res is not None:
                all_closes.append(res)
            time.sleep(0.5)

    if not all_closes:
        raise ValueError('데이터를 다운로드하지 못했습니다.')

    combined = pd.concat(all_closes, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.to_pickle(cache_file)

    benchmark = combined[benchmark_ticker].dropna() if benchmark_ticker in combined.columns else None
    stocks = combined.drop(columns=[benchmark_ticker], errors='ignore')
    return stocks, benchmark


def evaluate_single_stock_financials(ticker_code):
    """미너비니 재무 검사 함수 (분기 매출/순이익 YoY 20% 이상 & 연간 ROE 15% 이상)"""
    try:
        stock = yf.Ticker(ticker_code)
        q_financials = stock.quarterly_financials
        annual_financials = stock.financials
        balance_sheet = stock.balance_sheet

        if q_financials.empty or annual_financials.empty or balance_sheet.empty:
            return False, 0, 0, 0

        # --- [1] 분기 성장성 검사 ---
        q_financials = q_financials.reindex(sorted(q_financials.columns), axis=1)
        if 'Total Revenue' not in q_financials.index or 'Net Income' not in q_financials.index:
            return False, 0, 0, 0

        revenue = q_financials.loc['Total Revenue'].dropna()
        net_income = q_financials.loc['Net Income'].dropna()

        if len(revenue) < 5 or len(net_income) < 5:
            return False, 0, 0, 0

        rev_growth_yoy = (revenue.iloc[-1] / revenue.iloc[-5]) - 1
        ni_growth_yoy = (net_income.iloc[-1] / net_income.iloc[-5]) - 1

        # --- [2] ROE 계산 (최근 연간 기준) ---
        if 'Net Income' not in annual_financials.index:
            return False, 0, 0, 0
        latest_annual_ni = annual_financials.loc['Net Income'].iloc[0]

        equity_keys = [
            'Stockholders Equity',
            'Total Equity Gross Minority Interest',
            'Common Stock Equity',
        ]
        total_equity = None
        for key in equity_keys:
            if key in balance_sheet.index:
                total_equity = balance_sheet.loc[key].iloc[0]
                break

        if total_equity and total_equity > 0:
            roe = (latest_annual_ni / total_equity) * 100
        else:
            roe = 0.0

        # --- [3] 최종 미너비니 펀더멘털 조건 판정 ---
        cond_growth = ni_growth_yoy >= 0.20 or rev_growth_yoy >= 0.20
        cond_roe = roe >= 15.0

        if cond_growth and cond_roe:
            return True, rev_growth_yoy * 100, ni_growth_yoy * 100, roe
    except Exception:
        pass

    return False, 0, 0, 0


def detect_vcp_and_pivot(df, lookback=40):
    """마크 미너비니의 핵심인 VCP(변동성 축소 패턴) 및 피벗 돌파 연산"""
    df_recent = df.tail(lookback).copy()
    if len(df_recent) < lookback:
        return False, 1.0, 1.0, "관망"

    std_recent = df_recent['Close'].tail(5).std()
    std_past = df_recent['Close'].iloc[:-10].std()
    vcp_ratio = round(std_recent / std_past, 2) if std_past > 0 else 1.0

    vol_recent_shrink = df_recent['Volume'].tail(3).mean() if 'Volume' in df_recent.columns else 0
    vol_past_shrink = df_recent['Volume'].mean() if 'Volume' in df_recent.columns else 1
    vol_shrink_ratio = round(vol_recent_shrink / vol_past_shrink, 2) if vol_past_shrink > 0 else 1.0

    high_20d = df_recent['High'].iloc[-20:-2].max() if 'High' in df_recent.columns else df_recent['Close'].max()
    current_close = df_recent['Close'].iloc[-1]
    
    if vcp_ratio <= 0.65 and vol_shrink_ratio <= 0.70:
        m_point = "1차 타점 (VCP 수렴 완료)"
    elif current_close >= high_20d and 'Volume' in df_recent.columns and df_recent['Volume'].iloc[-1] > vol_past_shrink * 1.5:
        m_point = "2차 타점 (피벗 거래량 돌파)"
    else:
        m_point = "조건 미달 (수렴 진행중)"

    return True, vcp_ratio, vol_shrink_ratio, m_point


def calculate_minervini_base(df_hist):
    """와인스타인 2단계(Stage 2) 안에서 미너비니 베이스의 카운트를 계산합니다."""
    if len(df_hist) < 200:
        return 1
    
    df_hist['MA200_slope'] = df_hist['MA200'].diff(5)
    stage2_df = df_hist[df_hist['MA200_slope'] > 0]
    
    if len(stage2_df) < 20:
        return 1
        
    base_count = 1
    highest_price = stage2_df['Close'].iloc[0]
    in_correction = False
    
    for idx, row in stage2_df.iterrows():
        price = row['Close']
        if price > highest_price:
            highest_price = price
            if in_correction:
                base_count += 1
                in_correction = False
        elif price < highest_price * 0.88:
            in_correction = True
            
    return min(base_count, 4)


def generate_chart_image(ticker, name, df_hist, w_point, m_point, base_stage, rs_rating):
    """현재 Base는 U자형 곡선, 직전 모든 Base는 타원형으로 시각화하여 Base64로 반환"""
    df_plot = df_hist.tail(120).copy() 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df_plot.index, df_plot['Close'], label='현재가', color='#1e293b', linewidth=2)
    ax1.plot(df_plot.index, df_plot['MA20'], label='20일선', color='#ef4444', linestyle='--', alpha=0.6)
    ax1.plot(df_plot.index, df_plot['MA50'], label='50일선', color='#3b82f6', linestyle='--', alpha=0.5)
    ax1.plot(df_plot.index, df_plot['MA150'], label='150일선(와인스타인)', color='#10b981', linewidth=2)
    ax1.plot(df_plot.index, df_plot['MA200'], label='200일선(미너비니)', color='#8b5cf6', linewidth=1.5, alpha=0.5)

    if len(df_hist) >= 200:
        df_hist['MA200_slope'] = df_hist['MA200'].diff(5)
        stage2_df = df_hist[df_hist['MA200_slope'] > 0]
        
        if len(stage2_df) >= 20:
            bases_info = [] 
            current_base = 1
            base_start_idx = stage2_df.index[0]
            highest_price = stage2_df['Close'].iloc[0]
            in_correction = False
            
            for idx, row in stage2_df.iterrows():
                price_c = row['Close']
                if price_c > highest_price:
                    if in_correction:
                        bases_info.append((base_start_idx, idx, highest_price, current_base))
                        current_base += 1
                        base_start_idx = idx
                        in_correction = False
                    highest_price = price_c
                elif price_c < highest_price * 0.90:
                    in_correction = True
            
            bases_info.append((base_start_idx, stage2_df.index[-1], highest_price, current_base))
            
            for start_dt, end_dt, h_price, b_num in bases_info:
                if end_dt >= df_plot.index[0] and start_dt <= df_plot.index[-1]:
                    plot_start = max(start_dt, df_plot.index[0])
                    plot_end = min(end_dt, df_plot.index[-1])
                    try:
                        y_start = df_plot.loc[plot_start, 'Close']
                        y_end = df_plot.loc[plot_end, 'Close']
                        y_min = df_plot.loc[plot_start:plot_end, 'Close'].min()
                        y_max = df_plot.loc[plot_start:plot_end, 'Close'].max()
                    except KeyError:
                        continue
                
                x_start = mdates.date2num(plot_start)
                x_end = mdates.date2num(plot_end)
                x_mid = (x_start + x_end) / 2
                width = max(x_end - x_start, 5)
                
                if b_num == base_stage:
                    y_control = y_min - ((y_max - y_min) * 0.3 if (y_max > y_min) else h_price * 0.05)
                    path_data = [
                        (patches.Path.MOVETO, (x_start, y_start)),
                        (patches.Path.CURVE3, (x_mid, y_control)),
                        (patches.Path.CURVE3, (x_end, y_end))
                    ]
                    codes, verts = zip(*path_data)
                    path = patches.Path(verts, codes)
                    ax1.add_patch(patches.PathPatch(path, edgecolor='#f59e0b', facecolor='none', lw=2.5, alpha=0.9, zorder=4))
                    ax1.text(mdates.num2date(x_mid), y_control, f" 현재 Base {b_num}기 (곡선 수렴) ", color='#d97706', fontsize=9, fontweight='bold', ha='center', va='top')
                elif b_num < base_stage:
                    ellipse = patches.Ellipse(xy=(x_mid, (y_max+y_min)/2), width=width, height=max((y_max-y_min)*1.2, 100),
                                            edgecolor='#4338ca', facecolor='#e0e7ff', alpha=0.25, lw=2.0, linestyle='--', zorder=3)
                    ax1.add_patch(ellipse)
                    ax1.text(mdates.num2date(x_mid), y_min * 0.96, f" 직전 Base {b_num}기 ", color='#4338ca', fontsize=8, fontweight='bold', ha='center', va='top')

    info_text = f"▶ RS 상대강도: {rs_rating}점\n▶ 미너비니 타점: {m_point} [{base_stage}기]\n▶ 와인스타인 스테이지: {w_point}"
    ax1.text(0.02, 0.92, info_text, transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cbd5e1', alpha=0.9))
    ax1.set_title(f"📈 {name} ({ticker}) 정통 융합 스크리닝 차트", fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    if 'Volume' in df_plot.columns:
        colors = ['#ef4444' if row['Close'] >= row['Open'] else '#3b82f6' for idx, row in df_plot.iterrows() if 'Open' in row]
        if not colors:
            colors = '#3b82f6'
        ax2.bar(df_plot.index, df_plot['Volume'], color=colors, alpha=0.7, width=0.6)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(axis='both', labelsize=9)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def step3_analyze_and_screen(stocks, benchmark, csv_filename='krx_tickers.csv'):
    """단계 3: 전 종목 대상 기술적 조건 통과 후 미너비니 재무(성장성+ROE) 조건 정밀 검증"""
    print('🔍 [단계 3] 전 종목 대상 기술적 조건 스크리닝 및 미너비니 재무 지표 검증 중...')

    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f'{csv_filename} 파일이 존재하지 않습니다.')

    df_meta = pd.read_csv(csv_filename, dtype={'Symbol': str})
    df_meta = df_meta.drop_duplicates(subset=['Symbol']).reset_index(drop=True)
    meta_dict = df_meta.set_index('Symbol').to_dict('index')

    total_scanned = len(stocks.columns)

    # 1. 전 종목 수익률 계산 (거래정지 등 불량 데이터 사전 필터링)
    raw_returns = {}
    history_cache = {}
    for ticker in stocks.columns:
        s = stocks[ticker].dropna()
        if len(s) < 200:
            continue
        if s.iloc[-5:].nunique() == 1 or s.iloc[-1] < 100:
            continue
        raw_returns[ticker] = (s.iloc[-1] / s.iloc[0]) - 1
        
        # DataFrame 형태로 이평선 연산을 위해 보관
        df_h = pd.DataFrame({'Close': s})
        if 'Volume' in s.index: # 데이터프레임 구조 대응
            pass
        history_cache[ticker] = df_h

    if not raw_returns:
        return pd.DataFrame(), total_scanned, 0, {}

    # 2. RS 점수 백분위 정규화 (1~100점)
    returns_series = pd.Series(raw_returns)
    rs_ratings = (returns_series.rank(pct=True) * 100).round(1).to_dict()

    tech_passed_candidates = []

    for ticker in stocks.columns:
        if ticker not in raw_returns:
            continue

        sym = ticker.split('.')[0]
        df_hist = history_cache[ticker]
        s = df_hist['Close']

        df_hist['MA20'] = s.rolling(window=20).mean()
        df_hist['MA50'] = s.rolling(window=50).mean()
        df_hist['MA150'] = s.rolling(window=150).mean()
        df_hist['MA200'] = s.rolling(window=200).mean()

        current_close = s.iloc[-1]
        ma150 = df_hist['MA150'].iloc[-1]
        ma200 = df_hist['MA200'].iloc[-1]

        if pd.isna([current_close, ma150, ma200]).any():
            continue

        # --- [기술적 필터 조건 (와인스타인 & 미너비니 기본 템플릿)] ---
        w_stage2 = (current_close > ma150) and (ma150 > df_hist['MA150'].iloc[-20])
        m_template = (current_close > df_hist['MA50'].iloc[-1]) and (df_hist['MA50'].iloc[-1] > ma150) and (ma150 > ma200)

        if not (w_stage2 and m_template):
            continue

        rs_rating = rs_ratings.get(ticker, 0.0)
        if rs_rating < 70.0:  # RS 70점 이상만 선별
            continue

        info = meta_dict.get(sym, {})
        name = info.get('Name', sym)
        sector = info.get('Sector', '기타')

        tech_passed_candidates.append({
            'ticker': ticker,
            'Symbol': sym,
            'name': name,
            'sector': sector,
            'price': current_close,
            'rs_rating': rs_rating,
            'df_hist': df_hist,
        })

    print(f' - 기술적 + RS 조건 통과 종목: {len(tech_passed_candidates)}개. 미너비니 펀더멘털(매출/순이익/ROE) 검증 시작...')

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_item = {
            executor.submit(evaluate_single_stock_financials, item['ticker']): item
            for item in tech_passed_candidates
        }

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                passed, rev_g, ni_g, roe = future.result()
                if passed:
                    df_hist = item['df_hist']
                    success, vcp_ratio, vol_shrink_ratio, m_point = detect_vcp_and_pivot(df_hist)
                    if not success:
                        continue

                    disparity_150 = round((item['price'] / df_hist['MA150'].iloc[-1]) * 100, 1)
                    w_point = "Stage 2A (돌파 초입 우량)" if disparity_150 <= 112.0 else "Stage 2B (추세 확장 국면)"
                    box_base_stage = calculate_minervini_base(df_hist)

                    score_rs = round((item['rs_rating'] / 100) * 30)
                    score_vcp = 30 if vcp_ratio <= 0.70 else 10
                    score_vol = 25 if vol_shrink_ratio <= 0.70 else 10
                    score_pivot = 15 if disparity_150 <= 105.0 else 5

                    score = score_rs + score_vcp + score_vol + score_pivot
                    rating = "강력매수" if score >= 80 else ("매수" if score >= 55 else "관망/유지")

                    results.append({
                        '종목코드': item['Symbol'],
                        'ticker': item['ticker'],
                        '종목명': item['name'],
                        '업종': item['sector'],
                        '현재가': int(item['price']),
                        '추천등급': rating,
                        '와인스타인지점': w_point,
                        '미너비니지점': m_point,
                        '미너비니베이스': box_base_stage,
                        '변동성축소비율': vcp_ratio,
                        '거래량축소비율': vol_shrink_ratio,
                        '150일선이격도': disparity_150,
                        'RS상대강도(백분위)': item['rs_rating'],
                        '분기매출증가율': round(rev_g, 1),
                        '분기순이익증가율': round(ni_g, 1),
                        '연간ROE': round(roe, 1),
                        '종합점수': score,
                        'score_rs': score_rs,
                        'score_vcp': score_vcp,
                        'score_vol': score_vol,
                        'score_pivot': score_pivot,
                        'df_hist': df_hist
                    })
            except Exception:
                pass

    df_result = pd.DataFrame(results)
    if not df_result.empty:
        df_result = df_result.sort_values(by=['종합점수', 'RS상대강도(백분위)'], ascending=[False, False]).reset_index(drop=True)

    passed_count = len(df_result)
    return df_result, total_scanned, passed_count, history_cache


def step4_generate_combined_html_report(df_result, today_str, total_scanned, passed_count, chart_list):
    """최종 통합 웹 보고서 템플릿 마크업 빌더"""
    table_rows = ""
    for idx, row in df_result.iterrows():
        rank = idx + 1
        rating = row['추천등급']
        badge_style = "bg-danger text-white" if rating == "강력매수" else ("bg-primary text-white" if rating == "매수" else "bg-warning text-dark")
        vcp_active = "text-success font-bold" if row['변동성축소비율'] <= 0.70 else ""

        table_rows += f"""
        <tr>
            <td class="text-center font-bold" style="font-size: 1.1rem; color: #1e293b;">{rank}위</td>
            <td><span class="ticker-badge">{row['종목코드']}</span></td>
            <td><strong>{row['종목명']}</strong></td>
            <td>{row['업종']}</td>
            <td>{row['현재가']:,}원</td>
            <td class="text-center"><span class="badge {badge_style}" style="padding: 6px 12px; border-radius: 20px; font-weight: bold;">{rating}</span></td>
            <td style="font-size: 0.85rem;">
                매출증가: <strong class="text-success">+{row['분기매출증가율']}%</strong><br>
                순이익증가: <strong class="text-success">+{row['분기순이익증가율']}%</strong><br>
                ROE: <strong class="text-primary">{row['연간ROE']}%</strong>
            </td>
            <td class="text-center {vcp_active}">{row['변동성축소비율']}</td>
            <td class="text-center font-bold text-primary" style="font-size: 1.05rem; background-color: #f0fdf4;">{row['RS상대강도(백분위)']}점</td>
            <td class="text-center text-dark font-bold" style="font-size: 1.05rem; background-color: #faf5ff;">
                <strong>{row['종합점수']}점</strong>
            </td>
        </tr>
        """
        
    chart_sections = ""
    for chart in chart_list:
        chart_sections += f"""
        <div class="row align-items-center border-bottom py-4 bg-white px-3 my-3 rounded-3 shadow-sm" style="display: flex;">
            <div style="flex: 0 0 25%; padding-right: 20px;">
                <h4 class="fw-bold text-dark mb-1" style="margin: 0 0 5px 0; font-size: 1.2rem;">{chart['rank']}위. {chart['name']}</h4>
                <p class="text-muted small mb-3" style="margin: 0 0 15px 0; color: #6c757d; font-size: 0.9rem;">[{chart['ticker']}]</p>
                <div class="p-3 bg-light rounded-3 mb-2" style="background: #f8fafc; padding: 15px; border-radius: 8px; font-size: 0.88rem; border: 1px solid #e2e8f0;">
                    <div style="margin-bottom: 6px;"><strong>추천 등급:</strong> <span class="badge bg-danger" style="background-color:#dc3545; color:white; padding:3px 8px; border-radius:10px;">{chart['rating']}</span></div>
                    <div style="margin-bottom: 6px;"><strong>미너비니 단계:</strong> 베이스 {chart['base_stage']}기 현황</div>
                    <div style="margin-bottom: 6px; color: #2563eb;"><strong>🔥 RS 상대강도:</strong> <strong>{chart['rs_rating']}점</strong></div>
                    <div style="margin-bottom: 6px;"><strong>150일선 이격:</strong> {chart['disparity']}%</div>
                    <div class="fw-bold text-primary" style="font-size: 1.05rem; border-top: 1px solid #ddd; padding-top: 5px; margin-top: 5px; font-weight: bold; color: #0d6efd;">종합 스코어: {chart['score']}점 / 100점</div>
                    <div class="mt-2 text-muted" style="font-size: 0.8rem; line-height: 1.4; color:#6c757d; margin-top:8px;">
                        <span style="display:block;">▪ 주도주 RS 점수: {chart['score_rs']}점 / 30</span>
                        <span style="display:block;">▪ VCP 압축 점수: {chart['score_vcp']}점 / 30</span>
                        <span style="display:block;">▪ 거래공백 점수: {chart['score_vol']}점 / 25</span>
                        <span style="display:block;">▪ 피벗매물대 점수: {chart['score_pivot']}점 / 15</span>
                    </div>
                </div>
            </div>
            <div style="flex: 0 0 75%; text-align: center;">
                <img src="data:image/png;base64,{chart['img_base64']}" class="img-fluid rounded border shadow-xs" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #dee2e6;" alt="차트">
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>미너비니 & 와인스타인 국장 전종목 융합 초수익 리포트</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f4f6f9; font-family: 'Malgun Gothic', sans-serif; color: #334155; padding: 40px 0; }}
        .card {{ border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .theory-title {{ border-left: 5px solid #4f46e5; padding-left: 12px; font-weight: 700; }}
        .ticker-badge {{ background-color: #f1f5f9; color: #334155; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-weight: bold; }}
        .stat-card {{ background: linear-gradient(135deg, #4f46e5, #3b82f6); color: white; border-radius: 16px; padding: 25px; text-align: center; }}
        .table th {{ background-color: #f8fafc; color: #64748b; font-weight: 600; text-align: center; }}
        .font-bold {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container" style="max-width: 1350px;">
        <div class="text-center mb-5">
            <h1 class="fw-extrabold" style="color: #0f172a;">🚀 국장 전종목 펀더멘털(ROE·성장) × 기술적 추세추종 융합 스크리너</h1>
            <p class="text-muted fs-5">분석 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]} | 완벽한 데이터 수집 및 시각화 시스템</p>
        </div>
        <div class="row mb-4">
            <div class="col-md-4"><div class="stat-card"><h5>KIND 공식 전체 스캔</h5><h2 class="display-5 fw-bold">{total_scanned}개</h2><p class="mb-0">국내 상장 주식 전 종목 대상</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #10b981, #059669);"><h5>재무+기술 통과 종목</h5><h2 class="display-5 fw-bold">{passed_count}개</h2><p class="mb-0">성장성(매출/이익) + ROE + 정배열</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #f59e0b, #d97706);"><h5>최종 타점 포착 리포트</h5><h2 class="display-5 fw-bold">{len(df_result)}개</h2><p class="mb-0">VCP 수렴 및 입체 차트 시각화</p></div></div>
        </div>
        <div class="card p-4">
            <h3 class="theory-title mb-4">🔍 대가들의 계량적 조건 만족 주도주 종합 랭킹 리스트</h3>
            <div class="table-responsive">
                <table class="table table-hover align-middle">
                    <thead>
                        <tr>
                            <th>순위</th><th>종목코드</th><th>종목명</th><th>업종</th><th>현재가</th><th>추천등급</th>
                            <th>미너비니 펀더멘털 요약</th><th>변동성축소</th>
                            <th style="background-color: #e8f5e9;">RS상대강도</th><th>종합점수</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>
        <div class="card p-4">
            <h3 class="theory-title mb-4">📊 대가들의 기술적 정합성 입체 차트 분석</h3>
            <div class="container-fluid px-0">{chart_sections}</div>
        </div>
    </div>
</body>
</html>
"""
    file_html = f"국장_전종목_융합_리포트_{today_str}.html"
    with open(file_html, "w", encoding="utf-8-sig") as f:
        f.write(html_content)
    return file_html


if __name__ == "__main__":
    print("🚀 [전 종목 펀더멘털 X VCP 정통 스크리너] 시스템 가동")
    csv_filename = 'krx_tickers.csv'

    df_meta = step1_download_official_krx_to_csv(csv_filename)
    if df_meta.empty:
        print("❌ KIND 상장 종목 리스트를 가져오지 못해 실행을 중단합니다.")
    else:
        stocks, benchmark = step2_fetch_historical_data(df_meta, benchmark_ticker='^KS11')
        df_result, total_scanned, passed_count, history_cache = step3_analyze_and_screen(stocks, benchmark, csv_filename)

        if not df_result.empty:
            today_str = datetime.today().strftime("%Y%m%d")
            df_result.to_csv(f"국장_전종목_융합_스크리닝_{today_str}.csv", index=False, encoding='utf-8-sig')

            print(f"\n📊 조건 만족 종목 ({len(df_result)}개) 기술적 연동 차트 빌드 중...")
            chart_list = []
            
            for idx, row in df_result.iterrows():
                code = row['ticker']
                df_hist_target = history_cache.get(code)
                if df_hist_target is not None:
                    img_base64 = generate_chart_image(
                        ticker=row['종목코드'], name=row['종목명'], df_hist=df_hist_target,
                        w_point=row['와인스타인지점'], m_point=row['미너비니지점'],
                        base_stage=row['미너비니베이스'], rs_rating=row['RS상대강도(백분위)']
                    )
                    
                    chart_list.append({
                        'rank': idx + 1, 'ticker': row['종목코드'], 'name': row['종목명'], 'rating': row['추천등급'],
                        'base_stage': row['미너비니베이스'], 'disparity': row['150일선이격도'],
                        'rs_rating': row['RS상대강도(백분위)'], 'score': row['종합점수'], 'img_base64': img_base64,
                        'score_rs': row['score_rs'], 'score_vcp': row['score_vcp'], 'score_vol': row['score_vol'], 'score_pivot': row['score_pivot']
                    })
                    
            file_html = step4_generate_combined_html_report(df_result, today_str, total_scanned, passed_count, chart_list)
            print(f"\n🎉 [엔진 종료] 스크리닝 완료!\n🌐 HTML 종합 보고서: {file_html}")
        else:
            print("\n❌ 모든 조건을 동시에 충족하는 종목이 오늘 시장에 없습니다.")
