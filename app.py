import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, date, datetime
import openai
import platform
from matplotlib import font_manager, rc
from dotenv import load_dotenv
import os
import requests
import plotly.graph_objects as go

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("KOREA_EXIM_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")  # NewsAPI 키도 .env에 추가 필요

def set_korean_font():
    try:
        font_path = os.path.join("fonts", "NanumGothicCoding.ttf")
        if not os.path.exists(font_path):
            raise FileNotFoundError("NanumGothicCoding.ttf not found")
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        font_manager.fontManager.addfont(font_path)
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[LOG] 한글 폰트 적용됨: {font_name}")
    except Exception as e:
        print(f"[WARN] 한글 폰트 적용 실패: {e}")

set_korean_font()

# 환율 데이터 전처리
print("[LOG] Loading and preprocessing data...")
df = pd.read_csv("switch.csv")
df['날짜'] = pd.to_datetime(df['변환'], format='%Y/%m/%d', errors='coerce')
df['환율'] = df['원자료'].str.replace(',', '').astype(float)
df = df[['날짜', '환율']].dropna().sort_values('날짜')
df = df.rename(columns={'날짜': 'ds', '환율': 'y'})

# API 환율 데이터 추가
def fetch_korea_exim_rates(start_date, end_date, auth_key):
    url = f"https://www.koreaexim.go.kr/site/program/financial/exchangeJSON?authkey={auth_key}&searchdate={{}}&data=AP01"
    records = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    while current <= end:
        formatted = current.strftime("%Y%m%d")
        try:
            res = requests.get(url.format(formatted))
            data = res.json()
            for item in data:
                if item['cur_unit'] == 'USD':
                    records.append({"ds": current, "y": float(item['deal_bas_r'].replace(',', ''))})
                    break
        except Exception as e:
            print(f"[WARN] Failed to fetch for {formatted}: {e}")
        current += timedelta(days=1)
    return pd.DataFrame(records)

api_df = fetch_korea_exim_rates("2025-05-03", date.today(), api_key)
df = pd.concat([df, api_df]).drop_duplicates(subset="ds").sort_values("ds")

model = Prophet()
model.fit(df)

latest_date = df['ds'].max()

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center;'>
        💱 AI FxSense(AI환율예측시스템)<br>
        <span style='font-size: 0.5em;'>SWitch Money 팀 — 서울교육대학교 소프트웨어영재교육원</span><br>
        <span style='font-size: 0.5em;'>김서진, 김우현, 박재민</span>
    </h1>
""", unsafe_allow_html=True)

if st.button("초기화"):
    st.session_state.clear()
    st.rerun()

st.markdown("<p style='text-align: center;'>기사 내용을 입력하세요:</p>", unsafe_allow_html=True)
article_text = st.text_area("기사 입력", label_visibility="collapsed", height=150)
days = st.number_input("그래프에 표시할 미래 일수", min_value=1, max_value=3650, value=30, step=1)
sensitivity = st.slider("감성 보정 강도", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

def analyze_article_sentiment(article):
    prompt = f"""
    아래 기사가 원/달러 환율에 어떤 영향을 미칠지 판단해줘.
    '긍정적', '부정적', '중립적' 중 하나로 응답하고 간단한 이유도 설명해줘.
    기사:
    {article}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] 감성 분석 실패: {e}"

if st.button("환율 예측 실행"):
    if 'predicted' not in st.session_state:
        st.session_state['predicted'] = True
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        next_day = date.today() + timedelta(days=1)
        next_pred = forecast[forecast['ds'] == pd.to_datetime(next_day)]['yhat'].values
        adj = 0.0
        if article_text.strip():
            st.subheader("AI 감성 분석 결과")
            sentiment_result = analyze_article_sentiment(article_text)
            st.write(sentiment_result)
            if "긍정" in sentiment_result:
                adj = -sensitivity / 100
            elif "부정" in sentiment_result:
                adj = sensitivity / 100
        if len(next_pred) > 0:
            raw = next_pred[0]
            adj_val = raw * (1 + adj)
            st.success(f"예측 환율: {raw:.2f}₩ (감성 보정: {adj_val:.2f}₩)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], label='실제 환율', color='black')
        ax.plot(forecast['ds'], forecast['yhat'], label='예측 환율', color='blue')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)
        ax.axvline(x=latest_date, color='gray', linestyle='--')
        ax.set_xlim([latest_date - timedelta(days=100), forecast['ds'].max()])
        ax.set_ylim(0, 2000)
        st.pyplot(fig)
        st.download_button("예측 결과 다운로드 (CSV)", forecast.to_csv(index=False).encode('utf-8-sig'), "환율_예측.csv")

# 📌 과거 전체 시계열 및 기사 검색 기능
st.subheader("과거 전체 환율 그래프")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='환율'))
fig2.update_layout(title="전체 환율 추이", xaxis_title="날짜", yaxis_title="₩", hovermode='x unified')
st.plotly_chart(fig2, use_container_width=True)

selected_date = st.date_input("기사를 보고 싶은 날짜 선택", value=date.today())

def fetch_news_for_date(target_date):
    query = f"환율+OR+경제+OR+무역+OR+금리"
    url = f"https://newsapi.org/v2/everything?q={query}&from={target_date}&to={target_date}&language=ko&sortBy=relevancy&apiKey={news_api_key}"
    try:
        res = requests.get(url)
        articles = res.json().get('articles', [])[:10]
        return [{"title": a['title'], "url": a['url']} for a in articles]
    except Exception as e:
        return [
            {"title": f"뉴스 로딩 실패: {e}", "url": "#"}
        ]

if st.button("선택한 날짜 기사 보기"):
    news_list = fetch_news_for_date(selected_date)
    st.subheader(f"{selected_date} 주요 뉴스")
    for i, news in enumerate(news_list):
        st.markdown(f"{i+1}. [{news['title']}]({news['url']})")

st.markdown("""
<div style='text-align: center; margin-top: 2em;'>
    <a href='https://ember-ski-cec.notion.site/AI-SWitch-Money-204d0abe582680349dd5e39320e73eaf?source=copy_link'
       target='_blank'
       style='text-decoration: none; font-size: 1.1em; color: #2d79c7; font-weight: 600;'>
        📘 자세한 설명 보러가기 (Notion)
    </a>
</div>
""", unsafe_allow_html=True)
