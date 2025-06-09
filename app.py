import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, date
import openai
import platform
from matplotlib import font_manager, rc
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("KOREA_EXIM_API_KEY")


def set_korean_font():
    try:
        font_path = os.path.join("fonts", "NanumGothicCoding.ttf")
        if not os.path.exists(font_path):
            raise FileNotFoundError("NanumGothicCoding.ttf not found")
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        font_manager.fontManager.addfont(font_path)  # 폰트 등록
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[LOG] 한글 폰트 적용됨: {font_name}")
    except Exception as e:
        print(f"[WARN] 한글 폰트 적용 실패: {e}")



set_korean_font()

# 환율 데이터 로딩 및 전처리
print("[LOG] Loading and preprocessing data...")
df = pd.read_csv("switch.csv")
df['날짜'] = pd.to_datetime(df['변환'], format='%Y/%m/%d', errors='coerce')
df['환율'] = df['원자료'].str.replace(',', '').astype(float)
df = df[['날짜', '환율']].dropna().sort_values('날짜')
df = df.rename(columns={'날짜': 'ds', '환율': 'y'})

# Prophet 모델 학습
print("[LOG] Fitting Prophet model...")

# 한국수출입은행 API에서 2025-05-03 이후 실시간 데이터 가져오기 
import requests
from datetime import datetime

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
                    records.append({"ds": pd.to_datetime(item['deal_bas_r'].replace(',', ''), errors='coerce', format='%Y-%m-%d'),
                                     "y": float(item['deal_bas_r'].replace(',', '')),
                                     "date": current})
                    break
        except Exception as e:
            print(f"[WARN] Failed to fetch for {formatted}: {e}")
        current += timedelta(days=1)

    return pd.DataFrame([{"ds": r["date"], "y": r["y"]} for r in records if r["y"] is not None])

fetch_start = pd.to_datetime("2025-05-03")
fetch_end = pd.to_datetime(date.today())
api_df = fetch_korea_exim_rates(fetch_start, fetch_end, api_key)

# 데이터 병합 및 중복 제거
df = pd.concat([df, api_df]).drop_duplicates(subset="ds").sort_values("ds")
model = Prophet()
model.fit(df)


# 마지막 날짜 추출
latest_date = df['ds'].max()
print(f"[LOG] Latest date in data: {latest_date}")

# Streamlit UI 시작
st.markdown("""
    <h1 style='text-align: center; color: #1e1e1e; font-size: 3em; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
        💱 AI FxSense(AI환율예측시스템)<br>
        <span style='font-size: 0.5em;'>SWitch Money 팀 — 서울교육대학교 소프트웨어영재교육원</span><br>
        <span style='font-size: 0.5em;'>김서진, 김우현, 박재민</span>
    </h1>
""", unsafe_allow_html=True)

# 초기화 버튼

st.markdown("""
<div style='text-align: center;'>
    
</div>
""", unsafe_allow_html=True)

if st.button("초기화"):
    st.session_state.clear()
    st.rerun()
st.markdown("""
<div style='text-align: center;'>
    <p style='font-size: 1.2em;'>기사 내용을 입력하세요:</p>
</div>
""", unsafe_allow_html=True)
article_text = st.text_area("기사 입력", label_visibility="collapsed", height=150)

from datetime import date

today_input = date.today()
days = st.number_input("그래프에 표시할 미래 일수 (예: 30)", min_value=1, max_value=3650, value=30, step=1)
sensitivity = st.slider("감성 보정 강도 (% 조정폭)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

# 감성 분석 함수 (OpenAI >=1.0.0 호환)
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
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        return f"[ERROR] 감성 분석 실패: {e}"

if st.button("환율 예측 실행"):
    try:
        if 'predicted' not in st.session_state:
            st.session_state['predicted'] = True
            print("[LOG] Creating future dataframe...")
            delta_days = (today_input + timedelta(days=1) - latest_date.date()).days
            periods_needed = max(delta_days, days)
            future = model.make_future_dataframe(periods=periods_needed)
            print("[LOG] Predicting future...")
            forecast = model.predict(future)

            # 내일 환율 예측
            next_day = today_input + timedelta(days=1)
            next_prediction = forecast[forecast['ds'] == pd.to_datetime(next_day)]['yhat'].values

            # 감성 분석 결과
            adjustment_factor = 0.0
            if article_text.strip():
                st.subheader("AI 감성 분석 결과")
                sentiment_result = analyze_article_sentiment(article_text)
                st.write(sentiment_result)

                if "긍정" in sentiment_result:
                    adjustment_factor = -sensitivity / 100
                    st.success(f"📉 긍정적 기사 → 원화 강세 → 환율 하락 (-{sensitivity:.1f}%)")
                elif "부정" in sentiment_result:
                    adjustment_factor = sensitivity / 100
                    st.error(f"📈 부정적 기사 → 원화 약세 → 환율 상승 (+{sensitivity:.1f}%)")
                elif "중립" in sentiment_result:
                    st.info("중립적 기사 → 환율 영향 없음")

            if len(next_prediction) == 0:
                st.warning(f"{next_day} 환율 예측 없음")
                print(f"[WARN] No prediction available for {next_day}")
            else:
                adjusted = next_prediction[0] * (1 + adjustment_factor)
                st.success(f"{next_day} 환율 예측: {next_prediction[0]:.2f} ₩")
                if adjustment_factor != 0.0:
                    st.info(f"감성 분석 반영 환율: {adjusted:.2f} ₩")

            # 그래프 그리기
            print("[LOG] Plotting forecast...")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['ds'], df['y'], label='실제 환율', color='black')
            ax.plot(forecast['ds'], forecast['yhat'], label='예측 환율', color='blue')
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                            color='lightblue', alpha=0.4)
            ax.set_xlim([latest_date - timedelta(days=100), forecast['ds'].max()])
            ax.set_ylim(0, 2000)
            ax.set_title('환율 예측 그래프')
            ax.set_xlabel('날짜')
            ax.set_ylabel('환율 (₩)')
            ax.legend()
            ax.axvline(x=latest_date, color='gray', linestyle='--', label='예측 시작')
            st.pyplot(fig)

            # 예측 결과 다운로드
            st.download_button(
                label="예측 결과 다운로드 (CSV)",
                data=forecast.to_csv(index=False).encode('utf-8-sig'),
                file_name='환율_예측.csv',
                mime='text/csv'
            )
        else:
            print("[INFO] Prediction already run this session")

    except Exception as e:
        st.error(f"오류 발생: {e}")
        print(f"[ERROR] {e}")

    except Exception as e:
        st.error(f"오류 발생: {e}")
        print(f"[ERROR] {e}")

st.markdown("""
<div style='text-align: center; margin-top: 2em;'>
    <a href='https://ember-ski-cec.notion.site/AI-SWitch-Money-204d0abe582680349dd5e39320e73eaf?source=copy_link'
       target='_blank'
       style='text-decoration: none; font-size: 1.1em; color: #2d79c7; font-weight: 600;'>
        📘 자세한 설명 보러가기 (Notion)
    </a>
</div>
""", unsafe_allow_html=True)
