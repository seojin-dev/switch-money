# 💱 AI 환율 예측 시스템

AI를 이용해 뉴스 기사 내용을 분석하고, 미래 환율을 예측하는 웹 기반 시스템입니다.  
SWitch Money 팀 — 서울교육대학교 소프트웨어영재교육원  
개발자: 김서진, 김우현, 박재민

---

## 🧩 주요 기능

- ✅ 과거 데이터를 기반으로 미래 환율 예측 (Prophet 사용)
- ✅ 뉴스 기사 감성 분석 (GPT-3.5 활용)
- ✅ 감성에 따라 예측값 보정
- ✅ 인터랙티브 그래프 시각화 및 CSV 다운로드
- ✅ 실시간 환율 보완 (한국수출입은행 API 연동)

---

## 🖥 실행 방법

1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

2. API 키 설정
   `.env` 파일을 생성하고, 그 파일 안에 다음 두 곳을 본인의 키로 수정

```python
openai.api_key = "your-openai-api-key"
api_key = "your-koreaexim-api-key"
```

3. 실행

```bash
streamlit run app.py
```

웹 브라우저가 열리며 `http://localhost:8501` 에서 앱 사용 가능

---

## 📊 예시 화면

- 기사 입력 → AI 감성 분석 결과 → 환율 예측 결과 → 그래프 출력
- 예측값은 `.csv`로 저장 가능

---

## 📘 자세한 설명

👉 [Notion 페이지 보기](https://ember-ski-cec.notion.site/AI-SWitch-Money-204d0abe582680349dd5e39320e73eaf?source=copy_link)

---

## 📜 라이선스

본 프로젝트는 교육 목적으로 개발되었으며, 자유롭게 참고 및 활용 가능합니다.
