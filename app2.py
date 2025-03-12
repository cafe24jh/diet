import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text
import openai
import re
import os  # 환경 변수 사용을 위해 추가

# -----------------------
# 1) OPENAI API 설정
# -----------------------
try:
    # 환경 변수에서 OPENAI_API_KEY 불러오기
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    openai.api_key = OPENAI_API_KEY
except Exception as e:
    st.error(f"GPT API 초기화 오류: {e}")

def analyze_meal_with_gpt(meal_info):
    """OpenAI ChatCompletion을 이용해 식단을 분석하는 함수"""
    if not openai.api_key:
        return "API 키가 설정되지 않아 식단 분석을 수행할 수 없습니다."
        
    try:
        system_prompt = """You are a professional nutritionist and diet expert. Analyze the given meal information and provide insights in Korean.
Provide your analysis in the following format:

🥗 전반적인 영양 균형
• Point 1
• Point 2

📊 칼로리 추정
• Point 1
• Point 2

💡 개선점 제안
• Point 1
• Point 2

🍳 다음 식사 추천
• Point 1
• Point 2

마지막 문장은 별도의 줄에 작성"""

        user_content = f"""다음 식단 정보를 분석해주세요:

아침: {meal_info['breakfast']}
점심: {meal_info['lunch']}
저녁: {meal_info['dinner']}

다음 항목들을 분석해주세요:
1. 전반적인 영양 균형
2. 칼로리 추정
3. 개선점 제안
4. 다음 식사 추천"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"GPT API 오류: {e}")
        return None

def display_analysis_result(analysis_result):
    """AI 분석 결과를 Streamlit에 표시"""
    if not analysis_result:
        st.error("식단 분석 중 오류가 발생했습니다.")
        return

    st.subheader("🤖 AI 식단 분석 결과")
    
    # 줄바꿈으로 섹션 분리
    sections = [section.strip() for section in analysis_result.split('\n\n') if section.strip()]
    
    for section in sections:
        # 이모지로 시작하는 섹션 처리
        if any(section.startswith(emoji) for emoji in ['🥗', '📊', '💡', '🍳']):
            # 제목과 내용 분리
            title, *content = section.split('\n')
            with st.expander(title, expanded=True):
                for line in content:
                    if line.strip():
                        # 불릿 포인트 처리
                        point = line.strip().lstrip('•').strip()
                        st.markdown(f"- {point}")
        else:
            # 일반 텍스트 (마지막 문장) 처리
            st.info(section)

def analyze_meal(meal_info):
    """식단 분석을 수행하고 결과를 표시하는 함수"""
    if meal_info['breakfast'] == "-" and meal_info['lunch'] == "-" and meal_info['dinner'] == "-":
        st.warning("분석할 식단 데이터가 없습니다.")
        return

    with st.spinner("AI가 식단을 분석중입니다..."):
        analysis_result = analyze_meal_with_gpt(meal_info)
        display_analysis_result(analysis_result)

# -----------------------
# 2) 페이지 기본 설정
# -----------------------
st.set_page_config(
    page_title="체중 및 식단 관리",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("체중 및 식단 관리 앱")

# -----------------------
# 3) MySQL 연결
# -----------------------
def create_connection():
    """환경 변수로부터 DB 접속 정보를 불러와 엔진을 생성"""
    try:
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASS")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        
        if not (db_user and db_pass and db_host and db_name):
            raise ValueError("DB_USER, DB_PASS, DB_HOST, DB_NAME 중 하나가 설정되지 않았습니다.")
        
        connection_string = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"MySQL 연결 오류: {e}")
        return None

def test_connection():
    """DB 연결 테스트"""
    try:
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASS")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        
        connection_string = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            return True
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {str(e)}")
        return False

# -----------------------
# 4) 데이터 불러오기 및 저장
# -----------------------
def create_empty_dataframe():
    """빈 체중 데이터프레임 생성"""
    weight_df = pd.DataFrame(columns=['date', 'weight'])
    weight_df['date'] = pd.to_datetime(weight_df['date'])
    return weight_df

def create_empty_meal_dataframe():
    """빈 식단 데이터프레임 생성"""
    meal_df = pd.DataFrame(columns=['date', 'breakfast', 'lunch', 'dinner'])
    meal_df['date'] = pd.to_datetime(meal_df['date'])
    return meal_df

@st.cache_data(ttl=300)  # 5분마다 갱신
def load_data():
    """DB에서 체중, 식단 데이터를 불러오는 함수"""
    engine = create_connection()
    if engine is None:
        st.error("데이터베이스 연결 실패")
        return create_empty_dataframe(), create_empty_meal_dataframe()
    
    try:
        # 체중 데이터
        weight_query = "SELECT date, weight FROM diet ORDER BY date"
        weight_df = pd.read_sql(weight_query, engine)
        if not weight_df.empty:
            weight_df['date'] = pd.to_datetime(weight_df['date'])
        
        # 식단 데이터
        meal_query = "SELECT date, breakfast, lunch, dinner FROM diet_meal ORDER BY date"
        try:
            meal_df = pd.read_sql(meal_query, engine)
            if not meal_df.empty:
                meal_df['date'] = pd.to_datetime(meal_df['date'])
        except:
            st.warning("식단 테이블에서 데이터를 불러올 수 없습니다.")
            meal_df = create_empty_meal_dataframe()
        
        return weight_df, meal_df
    
    except Exception as e:
        st.error(f"데이터 불러오기 오류: {e}")
        return create_empty_dataframe(), create_empty_meal_dataframe()

def save_weight_data(date, weight):
    """체중 데이터를 INSERT/UPDATE"""
    engine = create_connection()
    if engine is None:
        st.error("MySQL 연결 실패로 데이터를 저장할 수 없습니다.")
        return False
    
    try:
        with engine.connect() as conn:
            check_query = text("SELECT * FROM diet WHERE date = :date")
            result = conn.execute(check_query, {"date": date}).fetchone()
            
            if result:
                update_query = text("UPDATE diet SET weight = :weight WHERE date = :date")
                conn.execute(update_query, {"weight": weight, "date": date})
            else:
                insert_query = text("INSERT INTO diet (date, weight) VALUES (:date, :weight)")
                conn.execute(insert_query, {"date": date, "weight": weight})
            
            conn.commit()
            return True
    
    except Exception as e:
        st.error(f"데이터 저장 오류: {e}")
        return False

def save_meal_data(date, breakfast, lunch, dinner):
    """식단 데이터를 INSERT/UPDATE"""
    engine = create_connection()
    if engine is None:
        st.error("MySQL 연결 실패로 데이터를 저장할 수 없습니다.")
        return False
    
    try:
        with engine.connect() as conn:
            check_query = text("SELECT * FROM diet_meal WHERE date = :date")
            result = conn.execute(check_query, {"date": date}).fetchone()
            
            if result:
                update_query = text("""
                    UPDATE diet_meal 
                    SET breakfast = :breakfast, lunch = :lunch, dinner = :dinner 
                    WHERE date = :date
                """)
                conn.execute(
                    update_query,
                    {
                        "breakfast": breakfast,
                        "lunch": lunch,
                        "dinner": dinner,
                        "date": date
                    }
                )
            else:
                insert_query = text("""
                    INSERT INTO diet_meal (date, breakfast, lunch, dinner) 
                    VALUES (:date, :breakfast, :lunch, :dinner)
                """)
                conn.execute(
                    insert_query,
                    {
                        "date": date,
                        "breakfast": breakfast,
                        "lunch": lunch,
                        "dinner": dinner
                    }
                )
            
            conn.commit()
            return True
    
    except Exception as e:
        st.error(f"데이터 저장 오류: {e}")
        return False

def get_meal_info(date, meal_df):
    """특정 날짜의 식단 정보를 DB에서 가져오기"""
    try:
        engine = create_connection()
        if engine is None:
            return {'breakfast': "-", 'lunch': "-", 'dinner': "-"}
        
        with engine.connect() as conn:
            query = text("SELECT breakfast, lunch, dinner FROM diet_meal WHERE date = :date")
            result = conn.execute(query, {"date": date.strftime('%Y-%m-%d')}).fetchone()
            
            if result:
                return {
                    'breakfast': result[0] if result[0] else "-",
                    'lunch': result[1] if result[1] else "-",
                    'dinner': result[2] if result[2] else "-"
                }
            else:
                return {'breakfast': "-", 'lunch': "-", 'dinner': "-"}
    except Exception as e:
        st.error(f"식단 정보 조회 오류: {e}")
        return {'breakfast': "-", 'lunch': "-", 'dinner': "-"}

def calculate_predicted_weight(weight_df, target_date):
    """선형 회귀를 이용해 미래 체중을 예측"""
    if weight_df.empty or len(weight_df) < 2:
        return None
    
    x = np.arange(len(weight_df))
    y = weight_df['weight'].values
    
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    
    latest_date = weight_df['date'].iloc[-1]
    target_date_pd = pd.Timestamp(target_date)
    days_diff = (target_date_pd - latest_date).days
    
    pred_idx = n - 1 + days_diff
    predicted_weight = intercept + slope * pred_idx
    
    return predicted_weight

# -----------------------
# 5) 데이터 불러오기
# -----------------------
weight_df, meal_df = load_data()

# -----------------------
# 6) 사이드바 메뉴
# -----------------------
menu = st.sidebar.radio(
    "메뉴 선택",
    ["📊 대시보드", "⚖️ 체중 / 식단 기록", "📈 데이터 분석", "🔄 데이터 관리"]
)

# -----------------------
# 6-1) 대시보드 페이지
# -----------------------
if menu == "📊 대시보드":
    st.header("체중 및 식단 대시보드")
    
    if not weight_df.empty:
        # 주요 통계
        latest_weight = weight_df['weight'].iloc[-1]
        start_weight = weight_df['weight'].iloc[0]
        weight_loss = start_weight - latest_weight
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("현재 체중", f"{latest_weight:.1f} kg", f"{-weight_loss:.1f} kg")
        with col2:
            days = (weight_df['date'].iloc[-1] - weight_df['date'].iloc[0]).days
            st.metric("감량 기간", f"{days}일")
        with col3:
            avg_loss_per_day = weight_loss / days if days > 0 else 0
            st.metric("일일 평균 감량", f"{avg_loss_per_day:.2f} kg/일")
        
        # 체중 그래프
        st.subheader("체중 변화 그래프")
        fig = px.line(weight_df, x='date', y='weight', markers=True)
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="날짜",
            yaxis_title="체중 (kg)"
        )
        
        # 추세선 추가
        x = np.arange(len(weight_df))
        y = weight_df['weight']
        
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n
        trend_y = intercept + slope * x
        
        fig.add_trace(
            go.Scatter(
                x=weight_df['date'],
                y=trend_y,
                mode='lines',
                name='추세선',
                line=dict(color='orange', dash='dash')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 체중 예측
        st.subheader("체중 예측")
        target_date = st.date_input("예측 날짜", datetime.now() + timedelta(days=30), key="target_date")
        if st.button("예측하기", key="predict_weight"):
            predicted_weight = calculate_predicted_weight(weight_df, target_date)
            if predicted_weight is not None:
                st.success(f"예상 체중: **{predicted_weight:.1f} kg**")
                weight_diff = predicted_weight - latest_weight
                if weight_diff < 0:
                    st.info(f"현재보다 **{abs(weight_diff):.1f} kg** 감량")
                elif weight_diff > 0:
                    st.warning(f"현재보다 **{weight_diff:.1f} kg** 증가")
        
        # 목표 달성 예측
        st.subheader("목표 달성 예측")
        target_goal = st.number_input("목표 체중 (kg)", min_value=30.0, max_value=200.0, value=latest_weight-5.0, step=1.0)
        if st.button("계산하기", key="predict_goal"):
            if weight_loss == 0 or days == 0:
                st.warning("추세 계산 불가")
            else:
                daily_change = weight_loss / days
                weight_to_lose = latest_weight - target_goal
                
                if daily_change == 0:
                    st.warning("체중 변화 없음")
                elif (target_goal < latest_weight and daily_change > 0) or (target_goal > latest_weight and daily_change < 0):
                    days_needed = abs(weight_to_lose / daily_change)
                    goal_date = weight_df['date'].iloc[-1] + timedelta(days=days_needed)
                    st.success(f"달성 예상일: **{goal_date.strftime('%Y년 %m월 %d일')}**")
                    st.info(f"소요 기간: 약 {days_needed:.0f}일")
                else:
                    st.warning("현재 추세로는 달성 불가")
        
        st.markdown("---")
        st.subheader("식단 분석")
        analysis_date = st.date_input("날짜 선택", datetime.now(), key="analysis_date")
        meal_info = get_meal_info(analysis_date, meal_df)
        
        if meal_info['breakfast'] != "-" or meal_info['lunch'] != "-" or meal_info['dinner'] != "-":
            meal_text = f"🍳 아침: {meal_info['breakfast']}\n🍲 점심: {meal_info['lunch']}\n🍽️ 저녁: {meal_info['dinner']}"
            st.text_area("오늘의 식단", meal_text, height=150)
            if st.button("AI 분석", key="analyze_meal"):
                analyze_meal(meal_info)
        else:
            st.info("식단 데이터가 없습니다.")
    else:
        st.error("체중 데이터가 없습니다.")

# -----------------------
# 6-2) 체중/식단 기록 페이지
# -----------------------
elif menu == "⚖️ 체중 / 식단 기록":
    st.header("체중 / 식단 기록")
    
    if not weight_df.empty:
        st.subheader("기록 조회")
        selected_date = st.date_input("날짜 선택", datetime.now(), key="view_date")
        
        selected_weight_df = weight_df[weight_df['date'].dt.strftime('%Y-%m-%d') == selected_date.strftime('%Y-%m-%d')]
        meal_info = get_meal_info(selected_date, meal_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("체중 정보")
            if not selected_weight_df.empty:
                weight_val = selected_weight_df.iloc[0]['weight']
                st.info(f"**{selected_date.strftime('%Y년 %m월 %d일')}**의 체중: **{weight_val:.1f} kg**")
                
                # 이전 날짜와의 변화량
                prev_date = selected_date - timedelta(days=1)
                prev_weight_df = weight_df[weight_df['date'].dt.strftime('%Y-%m-%d') == prev_date.strftime('%Y-%m-%d')]
                if not prev_weight_df.empty:
                    prev_weight = prev_weight_df.iloc[0]['weight']
                    change = weight_val - prev_weight
                    if change > 0:
                        st.warning(f"전일 대비: **+{change:.1f} kg**")
                    elif change < 0:
                        st.success(f"전일 대비: **{change:.1f} kg**")
                    else:
                        st.info("전일과 동일")
            else:
                st.warning("해당 날짜의 체중 기록이 없습니다.")
        
        with col2:
            st.subheader("식단 정보")
            meal_col1, meal_col2, meal_col3 = st.columns(3)
            with meal_col1:
                st.markdown("#### 🍳 아침")
                st.write(meal_info['breakfast'])
            with meal_col2:
                st.markdown("#### 🍲 점심")
                st.write(meal_info['lunch'])
            with meal_col3:
                st.markdown("#### 🍽️ 저녁")
                st.write(meal_info['dinner'])
        
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 체중 입력 폼
        with st.form("weight_form"):
            st.subheader("새 체중 기록 추가")
            
            date = st.date_input("날짜", datetime.now(), key="weight_date")
            weight = st.number_input("체중 (kg)", 30.0, 200.0, 70.0, 0.1)
            
            submit_weight = st.form_submit_button("체중 기록 추가")
            if submit_weight:
                date_str = date.strftime('%Y-%m-%d')
                if save_weight_data(date_str, weight):
                    st.success(f"{date_str}의 체중 {weight}kg이 저장되었습니다.")
                    st.cache_data.clear()
                    weight_df, meal_df = load_data()
                    st.rerun()
                else:
                    st.error("데이터 저장에 실패했습니다.")
    
    with col2:
        # 식단 입력 폼
        with st.form("meal_form"):
            st.subheader("새 식단 기록 추가")
            
            meal_date = st.date_input("날짜", datetime.now(), key="meal_date")
            breakfast = st.text_area("아침 식단", height=100)
            lunch = st.text_area("점심 식단", height=100)
            dinner = st.text_area("저녁 식단", height=100)
            
            submit_meal = st.form_submit_button("식단 기록 추가")
            if submit_meal:
                date_str = meal_date.strftime('%Y-%m-%d')
                if save_meal_data(date_str, breakfast, lunch, dinner):
                    st.success(f"{date_str}의 식단이 저장되었습니다.")
                    st.cache_data.clear()
                    weight_df, meal_df = load_data()
                    st.rerun()
                else:
                    st.error("데이터 저장에 실패했습니다.")

# -----------------------
# 6-3) 데이터 분석 페이지
# -----------------------
elif menu == "📈 데이터 분석":
    st.header("데이터 분석")
    
    if not weight_df.empty:
        period = st.radio("분석 기간", ["전체", "최근 7일", "최근 30일"])
        
        if period == "전체":
            analysis_df = weight_df.copy()
        elif period == "최근 7일":
            end_date = weight_df['date'].max()
            start_date = end_date - timedelta(days=7)
            analysis_df = weight_df[weight_df['date'] >= start_date].copy()
        else:  # 최근 30일
            end_date = weight_df['date'].max()
            start_date = end_date - timedelta(days=30)
            analysis_df = weight_df[weight_df['date'] >= start_date].copy()
        
        if not analysis_df.empty:
            st.subheader("통계 요약")
            start_w = analysis_df['weight'].iloc[0]
            end_w = analysis_df['weight'].iloc[-1]
            min_w = analysis_df['weight'].min()
            max_w = analysis_df['weight'].max()
            avg_w = analysis_df['weight'].mean()
            total_change = end_w - start_w
            days = (analysis_df['date'].iloc[-1] - analysis_df['date'].iloc[0]).days
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 체중", f"{avg_w:.1f} kg")
            with col2:
                st.metric("최소 체중", f"{min_w:.1f} kg")
            with col3:
                st.metric("최대 체중", f"{max_w:.1f} kg")
            
            st.metric("총 변화량", f"{total_change:.1f} kg")
            
            if days > 0:
                st.metric("일일 평균 변화량", f"{total_change/days:.2f} kg/일")

            # 요일별 분석 (데이터가 7개 이상 있어야 의미가 있음)
            if len(analysis_df) > 7:
                st.subheader("요일별 분석")
                analysis_df['day_of_week'] = analysis_df['date'].dt.day_name()
                weekday_weight = analysis_df.groupby('day_of_week')['weight'].mean().reset_index()
                
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_names = {
                    'Monday': '월요일', 'Tuesday': '화요일', 'Wednesday': '수요일',
                    'Thursday': '목요일', 'Friday': '금요일', 'Saturday': '토요일', 'Sunday': '일요일'
                }
                
                weekday_weight['day_order'] = weekday_weight['day_of_week'].map(lambda x: days_order.index(x))
                weekday_weight = weekday_weight.sort_values('day_order')
                weekday_weight['day_of_week_kr'] = weekday_weight['day_of_week'].map(weekday_names)
                
                avg_weight = weekday_weight['weight'].mean()
                
                fig = px.bar(
                    weekday_weight, 
                    x='day_of_week_kr', 
                    y='weight',
                    title='요일별 평균 체중',
                    color='weight',
                    color_continuous_scale=['green', 'red'],
                    color_continuous_midpoint=avg_weight
                )
                
                y_min = weekday_weight['weight'].min() - 0.5
                y_max = weekday_weight['weight'].max() + 0.5
                fig.update_layout(yaxis_range=[y_min, y_max])
                
                fig.add_hline(
                    y=avg_weight,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="전체 평균",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 최고/최저 체중 요일 식단 패턴
                max_weight_day = weekday_weight.loc[weekday_weight['weight'].idxmax()]
                min_weight_day = weekday_weight.loc[weekday_weight['weight'].idxmin()]
                
                st.subheader("체중이 가장 높은/낮은 요일의 식단 패턴")
                
                max_day_meals = meal_df[meal_df['date'].dt.day_name() == max_weight_day['day_of_week']]
                min_day_meals = meal_df[meal_df['date'].dt.day_name() == min_weight_day['day_of_week']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📈 체중이 가장 높은 {max_weight_day['day_of_week_kr']} (평균 {max_weight_day['weight']:.1f}kg)**")
                    if not max_day_meals.empty:
                        st.markdown("🍳 **아침 식사 패턴:**")
                        breakfast_patterns = max_day_meals[['date', 'breakfast']].dropna()
                        for _, row in breakfast_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['breakfast']}")
                        
                        st.markdown("🍲 **점심 식사 패턴:**")
                        lunch_patterns = max_day_meals[['date', 'lunch']].dropna()
                        for _, row in lunch_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['lunch']}")
                        
                        st.markdown("🍽️ **저녁 식사 패턴:**")
                        dinner_patterns = max_day_meals[['date', 'dinner']].dropna()
                        for _, row in dinner_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['dinner']}")
                    else:
                        st.info("해당 요일의 식단 데이터가 없습니다.")
                
                with col2:
                    st.markdown(f"**📉 체중이 가장 낮은 {min_weight_day['day_of_week_kr']} (평균 {min_weight_day['weight']:.1f}kg)**")
                    if not min_day_meals.empty:
                        st.markdown("🍳 **아침 식사 패턴:**")
                        breakfast_patterns = min_day_meals[['date', 'breakfast']].dropna()
                        for _, row in breakfast_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['breakfast']}")
                        
                        st.markdown("🍲 **점심 식사 패턴:**")
                        lunch_patterns = min_day_meals[['date', 'lunch']].dropna()
                        for _, row in lunch_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['lunch']}")
                        
                        st.markdown("🍽️ **저녁 식사 패턴:**")
                        dinner_patterns = min_day_meals[['date', 'dinner']].dropna()
                        for _, row in dinner_patterns.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['dinner']}")
                    else:
                        st.info("해당 요일의 식단 데이터가 없습니다.")
        else:
            st.info(f"선택한 기간 ({period})에 데이터가 없습니다.")
    else:
        st.error("체중 데이터가 없습니다.")

# -----------------------
# 6-4) 데이터 관리 페이지
# -----------------------
elif menu == "🔄 데이터 관리":
    st.header("데이터 관리")
    
    st.subheader("데이터베이스 연결 상태")
    if test_connection():
        st.success("데이터베이스가 정상적으로 연결되어 있습니다.")
        if st.button("연결 테스트"):
            st.success("데이터베이스 연결 테스트 성공!")
    else:
        st.error("데이터베이스 연결에 실패했습니다.")
    
    st.subheader("데이터베이스 설정")
    with st.expander("데이터베이스 정보"):
        st.code("""
host: (DB_HOST)
database: (DB_NAME)
테이블:
  - diet: (date, weight)
  - diet_meal: (date, breakfast, lunch, dinner)
        """)
    
    st.subheader("데이터베이스 관리")
    init_col1, init_col2 = st.columns(2)
    
    with init_col1:
        if st.button("데이터 초기화 (주의: 모든 데이터 삭제)"):
            conn = create_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM diet")
                    conn.commit()
                    st.success("모든 데이터가 삭제되었습니다.")
                    st.cache_data.clear()
                    st.rerun()
                except Error as e:
                    st.error(f"데이터 초기화 오류: {e}")
                finally:
                    if conn is not None:
                        conn.dispose()
            else:
                st.error("MySQL 연결 실패로 초기화할 수 없습니다.")
    
    with init_col2:
        if st.button("데이터베이스 테이블 생성"):
            conn = create_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    create_weight_table = """
                    CREATE TABLE IF NOT EXISTS diet (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        date DATE NOT NULL,
                        weight FLOAT NOT NULL,
                        UNIQUE KEY (date)
                    )
                    """
                    cursor.execute(create_weight_table)
                    
                    create_meal_table = """
                    CREATE TABLE IF NOT EXISTS diet_meal (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        date DATE NOT NULL,
                        breakfast TEXT,
                        lunch TEXT,
                        dinner TEXT,
                        UNIQUE KEY (date)
                    )
                    """
                    cursor.execute(create_meal_table)
                    
                    conn.commit()
                    st.success("필요한 테이블이 생성되었습니다.")
                except Error as e:
                    st.error(f"테이블 생성 오류: {e}")
                finally:
                    if conn is not None:
                        conn.dispose()
            else:
                st.error("MySQL 연결 실패로 테이블을 생성할 수 없습니다.")
    
    st.subheader("데이터 백업")
    backup_col1, backup_col2 = st.columns(2)
    
    with backup_col1:
        if st.button("체중 데이터 백업"):
            if not weight_df.empty:
                csv = weight_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="체중 데이터 CSV 다운로드",
                    data=csv,
                    file_name="weight_data_backup.csv",
                    mime="text/csv"
                )
            else:
                st.warning("백업할 체중 데이터가 없습니다.")
    
    with backup_col2:
        if st.button("식단 데이터 백업"):
            if not meal_df.empty:
                csv = meal_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="식단 데이터 CSV 다운로드",
                    data=csv,
                    file_name="meal_data_backup.csv",
                    mime="text/csv"
                )
            else:
                st.warning("백업할 식단 데이터가 없습니다.")
