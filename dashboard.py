import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Coffee Sales Dashboard", layout="wide")

# Load data
df = pd.read_csv("coffee_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])
df['card'] = df['card'].fillna('CashUser')
df['month'] = df['date'].dt.strftime('%Y-%m')
df['weekday'] = df['date'].dt.day_name()
df['hour'] = df['datetime'].dt.hour

# ML model
features = ['coffee_name', 'cash_type', 'hour', 'weekday']
target = 'money'
X = df[features]
y = df[target]
X_encoded = pd.get_dummies(X, drop_first=True)
model = LinearRegression()
model.fit(X_encoded, y)

# Sidebar filters
st.sidebar.header("🔎 Filter Data")
selected_coffee = st.sidebar.multiselect("Coffee Type", df['coffee_name'].unique(), default=df['coffee_name'].unique())
selected_day = st.sidebar.multiselect("Weekday", df['weekday'].unique(), default=df['weekday'].unique())
selected_hour = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# Filter data
filtered_df = df[
    (df['coffee_name'].isin(selected_coffee)) &
    (df['weekday'].isin(selected_day)) &
    (df['hour'] >= selected_hour[0]) &
    (df['hour'] <= selected_hour[1])
]

# KPI section
total_revenue = filtered_df['money'].sum()
average_sale = filtered_df['money'].mean()
top_product = filtered_df['coffee_name'].value_counts().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"₹{total_revenue:,.2f}")
col2.metric("📊 Average Sale", f"₹{average_sale:,.2f}")
col3.metric("🏆 Top Product", top_product)

# Title
st.title("☕ Coffee Sales Dashboard")

# Prediction section
st.markdown("### 🔮 Predict Sale Amount")

coffee = st.selectbox("Coffee Type", df['coffee_name'].unique())
payment = st.selectbox("Payment Method", df['cash_type'].unique())
hour = st.slider("Hour", 0, 23, 10)
day = st.selectbox("Weekday", df['weekday'].unique())

input_df = pd.DataFrame([[coffee, payment, hour, day]], columns=features)
input_encoded = pd.get_dummies(input_df, drop_first=True)
missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

prediction = model.predict(input_encoded)[0]
st.success(f"Predicted Sale Amount: ₹{prediction:.2f}")

# Visuals
st.markdown("### 📈 Sales by Coffee Type")
chart_data = filtered_df.groupby('coffee_name')['money'].sum().sort_values()
st.bar_chart(chart_data)

st.markdown("### 🕒 Sales by Hour")
hour_data = filtered_df.groupby('hour')['money'].sum()
st.line_chart(hour_data)