# ☕ Coffee Sales Prediction Dashboard

An interactive Streamlit dashboard for visualizing coffee sales and predicting revenue using a machine learning model (Linear Regression).

## 📌 Features

- Predict sales revenue based on coffee type, payment method, hour, and weekday
- Interactive dropdowns and sliders
- Sales visualization by coffee type
- Clean and responsive UI using Streamlit

## 🚀 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/coffee-sales-dashboard.git
   cd coffee-sales-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## 📁 Dataset

Make sure `coffee_sales.csv` is in the root directory.

## 🧠 ML Model

- Model: Linear Regression
- Features: Coffee Type, Payment Type, Hour, Weekday
- Target: Sales Revenue (`money` column)

## ✨ Demo

You can deploy it for free on [Streamlit Cloud](https://streamlit.io/cloud).

---

Made with ❤️ using Python, Pandas, Scikit-learn, and Streamlit.