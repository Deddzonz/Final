import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from io import BytesIO
from fpdf import FPDF
import xgboost as xgb
import os

def load_data(file):
    df = pd.read_excel(file)
    return df

def generate_recommendations(df):
    if "Вид продукта" not in df.columns or "Объем продаж" not in df.columns:
        return "Недостаточно данных для формирования рекомендаций."
    top_product = df.groupby("Вид продукта")["Объем продаж"].sum().idxmax()
    return f"Рекомендуется увеличить производство {top_product}, так как это самый продаваемый товар."

def predict_sales(df):
    if "Дата" not in df.columns or "Объем продаж" not in df.columns:
        return pd.DataFrame(columns=["Дата", "Прогноз продаж"])
    df["Дата"] = pd.to_datetime(df["Дата"], errors='coerce')
    df.dropna(subset=["Дата"], inplace=True)
    df.sort_values("Дата", inplace=True)
    df["Дней с начала"] = (df["Дата"] - df["Дата"].min()).dt.days
    X = df[["Дней с начала"]]
    y = df["Объем продаж"]
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)
    future_days = np.array([df["Дней с начала"].max() + i for i in range(1, 31)]).reshape(-1, 1)
    future_sales = model.predict(future_days)
    return pd.DataFrame({"Дата": pd.date_range(df["Дата"].max() + pd.Timedelta(days=1), periods=30), "Прогноз продаж": future_sales})

def plot_sales(df):
    if "Дата" not in df.columns or "Объем продаж" not in df.columns or "Вид продукта" not in df.columns:
        return
    unique_products = df["Вид продукта"].unique()
    for product in unique_products:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df[df["Вид продукта"] == product], x="Дата", y="Объем продаж", ax=ax, marker='o', linestyle='-', color='b')
        ax.set_title(f"Продажи для {product}")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Объем продаж")
        ax.grid(True)
        st.pyplot(fig)

def plot_price_trend(df):
    if "Дата" not in df.columns or "Цена" not in df.columns or "Вид продукта" not in df.columns:
        return
    unique_products = df["Вид продукта"].unique()
    for product in unique_products:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df[df["Вид продукта"] == product], x="Дата", y="Цена", ax=ax, color='teal')
        ax.set_title(f"Тренд цен для {product}")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Цена")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def plot_map(df):
    if "Широта" in df.columns and "Долгота" in df.columns:
        fig = px.scatter_geo(df, lat="Широта", lon="Долгота", size="Объем продаж", hover_name="Местоположение")
        st.plotly_chart(fig)

def export_to_pdf(recommendations, future_df):
    pdf = FPDF()
    pdf.add_page()
    font_path = "DejaVuSansCondensed.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Отчет по продажам", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, recommendations)
    pdf.ln(10)
    pdf.cell(200, 10, "Прогноз на 30 дней:", ln=True)
    for _, row in future_df.iterrows():
        pdf.cell(200, 10, f"{row['Дата'].strftime('%Y-%m-%d')} - {round(row['Прогноз продаж'], 2)}", ln=True)
    return pdf.output(dest='S').encode('latin1')

def main():
    st.title("Sales-Smart - Аналитика продаж")
    uploaded_file = st.file_uploader("Загрузите Excel-файл с данными", type=["xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("### Загруженные данные:", df.head())
        recommendations = generate_recommendations(df)
        st.write("### Рекомендации:", recommendations)
        future_df = predict_sales(df)
        if not future_df.empty:
            st.write("### Прогноз продаж на 30 дней:", future_df)
        plot_sales(df)
        plot_price_trend(df)
        plot_map(df)
        pdf_data = export_to_pdf(recommendations, future_df)
        st.download_button("Скачать отчет в PDF", pdf_data, file_name="sales_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()

