import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from io import BytesIO
from fpdf import FPDF
import xgboost as xgb

# Загрузка данных
def load_data(file):
    df = pd.read_excel(file)
    return df

# Генерация рекомендаций
def generate_recommendations(df):
    if "Вид продукта" not in df.columns or "Объем продаж" not in df.columns:
        return "Недостаточно данных для формирования рекомендаций."
    top_product = df.groupby("Вид продукта")["Объем продаж"].sum().idxmax()
    recommendations = f"Рекомендуется увеличить производство {top_product}, так как это самый продаваемый товар."
    return recommendations

# Прогнозирование продаж
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
    future_df = pd.DataFrame({"Дата": pd.date_range(df["Дата"].max() + pd.Timedelta(days=1), periods=30), "Прогноз продаж": future_sales})
    return future_df

# Визуализация продаж
def plot_sales(df):
    if "Дата" not in df.columns or "Объем продаж" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="Дата", y="Объем продаж", hue="Вид продукта", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Визуализация цен
def plot_price_trend(df):
    if "Дата" not in df.columns or "Цена" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Дата", y="Цена", hue="Вид продукта", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Географическая карта
def plot_map(df):
    if "Широта" in df.columns and "Долгота" in df.columns:
        fig = px.scatter_geo(df, lat="Широта", lon="Долгота", size="Объем продаж", hover_name="Местоположение")
        st.plotly_chart(fig)

# Экспорт отчета в PDF
def export_to_pdf(recommendations, future_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, "Отчет по продажам", ln=True, align='C')
    pdf.multi_cell(0, 10, recommendations)
    pdf.cell(200, 10, "Прогноз на 30 дней:", ln=True)
    for i, row in future_df.iterrows():
        pdf.cell(200, 10, f"{row['Дата'].date()} - {round(row['Прогноз продаж'], 2)}", ln=True)
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()

# Основная функция
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
