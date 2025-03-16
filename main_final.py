import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import io
from fpdf import FPDF

def load_data(file):
    df = pd.read_excel(file)
    return df

def generate_recommendations(df):
    top_product = df.groupby("вид продукта")["общая выручка"].sum().idxmax()
    top_revenue = df.groupby("вид продукта")["общая выручка"].sum().max()
    
    recommendations = f"Наиболее прибыльный товар: {top_product} с выручкой {top_revenue:.2f} руб.\n"
    recommendations += "Рекомендации:\n"
    recommendations += "- Рассмотрите увеличение производства данного товара.\n"
    recommendations += "- Проверьте регионы с наибольшим спросом и увеличьте поставки.\n"
    recommendations += "- Оцените ценовую политику: возможен ли небольшой рост цены без потери спроса?\n"
    
    return recommendations

def predict_sales(df):
    df["timestamp"] = df["дата"].astype(np.int64) // 10**9
    model = LinearRegression()
    model.fit(df[["timestamp"]], df["общая выручка"])
    
    future_dates = pd.date_range(df["дата"].max() + pd.Timedelta(days=1), periods=30)
    future_timestamps = future_dates.astype(np.int64) // 10**9
    predictions = model.predict(future_timestamps.values.reshape(-1, 1))
    
    future_df = pd.DataFrame({"дата": future_dates, "прогнозируемая выручка": predictions})
    return future_df

def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Анализ продаж")
    processed_data = output.getvalue()
    return processed_data

def export_to_pdf(recommendations, future_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, "Отчет по анализу продаж", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, recommendations)
    pdf.ln(10)
    
    pdf.cell(200, 10, "Прогноз продаж", ln=True, align='C')
    pdf.ln(10)
    for index, row in future_df.iterrows():
        pdf.cell(200, 10, f"{row['дата'].date()} - {row['прогнозируемая выручка']:.2f} руб.", ln=True)
    
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

def main():
    st.title("Анализ продаж")
    
    uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xls", "xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("### Загруженные данные:")
        st.dataframe(df.head())
        
        required_columns = {"дата", "объем продаж", "цена", "вид продукта", "местоположение", "покупатель"}
        if required_columns.issubset(df.columns):
            df["дата"] = pd.to_datetime(df["дата"])
            df.sort_values("дата", inplace=True)
            df["общая выручка"] = df["объем продаж"] * df["цена"]
            
            st.sidebar.header("Фильтры")
            product_filter = st.sidebar.multiselect("Выберите продукт", df["вид продукта"].unique(), default=df["вид продукта"].unique())
            location_filter = st.sidebar.multiselect("Выберите местоположение", df["местоположение"].unique(), default=df["местоположение"].unique())
            buyer_filter = st.sidebar.multiselect("Выберите тип покупателя", df["покупатель"].unique(), default=df["покупатель"].unique())
            date_range = st.sidebar.date_input("Выберите диапазон дат", [df["дата"].min(), df["дата"].max()])
            
            filtered_df = df[(df["вид продукта"].isin(product_filter)) &
                             (df["местоположение"].isin(location_filter)) &
                             (df["покупатель"].isin(buyer_filter)) &
                             (df["дата"] >= pd.to_datetime(date_range[0])) &
                             (df["дата"] <= pd.to_datetime(date_range[1]))]
            
            st.write("### Отфильтрованные данные:")
            st.dataframe(filtered_df.head())
            
            st.write("### Рекомендации")
            recommendations = generate_recommendations(filtered_df)
            st.text(recommendations)
            
            st.write("### Прогноз продаж на следующий месяц")
            future_df = predict_sales(filtered_df)
            st.dataframe(future_df)
            
            excel_data = export_to_excel(filtered_df)
            st.download_button(label="Скачать отчет в Excel", data=excel_data, file_name="отчет.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            pdf_data = export_to_pdf(recommendations, future_df)
            st.download_button(label="Скачать отчет в PDF", data=pdf_data, file_name="отчет.pdf", mime="application/pdf")
        else:
            st.error("Файл должен содержать колонки: дата, объем продаж, цена, вид продукта, местоположение, покупатель")

if __name__ == "__main__":
    main()
