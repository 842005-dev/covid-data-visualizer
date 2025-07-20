import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 India Dashboard", layout="wide")
st.title("🦠 COVID-19 India Dashboard")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    df = pd.read_csv(url)
    india = df[df["Country"] == "India"].copy()
    india["Date"] = pd.to_datetime(india["Date"])
    india["Active"] = india["Confirmed"] - india["Deaths"] - india["Recovered"]
    return india

df = load_data()

# Show recent data
st.subheader("📊 Latest 5 Records")
st.dataframe(df.tail())

# Line chart
st.subheader("📈 COVID-19 Trend in India")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Confirmed"], label="Confirmed", color="orange")
ax.plot(df["Date"], df["Recovered"], label="Recovered", color="green")
ax.plot(df["Date"], df["Deaths"], label="Deaths", color="red")
ax.plot(df["Date"], df["Active"], label="Active", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Cases")
ax.set_title("COVID-19 Daily Cases in India")
ax.legend()
st.pyplot(fig)
