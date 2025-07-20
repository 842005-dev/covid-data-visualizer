import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="COVID-19 India Dashboard", layout="wide")

st.title("🦠 COVID-19 India Dashboard")
st.markdown("Visual analysis using real-time or saved COVID-19 data.")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    df = pd.read_csv(url)
    df_india = df[df["Country"] == "India"].copy()
    df_india["Date"] = pd.to_datetime(df_india["Date"])
    df_india["Active"] = df_india["Confirmed"] - df_india["Recovered"] - df_india["Deaths"]
    return df_india

df = load_data()

# Show summary
st.write("## 🇮🇳 India's COVID-19 Case Summary")
st.dataframe(df.tail())

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Confirmed"], label="Confirmed")
ax.plot(df["Date"], df["Recovered"], label="Recovered")
ax.plot(df["Date"], df["Deaths"], label="Deaths")
ax.plot(df["Date"], df["Active"], label="Active")
ax.set_title("COVID-19 Trend in India")
ax.legend()
st.pyplot(fig)
