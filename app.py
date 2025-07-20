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
    df = df[df["Country"] == "India"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Active"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
    return df

data = load_data()

# Display data
st.subheader("📊 Latest 5 Days")
st.dataframe(data.tail())

# Graph
st.subheader("📈 Trend Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data["Date"], data["Confirmed"], label="Confirmed", color="orange")
ax.plot(data["Date"], data["Recovered"], label="Recovered", color="green")
ax.plot(data["Date"], data["Deaths"], label="Deaths", color="red")
ax.plot(data["Date"], data["Active"], label="Active", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Cases")
ax.set_title("COVID-19 India Stats")
ax.legend()
st.pyplot(fig)
