import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier

# Load data & model
df = pd.read_csv("enriched_spacex_launches.csv")
model = joblib.load("models/launch_success.pkl")

st.set_page_config(page_title="🚀 SpaceX Launch Dashboard", layout="wide")

st.title("🚀 SpaceX Launch Analysis & Prediction Platform")

# ========== Sidebar Filters ==========
st.sidebar.header("📊 Filter Historical Launches")

years = sorted(pd.to_datetime(df['date_utc']).dt.year.unique())
selected_year = st.sidebar.selectbox("Select Year", options=years)

sites = df['launchpad'].dropna().unique()
selected_site = st.sidebar.selectbox("Select Launch Site", options=sites)

# Filter data
df['year'] = pd.to_datetime(df['date_utc']).dt.year
filtered_df = df[(df['year'] == selected_year) & (df['launchpad'] == selected_site)]

st.subheader(f"📅 Launches in {selected_year} at {selected_site}")
st.write(f"Total Launches: {len(filtered_df)}")
st.dataframe(filtered_df[['name', 'date_utc', 'success', 'rocket', 'temperature', 'humidity', 'wind_speed']])

# ========== Geospatial Map ==========
st.subheader("🗺️ Launch Sites Map")

map_df = df[['launchpad', 'date_utc', 'success', 'lat', 'lon']].dropna().copy()

m = folium.Map(location=[28.5, -80.6], zoom_start=3)

for _, row in map_df.iterrows():
    color = 'green' if row['success'] == 1 else 'red'
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        popup=f"{row['launchpad']}<br>{row['date_utc']}<br>{'Success' if row['success'] else 'Failure'}",
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)

st_folium(m, width=900)

# ========== Prediction Interface ==========
st.subheader("🎯 Predict Launch Success Probability")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature (°C)", -20.0, 50.0, 25.0)
    with col2:
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with col3:
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            'temperature': [temperature],
            'humidity': [humidity],
            'wind_speed': [wind_speed]
        })
        prob = model.predict_proba(input_df)[0][1]
        label = "✅ Likely to Succeed" if prob > 0.5 else "❌ Likely to Fail"

        st.metric(label="Success Probability", value=f"{prob*100:.2f}%")
        st.success(label)

# Footer
st.markdown("---")
st.markdown("Built by ❤️ Abubakar")
