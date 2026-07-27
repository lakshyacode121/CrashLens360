import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="CrashLens360",
    page_icon="🚗",
    layout="wide"
)


#  Dataset
df = pd.read_csv("cleaned_accident_dataset.csv")

encoders = {}

categorical_columns = [
    "city",
    "state",
    "date",
    "time",
    "road_type",
    "traffic_signal",
    "weather",
    "traffic_density",
    "cause",
    "festival"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

model = joblib.load("risk_model")


# Sidebar
st.sidebar.title("🚗 CrashLens360")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset", "Data Analysis","GIS Map","Prediction", "About"]
)


# HOME PAGE
if menu == "Home":

    st.title("🚗CrashLens360")
    st.subheader("Interactive Accident Intelligence and Risk Mapping System")

    st.markdown("---")

    st.header("Project Objective")

    st.write("""
    This project analyzes road accident data to identify accident patterns,
    high-risk areas, and important factors affecting road safety.

    The dashboard helps users understand accident trends through interactive
    graphs and visualizations.
    """)

    st.markdown("---")

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
     st.metric("🚗 Total Accidents", len(df))

    with col2:
     st.metric("🏙️ Total Cities", df["city"].nunique())

    with col3:
     st.metric("🌦️ Weather Types", df["weather"].nunique())

    with col4:
     st.metric("⚠️ Avg Risk Score", round(df["risk_score"].mean(), 2))


elif menu == "Dataset":

    st.title("📄 Accident Dataset")

    st.write("This section displays the cleaned accident dataset used in the project.")

    st.markdown("---")

    # first 10 rows
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(10))

    st.markdown("---")

    # Basic Information
    st.subheader("📌 Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    st.markdown("---")

    # Column Names
    st.subheader("🧾 Column Names")
    st.write(list(df.columns))

    st.markdown("---")

    # Missing Values
    st.subheader("❓ Missing Values")

    missing = df.isnull().sum()

    st.dataframe(missing[missing > 0].reset_index().rename(columns={"index":"Column",0:"Missing Values"}))

    st.markdown("---")

    # Data Types
    st.subheader("🔧 Data Types")

    dtypes = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(dtypes)

    st.markdown("---")

    # Quick Summary
    st.subheader("📊 Quick Summary")

    st.success(f"""
    Dataset contains **{df.shape[0]} accident records** and **{df.shape[1]} columns**.
    Most columns are already cleaned and ready for analysis and visualization.
    """)


elif menu == "Data Analysis":

    st.title("📊 Data Analysis")


    st.subheader("🔍 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
     selected_city = st.selectbox(
        "Select City",
        ["All"] + sorted(df["city"].unique().tolist())
    )

    with col2:
     selected_weather = st.selectbox(
        "Select Weather",
        ["All"] + sorted(df["weather"].unique().tolist())
    )

    with col3:
     selected_severity = st.selectbox(
        "Select Severity",
        ["All"] + sorted(df["accident_severity"].unique().tolist())
    )

 

    filtered_df = df.copy()

    if selected_city != "All":
     filtered_df = filtered_df[filtered_df["city"] == selected_city]

    if selected_weather != "All":
     filtered_df = filtered_df[filtered_df["weather"] == selected_weather]

    if selected_severity != "All":
     filtered_df = filtered_df[
        filtered_df["accident_severity"] == selected_severity
    ]

    st.success(f"Showing {len(filtered_df)} accident records.")

    st.markdown("---")

    st.subheader("🚗 Accident Severity Analysis")

    severity = filtered_df["accident_severity"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(severity.index, severity.values)

    ax.set_xlabel("Severity")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accident Severity Distribution")

    st.pyplot(fig)

    highest = severity.idxmax()

    st.info(f"""
     ### Analysis

    > The most common accident severity is **{highest}**.
    > This indicates that most accidents in the selected data fall under the **{highest}** category.
    > Use the filters above to compare different cities, weather conditions, and severity levels.
     """)


    st.markdown("---")

    st.subheader("🕒 Hour-wise Accident Analysis")


    hourly = filtered_df["hour"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,4))


    ax.plot(hourly.index, hourly.values, marker="o", linewidth=2)


    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accidents by Hour")

    st.pyplot(fig)



    peak_hour = hourly.idxmax()

    st.info(f"""
    ###  Analysis

    > The highest number of accidents occurs around **{peak_hour}:00 hours**.

    > This indicates that this hour has the maximum accident frequency in the selected data.

    > Users can change the filters to compare accident timings for different cities and weather conditions.
    """)


    st.markdown("---")

    st.subheader("🌦️ Weather Analysis")

    weather = filtered_df["weather"].value_counts()


    fig, ax = plt.subplots(figsize=(6,6))


    ax.pie(
     weather.values,
     labels=weather.index,
     autopct="%1.1f%%",
     startangle=90
    )

    ax.set_title("Accidents by Weather Condition")

    st.pyplot(fig)



    highest_weather = weather.idxmax()

    st.info(f""" 

    ###  Analysis

    > Most accidents occurred during **{highest_weather}** weather.

    > This indicates that **{highest_weather}** weather had the highest accident frequency in the selected data.

    > Use the filters to compare weather conditions across different cities and severity levels.
     """)

    st.markdown("---")

    st.subheader("🛣️ Road Type Analysis")

    road = filtered_df["road_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(road.index, road.values, color="green")

    ax.set_xlabel("Road Type")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accidents by Road Type")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    highest_road = road.idxmax()

    st.info(f"""
     ###  Analysis

    > Most accidents occurred on **{highest_road}** roads.

    > This suggests that **{highest_road}** roads have the highest accident frequency in the selected data.
     """)



    st.markdown("---")

    st.subheader("🚦 Traffic Density Analysis")

    traffic = filtered_df["traffic_density"].value_counts()

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(traffic.index, traffic.values, color="orange")

    ax.set_xlabel("Traffic Density")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accidents by Traffic Density")

    st.pyplot(fig)

    highest_traffic = traffic.idxmax()

    st.info(f"""

      ###  Analysis

     > The highest number of accidents occurred during **{highest_traffic}** traffic conditions.

     > Traffic density plays an important role in accident occurrence.
       """)


    st.markdown("---")

    st.subheader("🌡️ Temperature Distribution")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(filtered_df["temperature"], bins=20, color="purple")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Temperature Distribution")

    st.pyplot(fig)

    avg_temp = round(filtered_df["temperature"].mean(), 2)

    st.info(f"""
     ###  Analysis

     > The average temperature during accidents is **{avg_temp} °C**.

     > The histogram shows how accident records are distributed across different temperature ranges.
       """)
elif menu == "GIS Map":

    st.title("🗺️ GIS-Based Accident Visualization")

    st.write("This page displays accident locations on an interactive map.")

    # Create map centered on India
    # Create India Map
    m = folium.Map(location=[22.97, 78.65], zoom_start=5)

   # Display first 500 accident locations
    # Prepare latitude and longitude for HeatMap
    heat_data = df[["latitude", "longitude"]].dropna().values.tolist()

    # Add HeatMap
    HeatMap(
     heat_data,
     radius=12,
     blur=18,
     max_zoom=10
    ).add_to(m)
    st_folium(m, width=1000, height=600)
elif menu == "Prediction":

    st.title(" Accident Risk Prediction")

    st.write("Enter accident details to estimate the accident risk score.")

    city = st.selectbox("City", encoders["city"].classes_)
    state = st.selectbox("State", encoders["state"].classes_)
    date = st.selectbox("Date", encoders["date"].classes_)
    time = st.selectbox("Time", encoders["time"].classes_)
    hour = st.slider("Hour", 0, 23, 12)
    day_of_week = st.slider("Day of Week", 0, 6, 0)
    is_weekend = st.selectbox("Weekend", [0, 1])
    road_type = st.selectbox("Road Type", encoders["road_type"].classes_)
    lanes = st.number_input("Number of Lanes", 1, 10, 2)
    traffic_signal = st.selectbox("Traffic Signal", encoders["traffic_signal"].classes_)
    weather = st.selectbox("Weather", encoders["weather"].classes_)
    visibility = st.number_input("Visibility", value=10.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    traffic_density = st.selectbox("Traffic Density", encoders["traffic_density"].classes_)
    cause = st.selectbox("Cause", encoders["cause"].classes_)
    vehicles_involved = st.number_input("Vehicles Involved", 1, 10, 2)
    casualties = st.number_input("Casualties", 0, 20, 0)
    is_peak_hour = st.selectbox("Peak Hour", [0, 1])
    festival = st.selectbox("Festival", encoders["festival"].classes_)
    latitude = st.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=28.6139,
    format="%.6f"
     )

    longitude = st.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=77.2090,
    format="%.6f"
     )

    if st.button("Predict Risk Score"):
        input_data = pd.DataFrame([{
            "city": encoders["city"].transform([city])[0],
            "state": encoders["state"].transform([state])[0],
            "latitude": latitude,
            "longitude": longitude,
            "date": encoders["date"].transform([date])[0],
            "time": encoders["time"].transform([time])[0],
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "road_type": encoders["road_type"].transform([road_type])[0],
            "lanes": lanes,
            "traffic_signal": encoders["traffic_signal"].transform([traffic_signal])[0],
            "weather": encoders["weather"].transform([weather])[0],
            "visibility": visibility,
            "temperature": temperature,
            "traffic_density": encoders["traffic_density"].transform([traffic_density])[0],
            "cause": encoders["cause"].transform([cause])[0],
            "vehicles_involved": vehicles_involved,
            "casualties": casualties,
            "is_peak_hour": is_peak_hour,
            "festival": encoders["festival"].transform([festival])[0]
             }])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Risk Score: {prediction:.2f}")

        if prediction < 0.3:
          st.success(" Low Risk")
        elif prediction < 0.7:
          st.warning(" Medium Risk")
        else:
          st.error(" High Risk")




elif menu == "About":

    st.title("ℹ️ About CrashLens360")

    st.markdown("""
       ### 🚗 Project Name

        CrashLens360: Interactive Accident Intelligence and Risk Mapping System

       ### 🎯 Objective

        Analyze road accidents using data visualization and machine learning.

       ### 🛠 Technologies Used

      - Python
      - Streamlit
      - Pandas
      - Matplotlib
      - Scikit-Learn
      - Power BI

      ### 👨‍🎓 Developed By

      Lakshya Sharma    AND   Tanishka Solanki
      BCA 2nd Year Project
       """)

    csv = df.to_csv(index=False)

    st.download_button(
    "📥 Download Dataset",
    csv,
    "AccidentData.csv",
    "text/csv" 
     )