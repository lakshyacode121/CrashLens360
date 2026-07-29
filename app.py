import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="CrashLens360",
    page_icon="🚗",
    layout="wide"
)
st.markdown("""

<style>

.stApp{
    background: linear-gradient(135deg,#020617,#0f172a,#172554);
}

.block-container{

    background:rgba(15,23,42,.75);

    backdrop-filter:blur(12px);

    border-radius:20px;

    padding:2rem;

}

</style>
""", unsafe_allow_html=True)


#  Dataset
df = pd.read_csv("cleaned_accident_dataset.csv")

# Keep original data for dashboard
df_original = df.copy()

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

# Create encoded copy only for prediction
df_model = df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

model = joblib.load("risk_model.pkl")


# Sidebar
st.sidebar.image(
    "https://img.icons8.com/fluency/96/car--v1.png",
    width=80
)

st.sidebar.title("CrashLens360")

st.sidebar.caption("AI Road Accident Intelligence")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Data Analysis",
        "🗺 GIS Map",
        "🎯 Prediction",
        "📈 Model Performance",
        "ℹ About"
    ]
     )


# HOME PAGE

if menu == "🏠 Home":
    st.title("🚦 CrashLens360")
    st.subheader("Interactive Accident Intelligence and Risk Mapping System")

    st.markdown("""
    Welcome to **CrashLens360**, an AI-powered dashboard for analyzing road accident data.
    Use the navigation menu to explore accident statistics, visualize trends, view GIS maps,
    and predict accident risk using the trained Machine Learning model.
    """)
    
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🚧 Total Accidents", len(df))

    with col2:
        st.metric("🏙 Cities", df["city"].nunique())

    with col3:
        st.metric("🌦 Weather Types", df["weather"].nunique())

    with col4:
        st.metric("🛣 Road Types", df["road_type"].nunique())

    st.markdown("---")

    st.subheader("📌 Project Overview")

    st.info("""
CrashLens360 is an AI-powered road accident analysis and prediction system.

The application helps users:

• Analyze accident trends

• Visualize accident hotspots using GIS

• Predict accident risk using Machine Learning

• Compare multiple ML models

• Support safer transportation planning
""")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""
### 📊 Features

✔ Interactive Dashboard

✔ GIS Accident Map

✔ Risk Prediction

✔ Model Comparison

✔ Feature Selection

✔ GridSearchCV Optimization
""")

    with c2:

        st.warning("""
### 🤖 Machine Learning

• Linear Regression

• Decision Tree

• Random Forest

• XGBoost

• Optimized Random Forest

• 5-Fold Cross Validation
""")
elif menu == "📂 Dataset":

    st.title("📄 Accident Dataset")

    st.write("This section displays the cleaned accident dataset used in the project.")

    st.markdown("---")

    # first 10 rows
    st.subheader("👀 Dataset Preview")
    st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
     )

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
    # Dataset Information
    st.subheader("🧾 Dataset Information")

    info_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "❓ Missing Values": df.isnull().sum().values
     })

    st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True
      )

    st.markdown("---")

    # Quick Summary
    st.subheader("📊 Quick Summary")

    st.success(f"""
    Dataset contains **{df.shape[0]} accident records** and **{df.shape[1]} columns**.
    Most columns are already cleaned and ready for analysis and visualization.
    """)


elif menu == "📊 Data Analysis":

    st.title("📊 Data Analysis")


    st.subheader("🔍 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
     selected_city = st.selectbox(
        "Select City",
        ["All"] + sorted(df_original["city"].unique().tolist())
    )

    with col2:
     selected_weather = st.selectbox(
        "Select Weather",
        ["All"] + sorted(df_original["weather"].unique().tolist())
    )

    with col3:
     selected_severity = st.selectbox(
        "Select Severity",
        ["All"] + sorted(df_original["accident_severity"].unique().tolist())
    )

 

    filtered_df = df_original.copy()

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
elif menu == "🗺 GIS Map":

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
elif menu == "🎯 Prediction":

    st.title(" Accident Risk Prediction")

    st.write("Enter accident details to estimate the accident risk score.")

    weather = st.selectbox(
    "🌦 Weather",
    encoders["weather"].classes_
     )

    traffic_density = st.selectbox(
    "🚗 Traffic Density",
    encoders["traffic_density"].classes_
     )

    visibility = st.number_input(
    "👀 Visibility",
    min_value=0.0,
    value=10.0
    )

    is_peak_hour = st.selectbox(
    "⏰ Peak Hour",
    [0, 1]
        )

    casualties = st.number_input(
    "🩹 Casualties",
    min_value=0,
    value=0
     )

    vehicles_involved = st.number_input(
    "🚘 Vehicles Involved",
    min_value=1,
    value=2
          )

    temperature = st.number_input(
    "🌡 Temperature (°C)",
    value=25.0
      )

    road_type = st.selectbox(
    "🛣 Road Type",
    encoders["road_type"].classes_
     )

    if st.button("Predict Risk Score"):
        input_data = pd.DataFrame([{

          "weather": encoders["weather"].transform([weather])[0],
          "traffic_density": encoders["traffic_density"].transform([traffic_density])[0],
          "visibility": visibility,
          "is_peak_hour": is_peak_hour,
          "casualties": casualties,
          "vehicles_involved": vehicles_involved,
          "temperature": temperature,
          "road_type": encoders["road_type"].transform([road_type])[0]
          }])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Risk Score: {prediction:.2f}")

        st.success(f"🎯 Predicted Risk Score: {prediction:.2f}")

        if prediction < 0.30:
           st.success("🟢 Low Risk")

        elif prediction < 0.70:
           st.warning("🟡 Medium Risk")

        else:
           st.error("🔴 High Risk")

elif menu == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.markdown("---")

    c1,c2,c3 = st.columns(3)

    with c1:
     st.metric("🏆 Final Model","Optimized Random Forest")

    with c2:
     st.metric("R² Score","0.9174")

    with c3:
     st.metric("Mean CV Score","0.9170")

    st.markdown("---")

    comparison = pd.read_csv("model_comparison.csv")

    st.subheader("📊 Model Comparison")

    st.dataframe(
    comparison,
    use_container_width=True
     )

    fig = px.bar(
    comparison,
    x="Model",
    y="R² Score",
    color="Model",
    text="R² Score",
    title="R² Score Comparison"
    )

    fig.update_layout(
    xaxis_title="Machine Learning Models",
    yaxis_title="R² Score"
     )

    st.plotly_chart(fig,use_container_width=True)

    fig = px.bar(
    comparison,
    x="Model",
    y="MAE",
    color="Model",
    text="MAE",
    title="MAE Comparison"
    )

    st.plotly_chart(fig,use_container_width=True)
    fig = px.bar(
    comparison,
    x="Model",
    y="RMSE",
    color="Model",
    text="RMSE",
    title="RMSE Comparison"
     )

    st.plotly_chart(fig,use_container_width=True)

    fig = px.bar(
    comparison,
    x="Model",
    y="Mean CV Score",
    color="Model",
    text="Mean CV Score",
    title="5-Fold Cross Validation Comparison"
    )

    st.plotly_chart(fig,use_container_width=True)


    st.subheader("⚙ GridSearchCV Best Parameters")

    st.code("""
    Best Parameters

    max_depth = 10

    min_samples_leaf = 2

    min_samples_split = 5

    n_estimators = 200

    Best Cross Validation Score = 0.9170
      """)

    st.subheader("📌 Conclusion")

    st.success("""
    Five Machine Learning regression models were evaluated.

    • Linear Regression

    • Decision Tree

    • Random Forest

    • XGBoost

    • Optimized Random Forest

     Among all models, the Optimized Random Forest achieved the highest performance after GridSearchCV hyperparameter tuning with:

     • R² Score = 0.9174       • MAE = 0.0373      • RMSE = 0.0620     • Mean CV Score = 0.9170
    Therefore, the Optimized Random Forest model was selected as the final prediction model for CrashLens360.
     """)
elif menu == "ℹ About":

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