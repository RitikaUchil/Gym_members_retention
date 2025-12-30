# --------------------------
# Gym Owner Dashboard - Retention Intelligence Pro (ML via Pickle)
# --------------------------

import pandas as pd
import numpy as np
import streamlit as st
import base64
import io
import plotly.express as px
import pickle

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Gym Retention Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Premium Background + UI
# --------------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Background */
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main glass container */
        .block-container {{
            background: rgba(0, 0, 0, 0.55);
            backdrop-filter: blur(14px);
            padding: 2rem;
            border-radius: 24px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: rgba(0,0,0,0.9);
            backdrop-filter: blur(12px);
        }}

        /* Metric cards */
        .metric-card {{
            background: rgba(15, 15, 15, 0.9);
            border-radius: 20px;
            padding: 22px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0,255,255,0.25);
            transition: all 0.35s ease;
        }}

        .metric-card:hover {{
            transform: scale(1.06);
            box-shadow: 0 0 40px rgba(255,0,255,0.6);
        }}

        .metric-card h1 {{
            font-size: 42px;
            background: linear-gradient(90deg, #00f5ff, #ff00f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }}

        .metric-card p {{
            color: #cccccc;
            font-size: 15px;
            margin-top: 6px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 ACTIVATE BACKGROUND
set_background("assets/bg.jpg")

# --------------------------
# Title
# --------------------------
st.title("🏋️ Gym Owner Retention Intelligence Dashboard")

# --------------------------
# Required Columns Mapping
# --------------------------
REQUIRED_MEMBERS_COLS = {
    "PhoneNumber": ["Number", "Mobile", "Phone", "Phone Number"],
    "Name": ["Name", "Member Name"],
    "DOB": ["DOB", "Date of Birth"],
    "StartDate": ["Start Date", "Joining Date"],
    "EndDate": ["End Date", "Expiry Date"],
    "PlanName": ["Plan Name", "Membership Plan"],
    "PlanStatus": ["Plan Status", "Status"],
    "NetAmount": ["Net Amount", "Total Amount"],
    "ReceivedAmount": ["Received Amount", "Paid Amount"],
    "TrainerID": ["Trainer ID", "PT ID"]
}

REQUIRED_ATTENDANCE_COLS = {
    "PhoneNumber": ["Mobile Number", "Phone Number", "Number"],
    "CheckinTime": ["Checkin Time", "Check-in Time", "Attendance Time"]
}

# --------------------------
# Auto Column Mapping
# --------------------------
def auto_map_columns(df, required_map):
    mapped = {}
    for standard_col, possible_names in required_map.items():
        for col in df.columns:
            if col.strip().lower() in [p.lower() for p in possible_names]:
                mapped[col] = standard_col
                break
    return mapped

# --------------------------
# File Upload
# --------------------------
members_file = st.file_uploader("📤 Upload Members Excel", type=["xlsx"])
attendance_file = st.file_uploader("📤 Upload Attendance Excel", type=["xlsx"])

if members_file and attendance_file:

    members = pd.read_excel(members_file)
    attendance = pd.read_excel(attendance_file)

    members.rename(columns=auto_map_columns(members, REQUIRED_MEMBERS_COLS), inplace=True)
    attendance.rename(columns=auto_map_columns(attendance, REQUIRED_ATTENDANCE_COLS), inplace=True)

    # --------------------------
    # Feature Engineering
    # --------------------------
    members['DOB'] = pd.to_datetime(members['DOB'], errors='coerce')
    members['StartDate'] = pd.to_datetime(members['StartDate'], errors='coerce')

    members['Age'] = (pd.Timestamp.today() - members['DOB']).dt.days // 365
    members['TrainerAssigned'] = np.where(members['TrainerID'].notna(), 1, 0)
    members['PaymentRatio'] = (members['ReceivedAmount'] / members['NetAmount']).fillna(0)

    if 'Gender' not in members.columns:
        members['Gender'] = 'Other'

    attendance['CheckinTime'] = pd.to_datetime(attendance['CheckinTime'], errors='coerce')
    attendance_agg = attendance.groupby('PhoneNumber').size().reset_index(name='TotalVisits')

    data = members.merge(attendance_agg, on='PhoneNumber', how='left').fillna(0)
    data['MembershipWeeks'] = ((pd.Timestamp.today() - data['StartDate']).dt.days / 7).clip(lower=1)
    data['AvgVisitsPerWeek'] = data['TotalVisits'] / data['MembershipWeeks']

    # --------------------------
    # Load ML Model
    # --------------------------
    with open("gym_churn.pkl", "rb") as f:
        model = pickle.load(f)

    # --------------------------
    # Prepare Prediction Data
    # --------------------------
    features = ['Age','Gender','PlanName','TrainerAssigned','PaymentRatio','TotalVisits','AvgVisitsPerWeek']
    X = pd.get_dummies(data[features], columns=['Gender','PlanName'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0

    X = X[model.feature_names_in_]

    # --------------------------
    # Predictions
    # --------------------------
    data['ChurnProbability'] = model.predict_proba(X)[:,1]
    data['RiskLevel'] = pd.cut(
        data['ChurnProbability'],
        bins=[0,0.4,0.7,1],
        labels=['Low','Medium','High']
    )

    # --------------------------
    # Metrics
    # --------------------------
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h1>{len(data)}</h1><p>Total Members</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h1>{(data['RiskLevel']=='High').sum()}</h1><p>High Risk</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h1>{round(data['AvgVisitsPerWeek'].mean(),2)}</h1><p>Avg Visits / Week</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h1>{round(data['PaymentRatio'].mean(),2)}</h1><p>Payment Ratio</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Risk Distribution")
    st.plotly_chart(px.pie(data, names="RiskLevel", hole=0.45, template="plotly_dark"), use_container_width=True)

else:
    st.info("⬆️ Upload Members & Attendance files to continue")
