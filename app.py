# --------------------------------------------------
# Gym Owner Dashboard – Retention Intelligence Pro
# ML-based Churn Prediction + Dynamic Filters
# --------------------------------------------------

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
    layout="wide"
)

# --------------------------
# Background + Glass UI
# --------------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(0,0,0,0.65);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 18px;
        }}
        .metric-card {{
            background: rgba(0,0,0,0.85);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
        }}
        .metric-card h1 {{
            background: linear-gradient(to right, #00f5ff, #ff00f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 36px;
        }}
        .metric-card p {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# OPTIONAL background image
# set_background("assets/bg.jpg")

# --------------------------
# Title
# --------------------------
st.title("🏋️ Gym Owner Retention Dashboard (ML Powered)")

# --------------------------
# Column Mapping
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

def auto_map_columns(df, required_map):
    mapped = {}
    for std_col, possible_names in required_map.items():
        for col in df.columns:
            if col.strip().lower() in [p.lower() for p in possible_names]:
                mapped[col] = std_col
                break
    return mapped

# --------------------------
# Upload Files
# --------------------------
members_file = st.file_uploader("📂 Upload Members Excel", type=["xlsx"])
attendance_file = st.file_uploader("📂 Upload Attendance Excel", type=["xlsx"])

if members_file and attendance_file:

    members = pd.read_excel(members_file)
    attendance = pd.read_excel(attendance_file)

    # --------------------------
    # Auto Column Mapping
    # --------------------------
    members = members.rename(columns=auto_map_columns(members, REQUIRED_MEMBERS_COLS))
    attendance = attendance.rename(columns=auto_map_columns(attendance, REQUIRED_ATTENDANCE_COLS))

    # --------------------------
    # Feature Engineering
    # --------------------------
    members['DOB'] = pd.to_datetime(members['DOB'], errors='coerce')
    members['StartDate'] = pd.to_datetime(members['StartDate'], errors='coerce')
    members['EndDate'] = pd.to_datetime(members['EndDate'], errors='coerce')

    members['Age'] = (pd.Timestamp.today() - members['DOB']).dt.days // 365
    members['TrainerAssigned'] = np.where(members['TrainerID'].notna(), 1, 0)
    members['PaymentRatio'] = (members['ReceivedAmount'] / members['NetAmount']).fillna(0)

    if 'Gender' not in members.columns:
        members['Gender'] = 'Other'

    attendance['CheckinTime'] = pd.to_datetime(attendance['CheckinTime'], errors='coerce')
    visits = attendance.groupby("PhoneNumber").size().reset_index(name="TotalVisits")

    data = members.merge(visits, on="PhoneNumber", how="left").fillna(0)

    data['MembershipWeeks'] = ((pd.Timestamp.today() - data['StartDate']).dt.days / 7).clip(lower=1)
    data['AvgVisitsPerWeek'] = data['TotalVisits'] / data['MembershipWeeks']

    # --------------------------
    # Load ML Model
    # --------------------------
    with open("gym_churn.pkl", "rb") as f:
        model = pickle.load(f)

    features = [
        'Age', 'Gender', 'PlanName',
        'TrainerAssigned', 'PaymentRatio',
        'TotalVisits', 'AvgVisitsPerWeek'
    ]

    X = data[features]
    X = pd.get_dummies(X, columns=['Gender', 'PlanName'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0

    X = X[model.feature_names_in_]

    data['ChurnProbability'] = model.predict_proba(X)[:, 1]
    data['RetentionProbability'] = 1 - data['ChurnProbability']

    data['RiskLevel'] = pd.cut(
        data['ChurnProbability'],
        bins=[0, 0.4, 0.7, 1],
        labels=['Low', 'Medium', 'High']
    )

    data['RecommendedAction'] = data['RiskLevel'].astype(str).map({
        'High': 'Personal call + Free PT',
        'Medium': 'WhatsApp reminder + Free class',
        'Low': 'Maintain engagement'
    })

    data['CouponOffer'] = data['RiskLevel'].astype(str).map({
        'High': '20% Renewal Discount',
        'Medium': '10% Discount',
        'Low': 'Referral Coupon'
    })

    # --------------------------
    # Sidebar Filters
    # --------------------------
    st.sidebar.header("🔎 Filters")

    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        options=data['RiskLevel'].unique(),
        default=data['RiskLevel'].unique()
    )

    plan_filter = st.sidebar.multiselect(
        "Plan Name",
        options=data['PlanName'].unique(),
        default=data['PlanName'].unique()
    )

    search_text = st.sidebar.text_input("Search by Name / Phone")

    filtered_data = data[
        (data['RiskLevel'].isin(risk_filter)) &
        (data['PlanName'].isin(plan_filter))
    ]

    if search_text:
        filtered_data = filtered_data[
            filtered_data['Name'].str.contains(search_text, case=False, na=False) |
            filtered_data['PhoneNumber'].astype(str).str.contains(search_text)
        ]

    # --------------------------
    # KPI Metrics
    # --------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"<div class='metric-card'><h1>{len(filtered_data)}</h1><p>Total Members</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h1>{(filtered_data['RiskLevel']=='High').sum()}</h1><p>High Risk</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h1>{round(filtered_data['AvgVisitsPerWeek'].mean(),2)}</h1><p>Avg Visits / Week</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h1>{round(filtered_data['PaymentRatio'].mean(),2)}</h1><p>Payment Ratio</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------
    # Charts
    # --------------------------
    st.subheader("📊 Risk Distribution")
    st.plotly_chart(px.pie(filtered_data, names="RiskLevel", hole=0.45, template="plotly_dark"), use_container_width=True)

    st.subheader("📊 Avg Visits per Week")
    st.plotly_chart(px.box(filtered_data, x="RiskLevel", y="AvgVisitsPerWeek", template="plotly_dark"), use_container_width=True)

    st.subheader("📊 Churn by Plan")
    plan_churn = filtered_data.groupby("PlanName")["ChurnProbability"].mean().reset_index()
    st.plotly_chart(px.bar(plan_churn, x="PlanName", y="ChurnProbability", template="plotly_dark"), use_container_width=True)

    # --------------------------
    # Members Table (RESTORED)
    # --------------------------
    st.subheader("👥 Members Overview")

    table_cols = [
        "Name", "PhoneNumber", "PlanName", "PlanStatus",
        "RiskLevel", "RetentionProbability",
        "AvgVisitsPerWeek", "PaymentRatio"
    ]

    st.dataframe(
        filtered_data[table_cols].sort_values("RetentionProbability"),
        use_container_width=True
    )

    # --------------------------
    # Recovery Action Plan
    # --------------------------
    st.subheader("📋 Recovery Action Plan")

    export_cols = [
        "Name", "PhoneNumber", "RiskLevel",
        "RecommendedAction", "CouponOffer",
        "RetentionProbability", "AvgVisitsPerWeek", "PaymentRatio"
    ]

    st.dataframe(filtered_data[export_cols], use_container_width=True)

    buffer = io.BytesIO()
    filtered_data[export_cols].to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "📥 Download Recovery Plan",
        data=buffer,
        file_name="gym_recovery_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("⬆️ Upload both Members & Attendance Excel files to start")
