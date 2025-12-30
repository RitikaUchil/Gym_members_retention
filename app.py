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
st.set_page_config(page_title="Gym Owner Dashboard", layout="wide")

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
# Auto Column Mapping Function
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
            background-position: center;
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
            text-shadow: 1px 1px 6px rgba(0,0,0,0.6);
        }}
        .metric-card p {{
            color: white;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Optional: set your gym image
set_background("assets/bg.jpg")

# --------------------------
# Header with Glow
# --------------------------
st.markdown(
    """
    <div style="
        background: rgba(0,0,0,0.65);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
    ">
        <h1 style='color:#00f5ff;>
            🏋️ Gym Owner Retention Dashboard (ML Predictions)
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Upload Section
# --------------------------
st.markdown(
    """
    <div style="
        background: rgba(0,0,0,0.65);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 30px;
    ">
    """,
    unsafe_allow_html=True
)

members_file = st.file_uploader("Upload Members Excel", type=["xlsx"], key="members")
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"], key="attendance")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Process Uploaded Files
# --------------------------
if members_file and attendance_file:
    members = pd.read_excel(members_file)
    attendance = pd.read_excel(attendance_file)

    # Map Columns
    member_map = auto_map_columns(members, REQUIRED_MEMBERS_COLS)
    attendance_map = auto_map_columns(attendance, REQUIRED_ATTENDANCE_COLS)

    members = members.rename(columns=member_map)
    attendance = attendance.rename(columns=attendance_map)

    # Check Required Columns
    missing_members = set(REQUIRED_MEMBERS_COLS.keys()) - set(members.columns)
    missing_attendance = set(REQUIRED_ATTENDANCE_COLS.keys()) - set(attendance.columns)
    if missing_members:
        st.error(f"Missing columns in Members file: {', '.join(missing_members)}")
        st.stop()
    if missing_attendance:
        st.error(f"Missing columns in Attendance file: {', '.join(missing_attendance)}")
        st.stop()

    # Feature Engineering
    members['DOB'] = pd.to_datetime(members['DOB'], errors='coerce')
    members['StartDate'] = pd.to_datetime(members['StartDate'], errors='coerce')
    members['EndDate'] = pd.to_datetime(members['EndDate'], errors='coerce')

    members['Age'] = (pd.Timestamp.today() - members['DOB']).dt.days // 365
    members['TrainerAssigned'] = np.where(members['TrainerID'].notna(), 1, 0)
    members['PaymentRatio'] = (members['ReceivedAmount'] / members['NetAmount']).fillna(0)

    if 'Gender' not in members.columns:
        members['Gender'] = 'Other'

    attendance['CheckinTime'] = pd.to_datetime(attendance['CheckinTime'], errors='coerce')
    attendance_agg = attendance.groupby('PhoneNumber').agg(TotalVisits=('CheckinTime','count')).reset_index()

    data = members.merge(attendance_agg, on='PhoneNumber', how='left').fillna(0)
    data['MembershipWeeks'] = ((pd.Timestamp.today() - data['StartDate']).dt.days / 7).clip(lower=1)
    data['AvgVisitsPerWeek'] = data['TotalVisits'] / data['MembershipWeeks']

    # Load Pickle Model
    with open("gym_churn.pkl", "rb") as f:
        model = pickle.load(f)

    # Prepare Features
    features = ['Age','Gender','PlanName','TrainerAssigned','PaymentRatio','TotalVisits','AvgVisitsPerWeek']
    X_app = data[features]
    X_app = pd.get_dummies(X_app, columns=['Gender','PlanName'], drop_first=True)

    missing_cols = set(model.feature_names_in_) - set(X_app.columns)
    for col in missing_cols:
        X_app[col] = 0
    X_app = X_app[model.feature_names_in_]

    # Predict Churn Probability
    data['ChurnProbability'] = model.predict_proba(X_app)[:,1]
    data['RetentionProbability'] = 1 - data['ChurnProbability']

    # Risk Levels
    data['RiskLevel'] = pd.cut(data['ChurnProbability'], bins=[0,0.4,0.7,1], labels=['Low','Medium','High'])

    # Actions & Coupons
    def action(r): return "Personal call + Free PT" if r=='High' else "WhatsApp reminder + Free class" if r=='Medium' else "Maintain engagement"
    def coupon(r): return "20% Renewal Discount" if r=='High' else "10% Discount" if r=='Medium' else "Referral Coupon"

    data['RecommendedAction'] = data['RiskLevel'].astype(str).apply(action)
    data['CouponOffer'] = data['RiskLevel'].astype(str).apply(coupon)

    # --------------------------
    # Sidebar Filters
    # --------------------------
    st.sidebar.header("Filters")
    risk_filter = st.sidebar.multiselect("Risk Level", data['RiskLevel'].unique(), default=data['RiskLevel'].unique())
    filtered_data = data[data['RiskLevel'].isin(risk_filter)]

    # --------------------------
    # Metrics
    # --------------------------
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h1>{len(filtered_data)}</h1><p>Total Members</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h1>{(filtered_data['RiskLevel']=='High').sum()}</h1><p>High Risk</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h1>{round(filtered_data['AvgVisitsPerWeek'].mean(),2)}</h1><p>Avg Visits / Week</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h1>{round(filtered_data['PaymentRatio'].mean(),2)}</h1><p>Payment Ratio</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------
    # Charts
    # --------------------------
    st.subheader("Risk Distribution")
    fig1 = px.pie(filtered_data, names="RiskLevel", hole=0.45, template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Avg Visits Per Week")
    fig2 = px.box(filtered_data, x="RiskLevel", y="AvgVisitsPerWeek", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Payment Ratio")
    fig3 = px.violin(filtered_data, x="RiskLevel", y="PaymentRatio", box=True, template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Churn by Plan")
    plan_churn = filtered_data.groupby("PlanName")["ChurnProbability"].mean().reset_index()
    fig4 = px.bar(plan_churn, x="PlanName", y="ChurnProbability", color="ChurnProbability", template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    # --------------------------
    # Recovery Action Table + Export
    # --------------------------
    st.subheader("📋 Recovery Action Plan")
    export_cols = ["Name","PhoneNumber","RiskLevel","RecommendedAction","CouponOffer","RetentionProbability","AvgVisitsPerWeek","PaymentRatio"]
    st.dataframe(filtered_data[export_cols], use_container_width=True)

    buffer = io.BytesIO()
    filtered_data[export_cols].to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button("📥 Download Recovery Plan Excel", data=buffer, file_name="gym_recovery_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Please upload both Members and Attendance files")
