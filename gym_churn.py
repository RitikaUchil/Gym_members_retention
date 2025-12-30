# --------------------------
# Gym Churn Prediction Script - Fixed Version
# --------------------------

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap

# --------------------------
# 1. Load Datasets
# --------------------------
members = pd.read_excel(r"C:\Users\yourdigitallift\Documents\GymProject\members.xlsx")
attendance = pd.read_excel(r"C:\Users\yourdigitallift\Documents\GymProject\attendance.xlsx")

# --------------------------
# 2. Preprocessing Members
# --------------------------
members.rename(columns={
    'Number':'PhoneNumber',
    'Start Date':'StartDate',
    'End Date':'EndDate',
    'Plan Name':'PlanName',
    'Plan Status':'PlanStatus',
    'Trainer ID':'TrainerID',
    'Net Amount':'NetAmount',
    'Received Amount':'ReceivedAmount',
    'Amount Pending':'AmountPending'
}, inplace=True)

members = members[['PhoneNumber','DOB','StartDate','EndDate','PlanName',
                   'PlanStatus','TrainerID','NetAmount','ReceivedAmount','AmountPending']]

members['DOB'] = pd.to_datetime(members['DOB'], errors='coerce')
members['StartDate'] = pd.to_datetime(members['StartDate'], errors='coerce')
members['EndDate'] = pd.to_datetime(members['EndDate'], errors='coerce')

members['Age'] = (pd.Timestamp.today() - members['DOB']).dt.days // 365
members['TrainerAssigned'] = np.where(members['TrainerID'].notna(), 1, 0)
members['PaymentRatio'] = members['ReceivedAmount'] / members['NetAmount']
members['PaymentRatio'] = members['PaymentRatio'].fillna(0)

# --------------------------
# 3. Preprocessing Attendance (FIXED)
# --------------------------
attendance.rename(columns={
    'Mobile Number':'PhoneNumber',
    'Checkin Time':'CheckinTime'
}, inplace=True)

attendance['CheckinTime'] = pd.to_datetime(attendance['CheckinTime'], errors='coerce')

attendance_agg = attendance.groupby('PhoneNumber').agg(
    TotalVisits=('CheckinTime', 'count'),
    LastVisit=('CheckinTime', 'max')
).reset_index()

# 🔧 FIX: merge StartDate safely instead of using .loc
attendance_agg = attendance_agg.merge(
    members[['PhoneNumber', 'StartDate']],
    on='PhoneNumber',
    how='left'
)

attendance_agg['MembershipWeeks'] = (
    (pd.Timestamp.today() - attendance_agg['StartDate']).dt.days / 7
)

attendance_agg['MembershipWeeks'] = attendance_agg['MembershipWeeks'].replace(0, np.nan)

attendance_agg['AvgVisitsPerWeek'] = (
    attendance_agg['TotalVisits'] / attendance_agg['MembershipWeeks']
)

attendance_agg['AvgVisitsPerWeek'] = attendance_agg['AvgVisitsPerWeek'].fillna(0)

# --------------------------
# 4. Merge Members + Attendance
# --------------------------
data = members.merge(
    attendance_agg[['PhoneNumber','TotalVisits','AvgVisitsPerWeek','LastVisit']],
    on='PhoneNumber',
    how='left'
)

data['TotalVisits'] = data['TotalVisits'].fillna(0)
data['AvgVisitsPerWeek'] = data['AvgVisitsPerWeek'].fillna(0)

# --------------------------
# 5. Create Churn Target
# --------------------------
today = pd.Timestamp.today()
data['Churn'] = np.where(
    (data['EndDate'] < today) & (data['PlanStatus'].str.lower() != 'active'),
    1, 0
)

# --------------------------
# 6. Model Features
# --------------------------
X = data[['Age','PlanName','TrainerAssigned','PaymentRatio','TotalVisits','AvgVisitsPerWeek']]
X = pd.get_dummies(X, columns=['PlanName'], drop_first=True)
y = data['Churn']

# --------------------------
# 7. Train Model
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 8. Evaluation
# --------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# --------------------------
# 9. Action Recommendations
# --------------------------
def recommend_action(row):
    actions = []
    if row['AvgVisitsPerWeek'] < 0.5:
        actions.append('Invite to group class')
    if row['PaymentRatio'] < 1:
        actions.append('Send payment reminder')
    if row['TrainerAssigned'] == 0:
        actions.append('Offer PT session')
    if not actions:
        actions.append('No immediate action')
    return ', '.join(actions)

data['ActionRecommendation'] = data.apply(recommend_action, axis=1)

# --------------------------
# 10. Save Output
# --------------------------
data['ChurnProbability'] = model.predict_proba(X)[:,1]
data.to_excel(
    r"C:\Users\yourdigitallift\Documents\GymProject\GymChurn_Predictions.xlsx",
    index=False
)

print("✅ Script ran successfully. Error fixed.")


import pickle

# Save the trained model
with open("gym_churn.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as gym_churn.pkl")