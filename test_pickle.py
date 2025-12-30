import pickle

with open("gym_churn.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Pickle loaded successfully. Model type:", type(model))