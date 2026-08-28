import joblib
model = joblib.load("churn_ensemble_pipeline.pkl")
print(list(model.feature_names_in_))
