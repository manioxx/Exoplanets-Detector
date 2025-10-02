import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

np.random.seed(42)

# 1. Load Dataset
df = pd.read_csv("cumulative.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Features & Labels
features = ['koi_period','koi_duration','koi_depth','koi_prad',
            'koi_teq','koi_insol','koi_steff','koi_srad']

X = df[features].fillna(0)
y = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

print("Feature matrix shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# 3. Split + SMOTE + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 4. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res_scaled, y_train_res)

rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nRandom Forest Report:\n")
print(classification_report(y_test, rf_pred, digits=4))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 4))

cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Planet","Planet"],
            yticklabels=["Not Planet","Planet"])
plt.title("Random Forest Confusion Matrix")
plt.show()

# Feature importance
feat_importances = pd.Series(rf_model.feature_importances_, index=features)
feat_importances.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importance for Exoplanet Detection")
plt.show()

# 5. Save Model & Scaler
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved: rf_model.pkl, scaler.pkl")

# 6. Sample Prediction
example_array = X_test_scaled[0].reshape(1, -1)
example_dict = dict(zip(features, example_array.flatten()))
feats_df = pd.DataFrame([example_dict])
example_pred = rf_model.predict(scaler.transform(feats_df))[0]
example_conf = rf_model.predict_proba(scaler.transform(feats_df))[0][1]
print("\nSample Prediction:", "Exoplanet 🌍✨" if example_pred==1 else "Not a planet ❌", 
      "Confidence:", round(example_conf,2))
