import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import joblib


df = pd.read_csv("employee_promotion.csv")

X = df.drop(columns=["is_promoted", "employee_id"])
y = df["is_promoted"]

categorical = ["department", "education", "gender", "recruitment_channel"]
numeric = ["awards_won",  "previous_year_rating", "avg_training_score"]

cat_encoder = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ("scaler", StandardScaler(with_mean=False))  
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_encoder, categorical),
        ("num", StandardScaler(), numeric)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "employee_promotion_model.pkl")
print("Model saved as employee_promotion_model.pkl")
