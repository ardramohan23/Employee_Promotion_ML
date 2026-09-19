import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline   
from imblearn.over_sampling import SMOTE
import joblib


df = pd.read_csv("employee_promotion.csv")


X = df.drop(columns=["is_promoted", "employee_id", "age", "length_of_service", "no_of_trainings", "region"])
y = df["is_promoted"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categorical = ["department", "education", "gender", "recruitment_channel"]
numeric = ["awards_won", "previous_year_rating", "avg_training_score"]


cat_encoder = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ("scaler", StandardScaler(with_mean=False)) 
])

num_encoder = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),            
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_encoder, categorical),
        ("num", num_encoder, numeric)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42,sampling_strategy=0.5)),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42))
])


pipeline.fit(X_train, y_train)


print("Accuracy:", pipeline.score(X_test, y_test))


joblib.dump(pipeline, "employee_promotion_model.pkl")
print("Model saved as employee_promotion_model.pkl")
