import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.config import TRAIN_DATA, BEST_MODEL


def ensure_train_data_exists():
    if TRAIN_DATA.exists():
        return

    TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    sample_data = [
        {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5849,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 128,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 4583,
            "CoapplicantIncome": 1508.0,
            "LoanAmount": 128,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Rural",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Female",
            "Married": "No",
            "Dependents": "0",
            "Education": "Not Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 4006,
            "CoapplicantIncome": 1526.0,
            "LoanAmount": 168,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "Yes",
            "ApplicantIncome": 5417,
            "CoapplicantIncome": 4196.0,
            "LoanAmount": 267,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Female",
            "Married": "No",
            "Dependents": "0",
            "Education": "Not Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 12841,
            "CoapplicantIncome": 10968.0,
            "LoanAmount": 349,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "2",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 2333,
            "CoapplicantIncome": 1516.0,
            "LoanAmount": 95,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Rural",
            "Loan_Status": "Y",
        },
        {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Not Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 3036,
            "CoapplicantIncome": 2504.0,
            "LoanAmount": 158,
            "Loan_Amount_Term": 360,
            "Credit_History": 0,
            "Property_Area": "Urban",
            "Loan_Status": "N",
        },
        {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 1853,
            "CoapplicantIncome": 2840.0,
            "LoanAmount": 120,
            "Loan_Amount_Term": 360,
            "Credit_History": 0,
            "Property_Area": "Urban",
            "Loan_Status": "N",
        },
    ]
    pd.DataFrame(sample_data).to_csv(TRAIN_DATA, index=False)
    print(
        f"Missing train.csv. Generated sample training data at {TRAIN_DATA}"
    )


def train_model():

    ensure_train_data_exists()
    print("train_model() was called")
    print("Loading data...")
    df = pd.read_csv(TRAIN_DATA)
    print("DATA LOADED:", df.shape)
    print(df.head())

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier())
    ])

    print("Training model...")
    pipeline.fit(X, y)
    print("FIT OK")
    print("PIPELINE OK")

    print("Saving to:", BEST_MODEL)
    os.makedirs(BEST_MODEL.parent, exist_ok=True)
    joblib.dump(pipeline, BEST_MODEL)
    print(f"Model saved: {BEST_MODEL}")
    print("Model saved successfully!")

    return pipeline
