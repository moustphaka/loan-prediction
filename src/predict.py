import joblib
import pandas as pd

from src.config import BEST_MODEL

FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

DEFAULT_FEATURES = {
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "CoapplicantIncome": 0.0,
    "Loan_Amount_Term": 360,
    "Property_Area": "Urban"
}


def predict_loan(data):
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            "Le modèle n'a pas été trouvé. Exécutez d'abord 'python main.py' "
            f"pour créer {BEST_MODEL}."
        )

    model = joblib.load(BEST_MODEL)

    df = pd.DataFrame([data])

    for col, value in DEFAULT_FEATURES.items():
        if col not in df.columns:
            df[col] = value

    df = df[FEATURE_COLUMNS]

    prediction = model.predict(df)
    return prediction[0]
