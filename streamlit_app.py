import pandas as pd
import streamlit as st
from pathlib import Path

from src.config import BEST_MODEL, TRAIN_DATA
from src.predict import predict_loan

st.set_page_config(
    page_title="Prédiction de prêt bancaire",
    page_icon="🏦",
    layout="wide",
)

dark_mode = st.sidebar.checkbox("Activer le mode sombre", value=False)

background = "#04081b" if dark_mode else "#f1f5f9"
app_background = (
    "linear-gradient(180deg, #020617 0%, #0f172a 100%)"
    if dark_mode
    else "linear-gradient(180deg, #e2e8f0 0%, #f8fafc 100%)"
)
card_background = "#0f172a" if dark_mode else "white"
card_border = (
    "rgba(148, 163, 184, 0.24)"
    if dark_mode
    else "rgba(148, 163, 184, 0.16)"
)
text_color = "#e2e8f0" if dark_mode else "#0f172a"
secondary_text = "#94a3b8" if dark_mode else "#475569"
hero_text = "#f8fafc" if dark_mode else "white"
button_bg = "#2563eb" if dark_mode else "#2563eb"
button_hover = "#1d4ed8"

st.markdown(
    f"""
    <style>
    body {{
        background: {background};
        color: {text_color};
    }}
    .stApp {{
        background: {app_background};
    }}
    .hero {{
        border-radius: 24px;
        padding: 2rem;
        background: linear-gradient(135deg, #111827 0%, #2563eb 100%);
        color: {hero_text};
        box-shadow: 0 25px 60px rgba(15, 23, 42, 0.24);
        animation: float 8s ease-in-out infinite;
    }}
    .card {{
        border-radius: 20px;
        padding: 1.5rem;
        background: {card_background};
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.1);
        border: 1px solid {card_border};
    }}
    .card, .card * {{
        color: {text_color} !important;
    }}
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    .stMarkdown strong,
    .stMarkdown em,
    .streamlit-expanderHeader,
    .stText {{
        color: {text_color} !important;
    }}
    .section-title {{
        color: {text_color};
        font-weight: 700;
    }}
    .highlight {{
        color: #60a5fa;
        font-weight: 700;
    }}
    .small-text {{
        color: {secondary_text};
    }}
    .stButton>button {{
        background: {button_bg};
        color: white;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        border: none;
    }}
    .stButton>button:hover {{
        background: {button_hover};
        color: white;
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dashboard_data():
    if TRAIN_DATA.exists():
        return pd.read_csv(TRAIN_DATA)

    return pd.DataFrame(
        [
            {
                "Loan_Status": "Y",
                "ApplicantIncome": 6000,
                "LoanAmount": 120,
                "Gender": "Male",
            },
            {
                "Loan_Status": "N",
                "ApplicantIncome": 3000,
                "LoanAmount": 85,
                "Gender": "Female",
            },
            {
                "Loan_Status": "Y",
                "ApplicantIncome": 4100,
                "LoanAmount": 120,
                "Gender": "Male",
            },
            {
                "Loan_Status": "Y",
                "ApplicantIncome": 5300,
                "LoanAmount": 150,
                "Gender": "Female",
            },
            {
                "Loan_Status": "N",
                "ApplicantIncome": 2500,
                "LoanAmount": 95,
                "Gender": "Male",
            },
        ]
    )


page = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Prédiction"],
    index=0,
)

if page == "Accueil":
    data = load_dashboard_data()
    application_count = len(data)
    approval_rate = round(
        data["Loan_Status"].fillna("N").map({"Y": 1, "N": 0}).mean() * 100,
        1,
    )
    avg_loan = (
        int(data["LoanAmount"].mean())
        if "LoanAmount" in data.columns
        else 0
    )
    avg_income = (
        int(data["ApplicantIncome"].mean())
        if "ApplicantIncome" in data.columns
        else 0
    )

    with st.container():
        if Path("assets/logo.svg").exists():
            st.image("assets/logo.svg", width=90)
        st.markdown(
            "<div class='hero'>"
            "<h1>Prédiction de prêt bancaire</h1>"
            "<p style='font-size:1.15rem; line-height:1.8;'>"
            "Une application premium pour visualiser rapidement la "
            "probabilité d'acceptation d'un dossier de prêt."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    stat_col1, stat_col2, stat_col3 = st.columns(3, gap="large")
    stat_col1.metric("Demandes traitées", application_count)
    stat_col2.metric("Taux d'acceptation", f"{approval_rate}%")
    stat_col3.metric("Montant moyen", f"{avg_loan}k€")

    st.markdown("### Statistiques modèles")
    status_counts = data["Loan_Status"].fillna("N").replace(
        {"Y": "Approuvé", "N": "Refusé"}
    )
    st.bar_chart(status_counts.value_counts())

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("## Comment fonctionne l'application ?")
        st.markdown(
            "Cette application utilise un modèle de machine learning "
            "entraîné sur des données de prêt pour prédire si une demande "
            "est susceptible d'être acceptée."
        )
        st.markdown("### 1. Collecte des informations")
        st.markdown(
            "L'utilisateur saisit les informations suivantes : sexe, situation"
            "matrimoniale, revenu, montant du prêt, historique de crédit, et "
            "autres paramètres clés."
        )
        st.markdown("### 2. Prétraitement intelligent")
        st.markdown(
            "Les données sont nettoyées et transformées automatiquement "
            "avant d'être envoyées au modèle."
            " Les variables catégorielles sont encodées, les valeurs "
            "manquantes sont imputées, et les variables numériques sont "
            "normalisées."
        )
        st.markdown("### 3. Prédiction rapide")
        st.markdown(
            "Le modèle renvoie une prédiction immédiate : prêt approuvé "
            "ou prêt refusé."
        )
    with col2:
        st.markdown(
            "<div class='card'>"
            "<h3 class='section-title'>Points forts</h3>"
            "<ul style='line-height:1.8; padding-left: 1rem;'>"
            "<li><span class='highlight'>Interface claire</span> "
            "adaptée aux décideurs.</li>"
            "<li><span class='highlight'>Modèle robuste</span> "
            "avec RandomForest.</li>"
            "<li><span class='highlight'>Résultat instantané</span> "
            "pour chaque simulation.</li>"
            "<li><span class='highlight'>Design professionnel</span> "
            "et moderne.</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## Avantages métier")
    st.markdown(
        "- Facilite les décisions de crédit avec des estimations rapides.\n"
        "- Améliore la cohérence des décisions de prêt.\n"
        "- Réduit le temps d'analyse manuel des dossiers."
    )
    st.markdown("---")
    st.markdown("## Instructions")
    st.markdown(
        "1. Allez dans l'onglet **Prédiction**.\n"
        "2. Renseignez les informations client.\n"
        "3. Cliquez sur **Prédire** pour obtenir le résultat."
    )

else:
    st.markdown(
        "<div class='section-title'><h2>Simulation de prêt</h2></div>",
        unsafe_allow_html=True,
    )
    st.write(
        "Remplissez le formulaire ci-dessous pour estimer la décision de prêt."
    )

    left, right = st.columns((2, 1), gap="large")

    with left:
        with st.form("loan_form"):
            gender = st.selectbox(
                "Sexe",
                ["Male", "Female"],
                index=0,
            )
            married = st.selectbox(
                "Marié",
                ["Yes", "No"],
                index=0,
            )
            dependents = st.selectbox(
                "Personnes à charge",
                ["0", "1", "2", "3+"],
                index=0,
            )
            education = st.selectbox(
                "Éducation",
                ["Graduate", "Not Graduate"],
                index=0,
            )
            self_employed = st.selectbox(
                "Auto-entrepreneur",
                ["No", "Yes"],
                index=0,
            )
            applicant_income = st.number_input(
                "Revenu du demandeur",
                min_value=0,
                value=5000,
            )
            coapplicant_income = st.number_input(
                "Revenu du co-demandeur",
                min_value=0,
                value=0,
            )
            loan_amount = st.number_input(
                "Montant du prêt",
                min_value=0,
                value=100,
            )
            loan_amount_term = st.selectbox(
                "Durée du prêt (mois)",
                [360, 180, 240, 120],
                index=0,
            )
            credit_history = st.selectbox(
                "Historique de crédit",
                [1, 0],
                format_func=lambda x: "Bon" if x == 1 else "Mauvais",
            )
            property_area = st.selectbox(
                "Zone immobilière",
                ["Urban", "Rural", "Semiurban"],
                index=0,
            )

            submit_button = st.form_submit_button("Prédire")

        if submit_button:
            sample = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }

            try:
                result = predict_loan(sample)
                if result == "Y":
                    st.success("Prêt approuvé")
                else:
                    st.error("Prêt refusé")
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")

    with right:
        st.markdown(
            "<div class='card'>"
            "<h3 class='section-title'>Résumé</h3>"
            "<p>Utilisez ce panneau pour vérifier vos paramètres "
            "avant prédiction.</p>"
            "<ul style='line-height:1.8; padding-left:1rem;'>"
            "<li><strong>Sexe:</strong> Male / Female</li>"
            "<li><strong>Historique:</strong> Bon ou Mauvais</li>"
            "<li><strong>Montant du prêt:</strong> "
            "à définir selon le dossier</li>"
            "<li><strong>Durée:</strong> 120 à 360 mois</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

        if not BEST_MODEL.exists():
            st.warning(
                "Le modèle n'est pas encore créé. Exécutez `python main.py` "
                "depuis le répertoire racine pour générer "
                "`models/best_model.pkl`."
            )
        else:
            st.success("Modèle détecté et prêt pour les prédictions.")
