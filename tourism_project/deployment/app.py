# Streamlit frontend for the Tourism Package Prediction model
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

HF_USER = "iamsubha"
MODEL_REPO = f"{HF_USER}/tourism-package-model"
MODEL_FILE = "tourism_rf_model.joblib"

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🧳")

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)

model = load_model()

st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Fill in the customer details on the left and hit **Predict** to see "
    "whether they're likely to buy the package."
)

# --- input widgets ---
col1, col2 = st.columns(2)

with col1:
    age             = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier       = st.selectbox("City Tier", [1, 2, 3])
    occupation      = st.selectbox("Occupation",
                                   ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status",
                                   ["Single", "Married", "Divorced", "Unmarried"])
    designation     = st.selectbox("Designation",
                                   ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income  = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
    num_trips       = st.number_input("Trips per year", 0, 30, 3)

with col2:
    duration       = st.number_input("Duration of Pitch (mins)", 0.0, 200.0, 15.0)
    num_visiting   = st.number_input("People visiting", 1, 10, 2)
    num_followups  = st.number_input("Number of follow-ups", 0, 10, 3)
    product        = st.selectbox("Product Pitched",
                                  ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    pref_star      = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    passport       = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car        = st.selectbox("Owns a Car?", ["No", "Yes"])
    num_children   = st.number_input("Children visiting (< 5 yrs)", 0, 10, 0)
    pitch_score    = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            type_of_contact,
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration,
        "Occupation":               occupation,
        "Gender":                   gender,
        "NumberOfPersonVisiting":   num_visiting,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           product,
        "PreferredPropertyStar":    pref_star,
        "MaritalStatus":            marital_status,
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              designation,
        "MonthlyIncome":            monthly_income,
    }])

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])

    if pred == 1:
        st.success(f"✅ Likely to purchase the package (probability: {prob:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase (probability: {prob:.1%})")
