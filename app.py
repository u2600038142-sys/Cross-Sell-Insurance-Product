import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIG: Databricks Serving Endpoint
# ============================================================

# Recommended: store in .streamlit/secrets.toml:
# DATABRICKS_HOST = "https://dbc-xxxx.cloud.databricks.com"
# DATABRICKS_TOKEN = "dapixxxxx"
# ENDPOINT_NAME = "cross_sell_insurance_proba"  # adjust to your endpoint

DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST"))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
ENDPOINT_NAME = st.secrets.get("ENDPOINT_NAME", os.getenv("ENDPOINT_NAME", "cross_sell_insurance_proba"))

# ============================================================
# FEATURE SCHEMA – MUST MATCH ENDPOINT SIGNATURE
# ============================================================
# Format:
#   "column_name": (type, default_value)
#   type:
#       - "num"  : numeric (int/float, including counts / amounts)
#       - "cat"  : categorical / string
#       - "flag" : 0/1 via checkbox, sent as float 0.0/1.0

FEATURE_SCHEMA = {
    # ID
    "client_id": ("num", 1),

    # Demographic & FATCA
    "gender": ("cat", "M"),
    "marital_status": ("cat", "Single"),
    "black_list": ("cat", "NO"),
    "occupancy": ("cat", "IN24"),
    "fatca": ("cat", "NO"),
    "isfatcacrs": ("cat", "NO"),
    "fatca_indicia": ("cat", "NO"),

    # Age band flags (0/1)
    "le17": ("flag", 0),
    "18-21": ("flag", 0),
    "22-27": ("flag", 0),
    "28-34": ("flag", 0),
    "35-39": ("flag", 1),
    "40-49": ("flag", 0),
    "ge50": ("flag", 0),

    # Vintage & counts
    "customer_vintage_in_month": ("num", 24.0),
    "agency_counts": ("num", 1.0),
    "ecomm_counts": ("num", 1.0),
    "accident_counts": ("num", 0.0),
    "cl_counts": ("num", 0.0),
    "rpul_counts": ("num", 0.0),
    "spul_counts": ("num", 0.0),
    "traditional_counts": ("num", 1.0),

    # Payment channel flags
    "credit_card_payment": ("flag", 0),
    "bank_transfer_payment": ("flag", 1),
    "auto_debet_payment": ("flag", 0),

    # Payment frequency flags
    "yearly_payment": ("flag", 1),
    "halfyearly_payment": ("flag", 0),
    "quarterly_payment": ("flag", 0),
    "monthly_payment": ("flag", 0),
    "single_payment": ("flag", 0),

    # Premium & policy status
    "ape_sums": ("num", 15000000.0),
    "inforce_counts": ("num", 2.0),
    "lapse_counts": ("num", 0.0),
    "fwd_max_flag": ("flag", 1),

    # Interest / spending categories (0/1 flags)
    "automotive": ("flag", 0),
    "book_&_movie": ("flag", 0),
    "children": ("flag", 0),
    "culinary": ("flag", 1),
    "fashion": ("flag", 0),
    "gadget": ("flag", 0),
    "gasoline": ("flag", 0),
    "go-pay": ("flag", 0),
    "health": ("flag", 0),
    "home_expense": ("flag", 0),
    "music": ("flag", 0),
    "property": ("flag", 0),
    "shopping": ("flag", 0),
    "sport": ("flag", 0),
    "style": ("flag", 0),
    "travel": ("flag", 0),
}

# ============================================================
# Helper: call Databricks Serving Endpoint
# ============================================================

def call_databricks_endpoint(records):
    """
    Send a list of dicts (records) to Databricks Serving Endpoint
    and return a list of prediction scores (probabilities 0–1).

    records: list[dict] - each dict = 1 row of features
    """
    if DATABRICKS_HOST is None or DATABRICKS_TOKEN is None or ENDPOINT_NAME is None:
        raise RuntimeError(
            "DATABRICKS_HOST, DATABRICKS_TOKEN, and ENDPOINT_NAME are not set. "
            "Please configure them via .streamlit/secrets.toml or environment variables."
        )

    url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    payload = {
        "dataframe_records": df.to_dict(orient="records")
    }

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, headers=headers, json=payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Error from endpoint: {e}\nStatus code: {resp.status_code}\nBody: {resp.text}"
        )

    out = resp.json()
    preds = out.get("predictions", out)

    # If output is a list of dicts, pick 'prediction' if available or the first value
    if len(preds) > 0 and isinstance(preds[0], dict):
        scores = []
        for p in preds:
            if "prediction" in p:
                scores.append(p["prediction"])
            else:
                scores.append(list(p.values())[0])
    else:
        scores = preds

    return scores

# ============================================================
# Helper: marketing-friendly explanations
# ============================================================

def build_explanations(user_input, score, threshold=0.2):
    """
    Build a list of human-friendly explanation bullets
    based on feature values, for Marketing and Business users.

    This is not SHAP, but a rule-based narrative on top of the model.
    """
    explanations = []

    # 1. Age segment from band flags
    age_map = {
        "le17": "≤ 17 years",
        "18-21": "18–21 years",
        "22-27": "22–27 years",
        "28-34": "28–34 years",
        "35-39": "35–39 years",
        "40-49": "40–49 years",
        "ge50": "≥ 50 years",
    }
    active_age = [age_map[k] for k in age_map.keys() if float(user_input.get(k, 0) or 0) > 0]
    if active_age:
        explanations.append(
            f"• The customer is in the **{', '.join(active_age)}** age segment, "
            "which is relevant for additional protection and active lifestyle products."
        )

    # 2. Relationship with FWD and payment behaviour
    if float(user_input.get("fwd_max_flag", 0) or 0) > 0:
        explanations.append(
            "• The customer already has a strong relationship with FWD (maximum flag active), "
            "which makes cross-sell offers more natural."
        )
    if float(user_input.get("yearly_payment", 0) or 0) > 0:
        explanations.append(
            "• The customer is used to **yearly premium payments**, which fits products with higher premium amounts."
        )
    if float(user_input.get("monthly_payment", 0) or 0) > 0:
        explanations.append(
            "• The customer is used to **monthly payments**, so an affordable instalment plan can be attractive."
        )

    # 3. E-Commerce activity & payment channels
    if float(user_input.get("ecomm_counts", 0) or 0) > 0:
        explanations.append(
            "• The customer is already active in **E-Commerce**, which makes Bebas Aksi relevant "
            "as additional protection for online transactions and daily activities."
        )
    if float(user_input.get("bank_transfer_payment", 0) or 0) > 0 or float(user_input.get("credit_card_payment", 0) or 0) > 0:
        explanations.append(
            "• The customer already uses **modern payment channels** (bank transfer/credit card), "
            "so the purchase process for new products is frictionless."
        )

    # 4. Premium & number of policies
    ape = float(user_input.get("ape_sums", 0) or 0)
    inforce = float(user_input.get("inforce_counts", 0) or 0)
    if ape > 0:
        explanations.append(
            f"• The customer has a total APE of approximately **{ape:,.0f}**, "
            "indicating sufficient spending capacity for an additional product."
        )
    if inforce > 0:
        explanations.append(
            f"• The customer currently holds **{inforce:.0f} in-force policies**, "
            "showing familiarity with insurance products."
        )

    # 5. Interests & lifestyle
    interest_features = [
        ("travel", "travel & holidays"),
        ("sport", "sports"),
        ("health", "health & wellness"),
        ("children", "family & children"),
        ("automotive", "vehicles & automotive"),
        ("culinary", "food & lifestyle"),
        ("shopping", "shopping & lifestyle"),
        ("gadget", "gadgets & technology"),
    ]
    active_interests = [
        label
        for feat, label in interest_features
        if float(user_input.get(feat, 0) or 0) > 0
    ]
    if active_interests:
        explanations.append(
            "• The customer shows interest in **"
            + ", ".join(active_interests)
            + "**, which can be used as angles for personalised campaign content."
        )

    # 6. Score segment
    if score >= 0.5:
        explanations.append(
            "• The model classifies this customer as **High Potential** for Bebas Aksi cross-sell."
        )
    elif score >= threshold:
        explanations.append(
            "• The model classifies this customer as **Medium Potential**, "
            "suitable for lighter offers or follow-up campaigns."
        )
    else:
        explanations.append(
            "• The model classifies this customer as **Low Potential**, "
            "and they can be deprioritised in targeted campaigns."
        )

    return explanations

# ============================================================
# Key model drivers (feature importance summary)
# ============================================================

TOP_DRIVERS = {
    "Ecomm_Counts": (
        "Customers with more E-Commerce transactions tend to be more engaged "
        "digitally and are more likely to respond to Bebas Aksi as an add-on protection."
    ),
    "Occupancy_IN24": (
        "Customers with occupancy **IN24** (specific occupation segment) show a higher "
        "propensity to purchase Bebas Aksi, possibly due to their risk profile and lifestyle."
    ),
    "Yearly_Payment": (
        "Customers who pay premiums **yearly** are typically more committed and comfortable "
        "with larger lump-sum payments, making them good candidates for additional coverage."
    ),
    "Occupancy_SA33": (
        "Customers in occupancy **SA33** form a segment with above-average likelihood to "
        "buy Bebas Aksi, based on their historical behaviour in the portfolio."
    ),
    "Occupancy_IN23": (
        "Customers in occupancy **IN23** also contribute strongly to the model’s prediction, "
        "indicating this occupational segment is attractive for cross-sell."
    ),
}

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Bebas Aksi Cross-Sell Scoring", page_icon="📈")

st.title("📈 Bebas Aksi Cross-Sell Scoring Demo")

st.write(
    "This showcase takes a customer profile, sends it to a **Databricks Model Serving Endpoint**, "
    "and returns the **probability** that the customer will buy the **Bebas Aksi** product.\n\n"
    "The page is designed so both Data teams and Marketing teams can understand and use it."
)

# --- Key model drivers section (at the top) ---
st.markdown("### 🔑 Key Model Drivers")

st.info(
    "Based on the trained Logistic Regression model, the most important features driving the "
    "Bebas Aksi propensity score are:\n\n"
    "- **Ecomm_Counts**\n"
    "- **Occupancy_IN24**\n"
    "- **Yearly_Payment**\n"
    "- **Occupancy_SA33**\n"
    "- **Occupancy_IN23**"
)

selected_driver = st.selectbox(
    "Select a key driver to see a short business explanation:",
    list(TOP_DRIVERS.keys()),
)

st.markdown(f"**{selected_driver}**")
st.write(TOP_DRIVERS[selected_driver])

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Endpoint Configuration")
    st.text_input("Databricks Host", value=DATABRICKS_HOST or "", disabled=True)
    st.text_input("Endpoint Name", value=ENDPOINT_NAME or "", disabled=True)
    st.markdown("---")
    st.caption(
        "Host, token, and endpoint name are configured via Streamlit secrets "
        "or environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN, ENDPOINT_NAME)."
    )

# Check config
if not (DATABRICKS_HOST and DATABRICKS_TOKEN and ENDPOINT_NAME):
    st.error(
        "DATABRICKS_HOST, DATABRICKS_TOKEN, or ENDPOINT_NAME is not set.\n"
        "Please configure them in .streamlit/secrets.toml or as environment variables "
        "before running the app."
    )
    st.stop()

st.markdown("### 🧾 Customer Profile Input")

with st.form("input_form"):
    user_input = {}

    # Layout in 3 columns for readability
    col1, col2, col3 = st.columns(3)

    for feature, (ftype, default) in FEATURE_SCHEMA.items():
        label = feature.replace("_", " ").title()

        # Decide which column to place the field in
        if feature in [
            "client_id",
            "gender",
            "marital_status",
            "black_list",
            "occupancy",
            "fatca",
            "isfatcacrs",
            "fatca_indicia",
        ]:
            container = col1
        elif feature in [
            "customer_vintage_in_month",
            "agency_counts",
            "ecomm_counts",
            "ape_sums",
            "inforce_counts",
            "lapse_counts",
        ]:
            container = col2
        else:
            container = col3

        with container:
            if feature == "gender":
                val = st.selectbox(
                    label,
                    options=["M", "F"],
                    index=0 if default == "M" else 1,
                )
            elif feature == "marital_status":
                val = st.selectbox(
                    label,
                    options=["Single", "Married", "Other"],
                    index=0,
                )
            elif feature in ["black_list", "fatca", "isfatcacrs", "fatca_indicia"]:
                val = st.selectbox(
                    label,
                    options=["NO", "YES"],
                    index=0,
                )
            elif ftype == "num":
                try:
                    default_val = float(default)
                except Exception:
                    default_val = 0.0
                val = st.number_input(label, value=default_val)
            elif ftype == "flag":
                default_bool = bool(default)
                val_bool = st.checkbox(label, value=default_bool)
                val = 1.0 if val_bool else 0.0
            else:
                val = st.text_input(label, value=str(default))

        user_input[feature] = val

    submitted = st.form_submit_button("🚀 Calculate Probability")

if submitted:
    st.markdown("### 📊 Scoring Result")

    try:
        scores = call_databricks_endpoint([user_input])
        score = float(scores[0])

        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.metric(
                label="Probability the Customer Buys Bebas Aksi",
                value=f"{score:.2%}",
                help="This value comes from the model deployed in Databricks and represents a probability between 0 and 1.",
            )
            # Progress bar for quick visual impression
            st.progress(min(max(score, 0.0), 1.0))

        with col_side:
            # Simple segmentation
            if score >= 0.5:
                st.success("Segment: **High Potential**")
            elif score >= 0.2:
                st.info("Segment: **Medium Potential**")
            else:
                st.warning("Segment: **Low Potential**")

        st.markdown("---")
        st.subheader("💡 Explanations for Marketing")

        explanations = build_explanations(user_input, score, threshold=0.2)
        for exp in explanations:
            st.markdown(exp)

        st.markdown("---")
        st.subheader("🔍 Full Payload Sent to the Endpoint")
        st.json(user_input)

    except Exception as e:
        st.error(f"An error occurred while calling the endpoint:\n{e}")
