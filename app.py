import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIG: Databricks Serving Endpoint
# ============================================================

# Disarankan: simpan di .streamlit/secrets.toml:
# DATABRICKS_HOST = "https://dbc-xxxx.cloud.databricks.com"
# DATABRICKS_TOKEN = "dapixxxxx"
# ENDPOINT_NAME = "cross_sell_insurance"  # ganti sesuai endpoint kamu

DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST"))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
ENDPOINT_NAME = st.secrets.get("ENDPOINT_NAME", os.getenv("ENDPOINT_NAME", "cross_sell_insurance"))

# ============================================================
# FEATURE SCHEMA – HARUS MATCH DENGAN SIGNATURE ENDPOINT
# ============================================================
# Berdasarkan error message endpoint, input schema yang diharapkan:
# ['client_id': integer (required),
#  'gender': string (optional),
#  'marital_status': string (optional),
#  'black_list': string (required),
#  'occupancy': string (optional),
#  'fatca': string (required),
#  'isfatcacrs': string (required),
#  'fatca_indicia': string (required),
#  'le17': integer (required),
#  '18-21': integer (required),
#  '22-27': integer (required),
#  '28-34': integer (required),
#  '35-39': integer (required),
#  '40-49': integer (required),
#  'ge50': integer (required),
#  'customer_vintage_in_month': double (optional),
#  'agency_counts': double (optional),
#  'ecomm_counts': double (optional),
#  'accident_counts': double (optional),
#  'cl_counts': double (optional),
#  'rpul_counts': double (optional),
#  'spul_counts': double (optional),
#  'traditional_counts': double (optional),
#  'credit_card_payment': double (optional),
#  'bank_transfer_payment': double (optional),
#  'auto_debet_payment': double (optional),
#  'yearly_payment': double (optional),
#  'halfyearly_payment': double (optional),
#  'quarterly_payment': double (optional),
#  'monthly_payment': double (optional),
#  'single_payment': double (optional),
#  'ape_sums': double (optional),
#  'inforce_counts': double (optional),
#  'lapse_counts': double (optional),
#  'fwd_max_flag': double (optional),
#  'automotive': double (optional),
#  'book_&_movie': double (optional),
#  'children': double (optional),
#  'culinary': double (optional),
#  'fashion': double (optional),
#  'gadget': double (optional),
#  'gasoline': double (optional),
#  'go-pay': double (optional),
#  'health': double (optional),
#  'home_expense': double (optional),
#  'music': double (optional),
#  'property': double (optional),
#  'shopping': double (optional),
#  'sport': double (optional),
#  'style': double (optional),
#  'travel': double (optional)]
#
# Kita definisikan skema ini sebagai FEATURE_SCHEMA.
# Format:
#   "nama_kolom": (tipe, default_value)
#   tipe:
#       - "num"  : numeric (int/float, termasuk 0/1 flag)
#       - "cat"  : categorical / string

FEATURE_SCHEMA = {
    # ID
    "client_id": ("num", 1),

    # Demografi & FATCA
    "gender": ("cat", "M"),
    "marital_status": ("cat", "Single"),
    "black_list": ("cat", "NO"),
    "occupancy": ("cat", "IN24"),
    "fatca": ("cat", "NO"),
    "isfatcacrs": ("cat", "NO"),
    "fatca_indicia": ("cat", "NO"),

    # Age band flags (0/1)
    "le17": ("num", 0),
    "18-21": ("num", 0),
    "22-27": ("num", 0),
    "28-34": ("num", 0),
    "35-39": ("num", 1),
    "40-49": ("num", 0),
    "ge50": ("num", 0),

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
    "credit_card_payment": ("num", 0.0),
    "bank_transfer_payment": ("num", 1.0),
    "auto_debet_payment": ("num", 0.0),

    # Payment frequency flags
    "yearly_payment": ("num", 1.0),
    "halfyearly_payment": ("num", 0.0),
    "quarterly_payment": ("num", 0.0),
    "monthly_payment": ("num", 0.0),
    "single_payment": ("num", 0.0),

    # Premi & policy status
    "ape_sums": ("num", 15000000.0),
    "inforce_counts": ("num", 2.0),
    "lapse_counts": ("num", 0.0),
    "fwd_max_flag": ("num", 1.0),

    # Interest / spending categories
    "automotive": ("num", 0.0),
    "book_&_movie": ("num", 0.0),
    "children": ("num", 0.0),
    "culinary": ("num", 1.0),
    "fashion": ("num", 0.0),
    "gadget": ("num", 0.0),
    "gasoline": ("num", 0.0),
    "go-pay": ("num", 0.0),
    "health": ("num", 0.0),
    "home_expense": ("num", 0.0),
    "music": ("num", 0.0),
    "property": ("num", 0.0),
    "shopping": ("num", 0.0),
    "sport": ("num", 0.0),
    "style": ("num", 0.0),
    "travel": ("num", 0.0),
}

# ============================================================
# Helper: Panggil Databricks Serving Endpoint
# ============================================================

def call_databricks_endpoint(records):
    """Mengirim list of dict (records) ke Databricks Serving Endpoint
    dan mengembalikan list skor prediksi.

    records: list[dict] - tiap dict = 1 row fitur
    """
    if DATABRICKS_HOST is None or DATABRICKS_TOKEN is None or ENDPOINT_NAME is None:
        raise RuntimeError(
            "DATABRICKS_HOST, DATABRICKS_TOKEN, dan ENDPOINT_NAME belum diset. "
            "Set di .streamlit/secrets.toml atau environment variable."
        )

    url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

    # Bersihkan NaN / inf dan siapkan payload
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

    # Jika output list of dict, ambil key 'prediction' kalau ada, atau value pertama
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
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Bebas Aksi Cross-Sell Scoring", page_icon="📈")

st.title("📈 Bebas Aksi Cross-Sell Scoring Demo")
st.write(
    "Masukkan profil customer (feature sesuai schema model) dan sistem akan "
    "mengembalikan probabilitas pembelian produk **Bebas Aksi** dari model "
    "yang disajikan via Databricks Serving Endpoint."
)

with st.sidebar:
    st.header("⚙️ Endpoint Config")
    st.text_input("Databricks Host", value=DATABRICKS_HOST or "", disabled=True)
    st.text_input("Endpoint Name", value=ENDPOINT_NAME or "", disabled=True)
    st.markdown("---")
    st.caption(
        "Host, token, dan nama endpoint dikonfigurasi lewat Streamlit secrets "
        "atau environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN, ENDPOINT_NAME)."
    )

# Cek config dulu
if not (DATABRICKS_HOST and DATABRICKS_TOKEN and ENDPOINT_NAME):
    st.error(
        "DATABRICKS_HOST, DATABRICKS_TOKEN, atau ENDPOINT_NAME belum di-set.\n"
        "Set di .streamlit/secrets.toml atau environment variable sebelum menjalankan app."
    )
    st.stop()

st.markdown("### 🧾 Input Feature")

with st.form("input_form"):
    user_input = {}

    for feature, (ftype, default) in FEATURE_SCHEMA.items():
        # Label lebih rapih tapi key tetap nama kolom asli
        label = feature.replace("_", " ").title()

        if ftype == "num":
            try:
                default_val = float(default)
            except Exception:
                default_val = 0.0
            val = st.number_input(label, value=default_val)
        else:
            # categorical / string
            val = st.text_input(label, value=str(default))

        user_input[feature] = val

    submitted = st.form_submit_button("🚀 Predict")

if submitted:
    st.markdown("### 📊 Hasil Prediksi")

    try:
        scores = call_databricks_endpoint([user_input])
        score = float(scores[0])

        st.metric(
            label="Probabilitas Customer Membeli Bebas Aksi",
            value=f"{score:.4f}",
        )

        # Threshold contoh (bisa disesuaikan)
        threshold = 0.2
        is_target = score >= threshold

        st.write(f"**Threshold (contoh):** {threshold:.2f}")
        if is_target:
            st.success("✅ Customer ini termasuk **TARGET** untuk cross-sell Bebas Aksi.")
        else:
            st.warning("⚪ Customer ini **bukan prioritas utama** untuk cross-sell Bebas Aksi.")

        st.subheader("🔍 Payload yang dikirim ke endpoint")
        st.json(user_input)

    except Exception as e:
        st.error(f"Terjadi error saat memanggil endpoint:\n{e}")
