import os
import json
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
# ENDPOINT_NAME = "Lapse_Score_Insurance1"  # ganti sesuai endpoint kamu

DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST"))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
ENDPOINT_NAME = st.secrets.get("ENDPOINT_NAME", os.getenv("ENDPOINT_NAME", "Lapse_Score_Insurance1"))

# ============================================================
# FEATURE SCHEMA
# ============================================================
# NOTE PENTING:
# - Lengkapi FEATURE_SCHEMA ini agar SESUAI dengan kolom yang digunakan
#   waktu training di table:
#   cross_sell_insurance.01_feature_staging.stage2_clean_feature_table
# - Kolom target (is_target_customer) TIDAK usah dimasukkan.
#
# Format:
#   "nama_kolom": (tipe, default_value)
#   tipe:
#       - "num"  : numeric (int/float)
#       - "cat"  : categorical / string
#       - "bool" : checkbox yang di-convert ke 0/1
#
# Silakan menambah / mengurangi kolom sesuai kebutuhan.

FEATURE_SCHEMA = {
    # Contoh kolom numeric
    "insured_age": ("num", 35),
    "policy_counts": ("num", 3),
    "customer_vintage_in_month": ("num", 24),
    "ecomm_counts": ("num", 1),
    "ape_sums": ("num", 15000000),
    "inforce_counts": ("num", 2),
    "lapse_counts": ("num", 0),

    # Contoh kolom pembayaran (numerik flag 0/1)
    "yearly_payment": ("num", 1),
    "halfyearly_payment": ("num", 0),
    "quarterly_payment": ("num", 0),
    "monthly_payment": ("num", 0),
    "single_payment": ("num", 0),

    # Contoh kolom kategorikal
    "gender": ("cat", "M"),
    "marital_status": ("cat", "Single"),
    "black_list": ("cat", "NO"),
    "occupancy": ("cat", "IN24"),

    # Contoh preferensi minat (numerik flag 0/1)
    "automotive": ("num", 0),
    "children": ("num", 0),
    "culinary": ("num", 1),
    "fashion": ("num", 0),
    "gadget": ("num", 0),
    "health": ("num", 0),
    "travel": ("num", 0),

    # TODO: tambahkan kolom lain sesuai schema asli stage2_clean_feature_table
    # Misal:
    # "fatca": ("cat", "NO"),
    # "isfatcaacrs": ("cat", "NO"),
    # "fatca_indicia": ("cat", "NO"),
    # "credit_card_payment": ("num", 0),
    # "bank_transfer_payment": ("num", 0),
    # "auto_debet_payment": ("num", 0),
    # dst...
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
        label = feature.replace("_", " ").title()

        if ftype == "num":
            # Numeric input
            try:
                default_val = float(default)
            except Exception:
                default_val = 0.0
            val = st.number_input(label, value=default_val)
        elif ftype == "bool":
            # Checkbox -> 0/1
            default_bool = bool(default)
            val_bool = st.checkbox(label, value=default_bool)
            val = 1 if val_bool else 0
        else:
            # Categorical / string
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
