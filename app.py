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
# ENDPOINT_NAME = "cross_sell_insurance_proba"  # ganti sesuai endpoint kamu

DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST"))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
ENDPOINT_NAME = st.secrets.get("ENDPOINT_NAME", os.getenv("ENDPOINT_NAME", "cross_sell_insurance_proba"))

# ============================================================
# FEATURE SCHEMA – HARUS MATCH DENGAN SIGNATURE ENDPOINT
# ============================================================
# Format:
#   "nama_kolom": (tipe, default_value)
#   tipe:
#       - "num"  : numeric (int/float, termasuk count / amount)
#       - "cat"  : categorical / string
#       - "flag" : 0/1 yang diinput via checkbox, dikirim sebagai float 0.0/1.0

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

    # Premi & policy status
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
# Helper: panggil Databricks Serving Endpoint
# ============================================================

def call_databricks_endpoint(records):
    """
    Mengirim list of dict (records) ke Databricks Serving Endpoint
    dan mengembalikan list skor prediksi (probability 0–1).

    records: list[dict] - tiap dict = 1 row fitur
    """
    if DATABRICKS_HOST is None or DATABRICKS_TOKEN is None or ENDPOINT_NAME is None:
        raise RuntimeError(
            "DATABRICKS_HOST, DATABRICKS_TOKEN, dan ENDPOINT_NAME belum diset. "
            "Set di .streamlit/secrets.toml atau environment variable."
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
# Helper: bangun penjelasan marketing-friendly dari input
# ============================================================

def build_explanations(user_input, score, threshold=0.2):
    """
    Membuat list penjelasan (string) berbasis feature-value
    untuk ditampilkan ke tim Marketing.

    Ini bukan SHAP teknis, tapi rule-based yang human-friendly.
    """
    explanations = []

    # 1. Segment usia dari age band flags
    age_map = {
        "le17": "≤ 17 tahun",
        "18-21": "18–21 tahun",
        "22-27": "22–27 tahun",
        "28-34": "28–34 tahun",
        "35-39": "35–39 tahun",
        "40-49": "40–49 tahun",
        "ge50": "≥ 50 tahun",
    }
    active_age = [age_map[k] for k in age_map.keys() if float(user_input.get(k, 0) or 0) > 0]
    if active_age:
        explanations.append(
            f"• Customer berada di segmen usia **{', '.join(active_age)}**, "
            "yang relevan untuk kebutuhan proteksi dan gaya hidup aktif."
        )

    # 2. Relasi dengan FWD dan pola pembayaran
    if float(user_input.get("fwd_max_flag", 0) or 0) > 0:
        explanations.append(
            "• Customer sudah memiliki relasi kuat dengan FWD (flag maksimum aktif), "
            "sehingga lebih mudah untuk penawaran cross-sell."
        )
    if float(user_input.get("yearly_payment", 0) or 0) > 0:
        explanations.append(
            "• Customer terbiasa dengan **pembayaran tahunan**, yang cocok untuk produk dengan premi lebih besar."
        )
    if float(user_input.get("monthly_payment", 0) or 0) > 0:
        explanations.append(
            "• Customer terbiasa dengan **pembayaran bulanan**, sehingga bisa ditawarkan skema cicilan yang ringan."
        )

    # 3. Aktivitas E-Commerce & channel pembayaran
    if float(user_input.get("ecomm_counts", 0) or 0) > 0:
        explanations.append(
            "• Customer sudah aktif di **E-Commerce**, sehingga penawaran Bebas Aksi "
            "bisa relevan sebagai proteksi tambahan saat bertransaksi online."
        )
    if float(user_input.get("bank_transfer_payment", 0) or 0) > 0 or float(user_input.get("credit_card_payment", 0) or 0) > 0:
        explanations.append(
            "• Customer sudah terbiasa menggunakan **channel pembayaran modern** "
            "(bank transfer/kartu kredit), memudahkan proses pembelian produk."
        )

    # 4. Premi & jumlah polis
    ape = float(user_input.get("ape_sums", 0) or 0)
    inforce = float(user_input.get("inforce_counts", 0) or 0)
    if ape > 0:
        explanations.append(
            f"• Customer memiliki total APE sekitar **{ape:,.0f}**, "
            "menunjukkan daya beli yang cukup untuk produk tambahan."
        )
    if inforce > 0:
        explanations.append(
            f"• Customer memiliki **{inforce:.0f} polis aktif**, "
            "sehingga sudah familiar dengan produk asuransi."
        )

    # 5. Minat & gaya hidup
    interest_features = [
        ("travel", "perjalanan & travelling"),
        ("sport", "olahraga"),
        ("health", "kesehatan & wellness"),
        ("children", "keluarga & anak"),
        ("automotive", "kendaraan & otomotif"),
        ("culinary", "kuliner & gaya hidup"),
        ("shopping", "belanja & lifestyle"),
        ("gadget", "gadget & teknologi"),
    ]
    active_interests = [
        label
        for feat, label in interest_features
        if float(user_input.get(feat, 0) or 0) > 0
    ]
    if active_interests:
        explanations.append(
            "• Customer menunjukkan minat pada **"
            + ", ".join(active_interests)
            + "**, yang bisa dijadikan angle komunikasi kampanye."
        )

    # 6. Segmentasi skor
    if score >= 0.5:
        explanations.append(
            "• Model mengklasifikasikan customer ini sebagai **High Potential** untuk cross-sell Bebas Aksi."
        )
    elif score >= threshold:
        explanations.append(
            "• Model mengklasifikasikan customer ini sebagai **Medium Potential**; "
            "cocok untuk kampanye dengan penawaran yang lebih ringan."
        )
    else:
        explanations.append(
            "• Model mengklasifikasikan customer ini sebagai **Low Potential**; "
            "bisa diprioritaskan lebih rendah dalam kampanye massal."
        )

    return explanations

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Bebas Aksi Cross-Sell Scoring", page_icon="📈")

st.title("📈 Bebas Aksi Cross-Sell Scoring Demo")
st.write(
    "Showcase ini mengambil profil customer, mengirimnya ke **Databricks Model Serving**, "
    "dan mengembalikan **probabilitas** bahwa customer akan membeli produk **Bebas Aksi**.\n\n"
    "Halaman ini dirancang agar dapat dipahami baik oleh tim data maupun tim Marketing."
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

st.markdown("### 🧾 Input Profil Customer")

with st.form("input_form"):
    user_input = {}

    # Form dibuat dalam 3 kolom agar lebih rapi
    col1, col2, col3 = st.columns(3)

    for feature, (ftype, default) in FEATURE_SCHEMA.items():
        label = feature.replace("_", " ").title()

        # Pilih kolom untuk field ini (supaya tersebar)
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

    submitted = st.form_submit_button("🚀 Hitung Probabilitas")

if submitted:
    st.markdown("### 📊 Hasil Skoring")

    try:
        scores = call_databricks_endpoint([user_input])
        score = float(scores[0])

        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.metric(
                label="Probabilitas Customer Membeli Bebas Aksi",
                value=f"{score:.2%}",
                help="Nilai ini berasal dari model di Databricks yang mengembalikan probabilitas (0–1).",
            )
            # Progress bar sebagai visual
            st.progress(min(max(score, 0.0), 1.0))

        with col_side:
            # Segmentasi sederhana
            if score >= 0.5:
                st.success("Segmen: **High Potential**")
            elif score >= 0.2:
                st.info("Segmen: **Medium Potential**")
            else:
                st.warning("Segmen: **Low Potential**")

        st.markdown("---")
        st.subheader("💡 Penjelasan untuk Tim Marketing")

        explanations = build_explanations(user_input, score, threshold=0.2)
        for exp in explanations:
            st.markdown(exp)

        st.markdown("---")
        st.subheader("🔍 Payload Lengkap yang Dikirim ke Endpoint")
        st.json(user_input)

    except Exception as e:
        st.error(f"Terjadi error saat memanggil endpoint:\n{e}")
