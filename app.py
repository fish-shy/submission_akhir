"""
Jaya Jaya Institut — Student Dropout Early Warning System
=========================================================
Prototype sistem machine learning untuk memprediksi risiko dropout mahasiswa.

Menjalankan aplikasi:
    streamlit run app.py

Aplikasi memuat pipeline terlatih (model/dropout_model.joblib) beserta "kontrak"-nya
(model/model_metadata.json) yang dihasilkan notebook.ipynb, sehingga tidak ada logika
preprocessing yang perlu diduplikasi di sini.

Author : Hafiz Nazwa Nugraha (Dicoding ID: FIshiwak)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Konfigurasi halaman
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dropout Early Warning — Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "dropout_model.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

# Palet warna — sama persis dengan yang dipakai di notebook.
BLUE, INK, INK_SOFT, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BAND_COLORS = {
    "Rendah": "#0ca30c",
    "Sedang": "#fab219",
    "Tinggi": "#ec835a",
    "Sangat Tinggi": "#d03b3b",
}
BAND_ICONS = {"Rendah": "✅", "Sedang": "🟡", "Tinggi": "🟠", "Sangat Tinggi": "🔴"}

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .hero {
        background: linear-gradient(120deg, #123a68 0%, #2a78d6 100%);
        color: #ffffff; padding: 1.6rem 1.9rem; border-radius: 14px; margin-bottom: 1.6rem;
    }
    .hero h1 {color:#ffffff; font-size: 1.65rem; margin: 0 0 .35rem 0;}
    .hero p  {color: #d8e6f8; margin: 0; font-size: .95rem; line-height: 1.5;}
    .result-card {
        border-radius: 14px; padding: 1.5rem 1.7rem; color: #ffffff; margin-bottom: 1rem;
    }
    .result-card .score {font-size: 3.1rem; font-weight: 700; line-height: 1; margin: .2rem 0;}
    .result-card .band  {font-size: 1.05rem; font-weight: 600; letter-spacing: .02em;}
    .result-card .note  {font-size: .92rem; opacity: .92; margin-top: .55rem; line-height: 1.5;}
    .flag {
        border-left: 4px solid #d03b3b; background: #fdf2f2; padding: .6rem .9rem;
        border-radius: 6px; margin-bottom: .45rem; font-size: .9rem; color: #0b0b0b;
    }
    .flag-ok {border-left-color: #0ca30c; background: #f1faf1;}
    .meter-track {background:#e9e9e4; border-radius: 99px; height: 13px; width: 100%; margin-top:.4rem;}
    .meter-fill  {height: 13px; border-radius: 99px;}
    .caption {color: #898781; font-size: .84rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .stTabs [data-baseweb="tab-list"] {gap: .35rem;}
    .stTabs [data-baseweb="tab"] {padding: .55rem 1.05rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pemuatan artefak
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "Berkas model tidak ditemukan. Jalankan seluruh sel `notebook.ipynb` terlebih dahulu "
        "agar `model/dropout_model.joblib` dan `model/model_metadata.json` terbentuk."
    )
    st.stop()

model, META = load_artifacts()
FEATURES = META["features"]
NUMERIC = META["numeric_features"]
BINARY = META["binary_features"]
CATEGORICAL = META["categorical_features"]
THRESHOLD = META["threshold"]
BANDS = META["risk_bands"]
LABELS = META["labels"]
DEFAULTS = META["defaults"]
RANGES = META["ranges"]


def risk_band(p: float) -> dict:
    """Kembalikan definisi kategori risiko untuk sebuah probabilitas."""
    for band in BANDS:
        if p < band["max"]:
            return band
    return BANDS[-1]


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prediksi satu atau banyak baris; kembalikan probabilitas, label, dan kategori."""
    proba = model.predict_proba(frame[FEATURES])[:, 1]
    out = pd.DataFrame({
        "skor_risiko": proba.round(4),
        "prediksi": np.where(proba >= THRESHOLD, "Berisiko dropout", "Aman"),
        "kategori_risiko": [risk_band(p)["label"] for p in proba],
        "tindakan_disarankan": [risk_band(p)["action"] for p in proba],
    })
    return out


def label_options(column: str):
    """Opsi (kode, label) untuk sebuah kolom kategorikal, urut berdasarkan label."""
    mapping = LABELS[column]
    return sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[1])


def warning_flags(row: dict) -> list[tuple[bool, str]]:
    """Indikator peringatan berbasis temuan EDA — penjelas, bukan isi model."""
    s1_enrolled = max(row["Curricular_units_1st_sem_enrolled"], 1)
    s2_enrolled = max(row["Curricular_units_2nd_sem_enrolled"], 1)
    ratio1 = row["Curricular_units_1st_sem_approved"] / s1_enrolled
    ratio2 = row["Curricular_units_2nd_sem_approved"] / s2_enrolled
    return [
        (row["Tuition_fees_up_to_date"] == 0,
         "SPP belum lunas — 86,6% mahasiswa dengan tunggakan SPP berakhir dropout"),
        (row["Debtor"] == 1,
         "Tercatat punya tunggakan (debtor) — dropout rate kelompok ini 62,0%"),
        (ratio2 < 0.5,
         f"Hanya {ratio2:.0%} mata kuliah semester 2 yang lulus (di bawah setengah beban studi)"),
        (ratio1 < 0.5,
         f"Hanya {ratio1:.0%} mata kuliah semester 1 yang lulus (di bawah setengah beban studi)"),
        (row["Curricular_units_2nd_sem_grade"] < 10 and row["Curricular_units_2nd_sem_enrolled"] > 0,
         "Nilai rata-rata semester 2 di bawah 10 dari skala 20"),
        (row["Scholarship_holder"] == 0,
         "Bukan penerima beasiswa — dropout rate 38,7% vs 12,2% pada penerima beasiswa"),
        (row["Age_at_enrollment"] > 23,
         "Mendaftar di atas usia 23 tahun — kelompok ini dropout di kisaran 51–58%"),
    ]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🎓 Jaya Jaya Institut")
    st.caption("Student Dropout Early Warning System")
    page = st.radio(
        "Menu",
        ["Prediksi Individu", "Prediksi Massal (CSV)", "Tentang Model"],
        label_visibility="collapsed",
    )
    st.divider()
    m = META["metrics_test"]
    st.markdown("**Performa model pada data uji**")
    st.markdown(
        f"- Recall (dropout terdeteksi): **{m['recall']:.1%}**\n"
        f"- Precision: **{m['precision']:.1%}**\n"
        f"- F1-score: **{m['f1']:.3f}**\n"
        f"- ROC-AUC: **{m['roc_auc']:.3f}**"
    )
    st.divider()
    st.caption(
        f"Model: {META['model_name']}  \n"
        f"Ambang risiko: {THRESHOLD:.2f}  \n"
        f"Dilatih: {META['trained_at']}"
    )


# --------------------------------------------------------------------------- #
# Halaman 1 — Prediksi Individu
# --------------------------------------------------------------------------- #
if page == "Prediksi Individu":
    st.markdown(
        """
        <div class="hero">
            <h1>Deteksi Dini Risiko Dropout</h1>
            <p>Isi data akademik dan administratif mahasiswa, lalu sistem akan memperkirakan
            peluang mahasiswa tersebut berhenti kuliah beserta tindakan yang disarankan.
            Gunakan hasilnya sebagai <b>daftar prioritas siapa yang perlu ditemui lebih dulu</b> —
            bukan sebagai vonis atas masa depan mahasiswa.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("form_mahasiswa"):
        tab_profil, tab_sem1, tab_sem2 = st.tabs(
            ["👤 Profil & Administrasi", "📘 Semester 1", "📗 Semester 2"])
        values: dict[str, float | int] = {}

        with tab_profil:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Identitas**")
                nama = st.text_input("Nama mahasiswa (opsional)", placeholder="mis. Andi Pratama")
                values["Age_at_enrollment"] = st.number_input(
                    "Usia saat mendaftar", min_value=int(RANGES["Age_at_enrollment"]["min"]),
                    max_value=int(RANGES["Age_at_enrollment"]["max"]),
                    value=int(DEFAULTS["Age_at_enrollment"]), step=1)
                gender_opts = label_options("Gender")
                values["Gender"] = st.selectbox(
                    "Jenis kelamin", [k for k, _ in gender_opts],
                    format_func=lambda k: dict(gender_opts)[k],
                    index=[k for k, _ in gender_opts].index(int(DEFAULTS["Gender"])))
                marital_opts = label_options("Marital_status")
                values["Marital_status"] = st.selectbox(
                    "Status pernikahan", [k for k, _ in marital_opts],
                    format_func=lambda k: dict(marital_opts)[k],
                    index=[k for k, _ in marital_opts].index(int(DEFAULTS["Marital_status"])))

            with c2:
                st.markdown("**Program studi & jalur masuk**")
                course_opts = label_options("Course")
                values["Course"] = st.selectbox(
                    "Program studi", [k for k, _ in course_opts],
                    format_func=lambda k: dict(course_opts)[k],
                    index=[k for k, _ in course_opts].index(int(DEFAULTS["Course"])))
                mode_opts = label_options("Application_mode")
                values["Application_mode"] = st.selectbox(
                    "Jalur masuk", [k for k, _ in mode_opts],
                    format_func=lambda k: dict(mode_opts)[k],
                    index=[k for k, _ in mode_opts].index(int(DEFAULTS["Application_mode"])))
                values["Application_order"] = st.number_input(
                    "Urutan pilihan prodi saat mendaftar (0 = pilihan pertama)",
                    min_value=int(RANGES["Application_order"]["min"]),
                    max_value=int(RANGES["Application_order"]["max"]),
                    value=int(DEFAULTS["Application_order"]), step=1)
                att_opts = label_options("Daytime_evening_attendance")
                values["Daytime_evening_attendance"] = st.selectbox(
                    "Waktu perkuliahan", [k for k, _ in att_opts],
                    format_func=lambda k: dict(att_opts)[k],
                    index=[k for k, _ in att_opts].index(
                        int(DEFAULTS["Daytime_evening_attendance"])))

            with c3:
                st.markdown("**Keuangan & bekal akademik**")
                tuition_opts = label_options("Tuition_fees_up_to_date")
                values["Tuition_fees_up_to_date"] = st.selectbox(
                    "Status pembayaran SPP", [k for k, _ in tuition_opts],
                    format_func=lambda k: dict(tuition_opts)[k],
                    index=[k for k, _ in tuition_opts].index(
                        int(DEFAULTS["Tuition_fees_up_to_date"])))
                debtor_opts = label_options("Debtor")
                values["Debtor"] = st.selectbox(
                    "Punya tunggakan ke institusi?", [k for k, _ in debtor_opts],
                    format_func=lambda k: dict(debtor_opts)[k],
                    index=[k for k, _ in debtor_opts].index(int(DEFAULTS["Debtor"])))
                scholar_opts = label_options("Scholarship_holder")
                values["Scholarship_holder"] = st.selectbox(
                    "Penerima beasiswa?", [k for k, _ in scholar_opts],
                    format_func=lambda k: dict(scholar_opts)[k],
                    index=[k for k, _ in scholar_opts].index(int(DEFAULTS["Scholarship_holder"])))
                disp_opts = label_options("Displaced")
                values["Displaced"] = st.selectbox(
                    "Mahasiswa perantau/displaced?", [k for k, _ in disp_opts],
                    format_func=lambda k: dict(disp_opts)[k],
                    index=[k for k, _ in disp_opts].index(int(DEFAULTS["Displaced"])))
                values["Admission_grade"] = st.slider(
                    "Nilai masuk (skala 0–200)", min_value=0.0, max_value=200.0,
                    value=float(DEFAULTS["Admission_grade"]), step=0.1)
                values["Previous_qualification_grade"] = st.slider(
                    "Nilai pendidikan sebelumnya (skala 0–200)", min_value=0.0, max_value=200.0,
                    value=float(DEFAULTS["Previous_qualification_grade"]), step=0.1)

        for tab, sem, prefix in ((tab_sem1, "1st", "📘"), (tab_sem2, "2nd", "📗")):
            with tab:
                st.markdown(f"**Hasil studi semester {'1' if sem == '1st' else '2'}**")
                st.caption(
                    "Isi 0 pada seluruh kolom bila semester tersebut belum dijalani. "
                    "Perlu diingat, akurasi model paling tinggi setelah nilai semester tersedia.")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    values[f"Curricular_units_{sem}_sem_enrolled"] = st.number_input(
                        "Mata kuliah diambil", min_value=0, max_value=30,
                        value=int(DEFAULTS[f"Curricular_units_{sem}_sem_enrolled"]), step=1,
                        key=f"enr_{sem}")
                with c2:
                    values[f"Curricular_units_{sem}_sem_approved"] = st.number_input(
                        "Mata kuliah lulus", min_value=0, max_value=30,
                        value=int(DEFAULTS[f"Curricular_units_{sem}_sem_approved"]), step=1,
                        key=f"app_{sem}")
                with c3:
                    values[f"Curricular_units_{sem}_sem_evaluations"] = st.number_input(
                        "Jumlah evaluasi/ujian", min_value=0, max_value=50,
                        value=int(DEFAULTS[f"Curricular_units_{sem}_sem_evaluations"]), step=1,
                        key=f"eva_{sem}")
                with c4:
                    values[f"Curricular_units_{sem}_sem_grade"] = st.number_input(
                        "Nilai rata-rata (0–20)", min_value=0.0, max_value=20.0,
                        value=float(DEFAULTS[f"Curricular_units_{sem}_sem_grade"]), step=0.1,
                        key=f"grd_{sem}")

        st.markdown("")
        submitted = st.form_submit_button("🔍 Prediksi Risiko Dropout", type="primary",
                                          width="stretch")

    if submitted:
        row = pd.DataFrame([values])[FEATURES]
        result = predict_frame(row).iloc[0]
        proba = float(result["skor_risiko"])
        band = risk_band(proba)
        color = BAND_COLORS[band["label"]]

        st.markdown("---")
        left, right = st.columns([1, 1.25])

        with left:
            st.markdown(
                f"""
                <div class="result-card" style="background:{color};">
                    <div class="band">{BAND_ICONS[band['label']]} RISIKO {band['label'].upper()}</div>
                    <div class="score">{proba:.1%}</div>
                    <div>probabilitas dropout{(' — ' + nama) if nama else ''}</div>
                    <div class="meter-track" style="background: rgba(255,255,255,.28);">
                        <div class="meter-fill" style="width:{max(proba, 0.02) * 100:.1f}%;
                             background:#ffffff;"></div>
                    </div>
                    <div class="note"><b>Tindakan disarankan:</b> {band['action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Mahasiswa ditandai **berisiko** bila skornya ≥ {THRESHOLD:.2f}. Ambang ini "
                f"sengaja ditetapkan rendah agar sedikit mungkin calon dropout yang terlewat "
                f"(recall {META['metrics_test']['recall']:.0%} pada data uji)."
            )

        with right:
            st.markdown("#### Indikator yang terdeteksi")
            flags = [(cond, text) for cond, text in warning_flags(values)]
            active = [t for c, t in flags if c]
            if active:
                for text in active:
                    st.markdown(f"<div class='flag'>⚠️ {text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='flag flag-ok'>✅ Tidak ada indikator risiko utama yang terdeteksi "
                    "pada mahasiswa ini.</div>", unsafe_allow_html=True)
            st.markdown(
                "<p class='caption'>Indikator di atas berasal dari temuan analisis data, bukan "
                "dari perhitungan internal model. Fungsinya membantu konselor memahami "
                "<i>mengapa</i> seorang mahasiswa perlu ditemui.</p>", unsafe_allow_html=True)

        st.markdown("#### Ringkasan input")
        ringkas = pd.DataFrame({
            "Aspek": ["Program studi", "Usia saat mendaftar", "Status SPP", "Beasiswa",
                      "MK lulus semester 1", "MK lulus semester 2", "Nilai rata-rata semester 2"],
            "Nilai": [
                LABELS["Course"][str(values["Course"])],
                f"{values['Age_at_enrollment']} tahun",
                LABELS["Tuition_fees_up_to_date"][str(values["Tuition_fees_up_to_date"])],
                LABELS["Scholarship_holder"][str(values["Scholarship_holder"])],
                f"{values['Curricular_units_1st_sem_approved']} dari "
                f"{values['Curricular_units_1st_sem_enrolled']} MK",
                f"{values['Curricular_units_2nd_sem_approved']} dari "
                f"{values['Curricular_units_2nd_sem_enrolled']} MK",
                f"{values['Curricular_units_2nd_sem_grade']:.1f} / 20",
            ],
        })
        st.dataframe(ringkas, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Halaman 2 — Prediksi Massal
# --------------------------------------------------------------------------- #
elif page == "Prediksi Massal (CSV)":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediksi Massal via CSV</h1>
            <p>Unggah data satu angkatan atau satu program studi sekaligus, lalu unduh hasilnya
            sebagai daftar prioritas penanganan yang sudah terurut dari yang paling genting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Format berkas yang dibutuhkan", expanded=False):
        st.markdown(
            f"Berkas CSV harus memuat **{len(FEATURES)} kolom** berikut (nama kolom persis sama "
            "seperti pada dataset asli). Kolom tambahan seperti `student_id` atau `nama` "
            "dibiarkan apa adanya dan ikut muncul di hasil.")
        st.code(", ".join(FEATURES), language="text")
        template = pd.DataFrame([DEFAULTS])[FEATURES]
        template.insert(0, "student_id", "JJI-00001")
        st.download_button(
            "⬇️ Unduh template CSV", template.to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_dropout.csv", mime="text/csv")

    uploaded = st.file_uploader("Unggah berkas CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            if len(raw.columns) == 1 and ";" in str(raw.columns[0]):
                uploaded.seek(0)
                raw = pd.read_csv(uploaded, sep=";")
        except Exception as exc:  # pragma: no cover - bergantung pada berkas pengguna
            st.error(f"Berkas gagal dibaca: {exc}")
            st.stop()

        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            st.error("Kolom berikut belum ada pada berkas Anda: " + ", ".join(missing))
            st.stop()

        preds = predict_frame(raw)
        hasil = pd.concat([raw.reset_index(drop=True), preds], axis=1)
        hasil = hasil.sort_values("skor_risiko", ascending=False).reset_index(drop=True)

        n = len(hasil)
        n_risk = int((hasil["skor_risiko"] >= THRESHOLD).sum())
        n_critical = int((hasil["kategori_risiko"] == "Sangat Tinggi").sum())

        st.markdown("### Ringkasan")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mahasiswa diproses", f"{n:,}")
        c2.metric("Ditandai berisiko", f"{n_risk:,}", f"{n_risk / n:.1%} dari total")
        c3.metric("Kategori Sangat Tinggi", f"{n_critical:,}",
                  "perlu intervensi segera", delta_color="inverse")
        c4.metric("Rata-rata skor risiko", f"{hasil['skor_risiko'].mean():.1%}")

        dist = (hasil["kategori_risiko"].value_counts()
                .reindex([b["label"] for b in BANDS]).fillna(0).astype(int))
        st.markdown("#### Sebaran kategori risiko")
        st.bar_chart(dist, color=BLUE, height=260)

        st.markdown("#### Daftar prioritas penanganan")
        pilihan = st.multiselect(
            "Tampilkan kategori", [b["label"] for b in BANDS],
            default=["Sangat Tinggi", "Tinggi"])
        tampil = hasil[hasil["kategori_risiko"].isin(pilihan)] if pilihan else hasil

        kolom_utama = [c for c in ["student_id", "nama", "Course"] if c in tampil.columns]
        kolom_tampil = kolom_utama + ["skor_risiko", "kategori_risiko", "prediksi",
                                      "tindakan_disarankan"]
        st.dataframe(
            tampil[kolom_tampil].style.format({"skor_risiko": "{:.1%}"}),
            hide_index=True, width="stretch", height=420)

        buffer = io.StringIO()
        hasil.to_csv(buffer, index=False)
        st.download_button(
            "⬇️ Unduh seluruh hasil prediksi (CSV)", buffer.getvalue().encode("utf-8"),
            file_name="hasil_prediksi_dropout.csv", mime="text/csv", type="primary")
    else:
        st.info(
            "Belum ada berkas yang diunggah. Untuk mencoba, gunakan `students_dashboard.csv` "
            "hasil notebook atau unduh template pada panel di atas.")

# --------------------------------------------------------------------------- #
# Halaman 3 — Tentang Model
# --------------------------------------------------------------------------- #
else:
    st.markdown(
        """
        <div class="hero">
            <h1>Tentang Model</h1>
            <p>Cara kerja sistem, seberapa akurat, dan — yang sama pentingnya —
            di mana batas kemampuannya.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m = META["metrics_test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{m['recall']:.1%}", "calon dropout yang tertangkap")
    c2.metric("Precision", f"{m['precision']:.1%}", "ketepatan peringatan")
    c3.metric("F1-score", f"{m['f1']:.3f}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}", "kemampuan memeringkat risiko")

    st.markdown(
        f"""
        #### Bagaimana model ini dibangun

        - **Data:** {META['n_train'] + META['n_test']:,} catatan mahasiswa Jaya Jaya Institut,
          dibagi {META['n_train']:,} untuk melatih dan {META['n_test']:,} untuk menguji.
          Dari seluruh mahasiswa, {META['dropout_rate_dataset']:.1%} berakhir dropout.
        - **Algoritma:** **{META['model_name']}** dengan parameter `{META['best_params']}`,
          terpilih setelah dibandingkan dengan Random Forest dan Gradient Boosting melalui
          5-fold cross-validation.
        - **Fitur:** {len(FEATURES)} kolom — hasil studi dua semester pertama, status keuangan,
          serta profil dan jalur masuk mahasiswa.
        - **Ambang keputusan:** {THRESHOLD:.2f} (bukan 0,50 bawaan). {META['threshold_policy']},
          karena melewatkan satu mahasiswa jauh lebih mahal daripada satu sesi konseling tambahan.
        """
    )

    st.markdown("#### Faktor paling berpengaruh")
    top = pd.Series(META["top_features"]).sort_values(ascending=False)
    st.bar_chart(top, color=BLUE, horizontal=True, height=320)
    st.caption(
        "Diukur dengan permutation importance: seberapa jauh performa model turun ketika nilai "
        "sebuah kolom diacak. Hasil studi semester berjalan mendominasi, jauh di atas atribut "
        "bawaan mahasiswa seperti jenis kelamin atau jalur masuk."
    )

    st.markdown("#### Kategori risiko dan tindak lanjutnya")
    st.dataframe(
        pd.DataFrame([{
            "Kategori": b["label"],
            "Rentang skor": f"{b['min']:.0%} – {b['max']:.0%}",
            "Tindakan disarankan": b["action"],
        } for b in BANDS]),
        hide_index=True, width="stretch")

    st.markdown(
        """
        #### Batasan yang perlu diketahui pengguna

        1. **Model memperkirakan risiko, bukan memastikan masa depan.** Sekitar 1 dari 4 mahasiswa
           yang ditandai berisiko sebenarnya akan baik-baik saja — konsekuensi yang memang sengaja
           diterima demi menekan jumlah mahasiswa yang terlewat.
        2. **Akurasi bergantung pada ketersediaan nilai semester.** Untuk mahasiswa baru yang belum
           punya riwayat akademik, keluarannya perlu diperlakukan sebagai perkiraan kasar.
        3. **Hubungan, bukan sebab-akibat.** Model menemukan bahwa mahasiswa dengan tunggakan SPP
           hampir selalu berakhir keluar; ini bukan berarti melunasi SPP dengan sendirinya mencegah
           dropout.
        4. **Gunakan sebagai pemicu percakapan, bukan label.** Skor sebaiknya dipakai untuk memutuskan
           siapa yang perlu ditanyai kabarnya lebih dulu. Menyampaikan skor ini langsung kepada
           mahasiswa berisiko menjadi ramalan yang justru mewujudkan dirinya sendiri.
        5. **Perlu dilatih ulang secara berkala.** Kurikulum, kebijakan biaya, dan profil mahasiswa
           berubah. Latih ulang model setiap tahun akademik dengan data terbaru.
        """
    )

    st.divider()
    st.caption(
        f"Proyek akhir Dicoding — Penerapan Data Science · {META['author']} "
        f"(ID Dicoding: {META['dicoding_id']}) · Model dilatih {META['trained_at']}"
    )
