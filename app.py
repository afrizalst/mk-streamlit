"""
Tugas P2 - Representasi Data & Pra-pemrosesan (Pembelajaran Mesin)
Dataset : Credit Card Customers (Kaggle, sakshigoyal7/credit-card-customers)
Model   : Gaussian Naive Bayes
Jalankan: streamlit run app.py
"""
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "BankChurners.csv"
TARGET = "Attrition_Flag"          # kolom target asli
POSITIF = "Attrited Customer"      # nilai target yang dianggap churn (=1)
KOLOM_ID = ["CLIENTNUM"]           # kolom identitas, bukan fitur
AWALAN_NB = "Naive_Bayes"          # dua kolom hasil Naive Bayes bawaan Kaggle
SEED = 42
BIRU, MERAH, ORANYE, ABU = "#3B6FB6", "#D9534F", "#F0A030", "#E5E7EB"
PILIHAN_SCALER = ["Tanpa scaling", "MinMax (normalisasi)", "Standard (standarisasi)"]

st.set_page_config(page_title="Praproses Credit Card Customers", layout="wide",
                   initial_sidebar_state="collapsed")

# gaya grafik yang seragam
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": ABU, "axes.axisbelow": True,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# gaya halaman (memakai warna transparan agar cocok di tema terang maupun gelap)
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
.banner {{ background: linear-gradient(120deg, {BIRU} 0%, #6C5CE7 100%); color: #fff;
          padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem; }}
.banner h1 {{ margin: 0; font-size: 1.9rem; color: #fff; }}
.banner p  {{ margin: .35rem 0 .7rem 0; opacity: .92; }}
.chip {{ display: inline-block; background: rgba(255,255,255,.18); padding: .15rem .7rem;
        border-radius: 999px; font-size: .78rem; margin-right: .4rem; }}
.ringkas {{ border-left: 5px solid {BIRU}; background: rgba(59,111,182,.10);
           padding: .7rem 1rem; border-radius: 8px; margin: .2rem 0 1rem 0; }}
div[data-testid="stMetric"] {{ background: rgba(128,128,128,.08); border-left: 4px solid {BIRU};
           padding: .6rem .9rem; border-radius: 10px; }}
div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
h2, h3 {{ margin-top: 1.2rem !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FUNGSI BANTU
# ----------------------------------------------------------------------------
def kotak(teks: str):
    """Kotak satu kalimat 'Apa yang dilakukan' di awal tiap tab."""
    st.markdown(f'<div class="ringkas"><b>Apa yang dilakukan:</b> {teks}</div>',
                unsafe_allow_html=True)


def tampil(fig):
    """Tampilkan grafik lalu tutup agar memori tidak menumpuk."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def kolom_kategori(df: pd.DataFrame) -> list:
    """Kolom non-numerik selain target (aman untuk pandas 2.x maupun 3.x)."""
    return [c for c in df.columns if c != TARGET and not is_numeric_dtype(df[c])]


def _baca(sumber) -> pd.DataFrame:
    """Baca CSV secara tahan banting.
    1) pemisah koma; 2) deteksi pemisah lain (mis. titik koma);
    3) perbaikan file yang SETIAP BARISNYA terbungkus tanda kutip (terbaca sebagai 1 kolom).
    """
    if isinstance(sumber, (bytes, bytearray)):
        teks = bytes(sumber).decode("utf-8-sig", errors="replace")
    else:
        teks = Path(sumber).read_text(encoding="utf-8-sig", errors="replace")

    df = pd.read_csv(io.StringIO(teks))
    if df.shape[1] == 1:                                   # coba pemisah lain
        try:
            alt = pd.read_csv(io.StringIO(teks), sep=None, engine="python")
            if alt.shape[1] > 1:
                df = alt
        except Exception:
            pass
    if df.shape[1] == 1 and "," in str(df.columns[0]):     # baris terbungkus kutip
        baris = []
        for b in teks.splitlines():
            b = b.strip()
            if len(b) >= 2 and b[0] == '"' and b[-1] == '"':
                b = b[1:-1].replace('""', '"')
            baris.append(b)
        df = pd.read_csv(io.StringIO("\n".join(baris)))

    df.columns = df.columns.astype(str).str.strip().str.strip('"').str.strip("'")
    # samakan nama kolom penting walau huruf besar/kecilnya berbeda
    peta = {c.lower(): c for c in [TARGET] + KOLOM_ID}
    return df.rename(columns={c: peta[c.lower()] for c in df.columns if c.lower() in peta})


@st.cache_data
def baca_csv_path(path: str, mtime: float) -> pd.DataFrame:
    # mtime ikut jadi kunci cache: kalau file diganti, data dibaca ulang
    return _baca(path)


@st.cache_data
def baca_csv_bytes(data: bytes) -> pd.DataFrame:
    return _baca(data)


def muat_data() -> pd.DataFrame:
    """Baca BankChurners.csv dari folder yang sama; kalau tidak ada, minta unggah."""
    if DATA_PATH.exists():
        return baca_csv_path(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    st.warning(f"File `{DATA_PATH.name}` tidak ditemukan di folder program. "
               "Letakkan di sebelah `app.py` atau unggah di bawah ini.")
    up = st.file_uploader("Unggah BankChurners.csv", type="csv")
    if up is None:
        st.stop()
    return baca_csv_bytes(up.getvalue())


def rapikan_kategori(seri: pd.Series) -> pd.Series:
    """Hapus spasi berlebih lalu samakan varian kapitalisasi ke bentuk yang paling sering."""
    s = seri.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    kunci = s.str.lower()
    # bentuk penulisan paling sering untuk tiap kunci huruf kecil
    kanonik = (pd.DataFrame({"k": kunci, "v": s}).dropna()
               .groupby("k")["v"].agg(lambda x: x.value_counts().index[0]))
    return kunci.map(kanonik).astype(object)


# ----------------------------------------------------------------------------
# PIPELINE PRAPROSES (di-cache supaya demo cepat)
# ----------------------------------------------------------------------------
@st.cache_data
def praproses(raw: pd.DataFrame) -> dict:
    P = {"raw": raw}

    # 1) Kolom yang dibuang: ID dan kolom Naive Bayes bawaan Kaggle
    buang = [c for c in raw.columns if c in KOLOM_ID or c.startswith(AWALAN_NB)]
    P["buang"] = buang
    df = raw.drop(columns=buang).copy()

    # 2) Koreksi tipe: kolom teks yang isinya angka diubah ke numerik
    P["tipe_awal"] = df.dtypes.astype(str)
    dikoreksi = []
    for c in kolom_kategori(df):
        konv = pd.to_numeric(df[c], errors="coerce")
        if konv.notna().sum() == df[c].notna().sum():
            df[c] = konv
            dikoreksi.append(c)
    P["dikoreksi"] = dikoreksi
    P["df_tipe"] = df.copy()                       # untuk tab Missing

    # 3) Rapikan kategori (spasi / kapitalisasi)
    kat = kolom_kategori(df)
    P["kat"] = kat
    sebelum = {c: df[c].value_counts(dropna=False) for c in kat}
    for c in kat:
        df[c] = rapikan_kategori(df[c])
    sesudah = {c: df[c].value_counts(dropna=False) for c in kat}
    P["kat_sebelum"], P["kat_sesudah"] = sebelum, sesudah
    P["sel_berubah"] = int(sum((P["df_tipe"][c].astype(object).fillna("~") != df[c].fillna("~")).sum()
                               for c in kat))

    # 4) Missing value (NaN) -> baris dibuang bila ada
    P["jumlah_nan"] = int(df.isna().sum().sum())
    P["baris_nan"] = int(df.isna().any(axis=1).sum())
    df = df.dropna().copy()

    # 5) Duplikasi (seluruh kolom selain ID)
    P["jumlah_duplikat"] = int(df.duplicated().sum())
    P["id_duplikat"] = (int(raw[KOLOM_ID[0]].duplicated().sum())
                        if KOLOM_ID[0] in raw.columns else 0)
    df = df.drop_duplicates().reset_index(drop=True)

    # 6) Target -> 0/1
    df["Churn"] = (df[TARGET] == POSITIF).astype(int)
    df = df.drop(columns=[TARGET])
    P["df_bersih"] = df

    # 7) Pisah fitur numerik / kategorikal, one-hot encoding
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    P["num"] = [c for c in X.columns if is_numeric_dtype(X[c])]
    P["kat_fitur"] = [c for c in X.columns if c not in P["num"]]
    P["X"] = pd.get_dummies(X, columns=P["kat_fitur"], dtype=int)
    P["y"] = y

    # 8) Split SEBELUM scaling (stratified, mencegah kebocoran statistik)
    Xtr, Xte, ytr, yte = train_test_split(
        P["X"], y, test_size=0.2, stratify=y, random_state=SEED)
    P["Xtr"], P["Xte"], P["ytr"], P["yte"] = Xtr, Xte, ytr, yte
    return P


def terapkan_scaler(P: dict, nama: str):
    """Fit scaler HANYA pada data latih, lalu transform latih, uji, dan seluruh data."""
    Xtr, Xte, Xall = P["Xtr"].copy(), P["Xte"].copy(), P["X"].copy()
    if nama == "Tanpa scaling":
        return Xtr, Xte, Xall
    sc = MinMaxScaler() if nama.startswith("MinMax") else StandardScaler()
    num = P["num"]
    Xtr[num] = sc.fit_transform(P["Xtr"][num])
    Xte[num] = sc.transform(P["Xte"][num])
    Xall[num] = sc.transform(P["X"][num])
    return Xtr, Xte, Xall


def latih_nb(P: dict, nama_scaler: str) -> dict:
    Xtr, Xte, Xall = terapkan_scaler(P, nama_scaler)
    nb = GaussianNB().fit(Xtr, P["ytr"])
    pred = nb.predict(Xte)
    yte = P["yte"]
    return {
        "akurasi": accuracy_score(yte, pred),
        "presisi": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "pred": pred, "Xall": Xall,
        "laporan": classification_report(yte, pred, target_names=["Tetap", "Churn"],
                                         zero_division=0),
    }


# ----------------------------------------------------------------------------
# MUAT DATA & JALANKAN PRAPROSES
# ----------------------------------------------------------------------------
st.markdown("""
<div class="banner">
  <h1>Pra-pemrosesan Data: Credit Card Customers</h1>
  <p>Tugas P2 - Pembelajaran Mesin &nbsp;|&nbsp; Klasifikasi churn pelanggan kartu kredit</p>
  <span class="chip">Model: Gaussian Naive Bayes</span>
  <span class="chip">Split 80:20 (stratified)</span>
  <span class="chip">random_state = 42</span>
</div>
""", unsafe_allow_html=True)

raw = muat_data()
if TARGET not in raw.columns:
    st.error(f"Kolom target `{TARGET}` tidak ditemukan. Pastikan file yang dipakai adalah BankChurners.csv.")
    st.write(f"File terbaca dengan **{raw.shape[0]:,} baris** dan **{raw.shape[1]} kolom**. "
             "Kolom yang terbaca:")
    st.code("\n".join(map(str, raw.columns)))
    st.stop()
P = praproses(raw)
df, num, katf = P["df_bersih"], P["num"], P["kat_fitur"]

tabs = st.tabs(["1 · Data & Tipe", "2 · Missing & Duplikasi", "3 · Kategori",
                "4 · Opsional", "5 · Scaling", "6 · Hasil Akhir"])

# ----------------------------------------------------------------------------
# TAB 1 - DATA & TIPE (4a)
# ----------------------------------------------------------------------------
with tabs[0]:
    kotak("Memahami struktur data, membuang kolom yang bukan fitur, dan mengoreksi tipe kolom.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{raw.shape[0]:,}")
    c2.metric("Kolom awal", raw.shape[1])
    c3.metric("Kolom dibuang", len(P["buang"]))
    c4.metric("Calon fitur", len(num) + len(katf))

    st.subheader("Temuan")
    info = pd.DataFrame({"Tipe": raw.dtypes.astype(str), "Non-null": raw.notna().sum(),
                         "Nilai unik": raw.nunique()})
    st.dataframe(info)

    st.subheader("Tindakan")
    alasan = []
    for c in P["buang"]:
        if c in KOLOM_ID:
            alasan.append((c, "Nomor identitas pelanggan, bukan fitur"))
        else:
            alasan.append((c, "Keluaran model Naive Bayes bawaan dataset: bocoran (leakage)"))
    alasan.append((TARGET, f"Dijadikan target 'Churn': 1 = {POSITIF}, 0 = lainnya"))
    st.table(pd.DataFrame(alasan, columns=["Kolom", "Keterangan"]))
    if P["dikoreksi"]:
        st.write("Kolom teks yang diubah ke numerik:", ", ".join(P["dikoreksi"]))
    else:
        st.write("Koreksi tipe: tidak ada kolom teks yang isinya angka, jadi tidak ada yang diubah.")

    st.subheader("Hasil")
    st.write(f"**{len(num)} fitur numerik** dan **{len(katf)} fitur kategorikal**, "
             f"total **{len(num) + len(katf)} calon fitur** (syarat minimal 10).")
    st.write("Numerik:", ", ".join(num))
    st.write("Kategorikal:", ", ".join(katf))

# ----------------------------------------------------------------------------
# TAB 2 - MISSING & DUPLIKASI (4b, 4c)
# ----------------------------------------------------------------------------
with tabs[1]:
    kotak("Mengecek nilai kosong (termasuk 'Unknown' yang tidak terdeteksi isna) dan baris duplikat.")
    dft = P["df_tipe"]

    st.subheader("Temuan")
    st.write("**Nilai kosong (NaN) per kolom**")
    nan_tbl = dft.isna().sum().rename("Jumlah NaN").to_frame()
    nan_tbl["Persen"] = (nan_tbl["Jumlah NaN"] / len(dft) * 100).round(2)
    st.dataframe(nan_tbl[nan_tbl["Jumlah NaN"] > 0] if nan_tbl["Jumlah NaN"].sum() > 0
                 else nan_tbl.head(0))
    c1, c2 = st.columns(2)
    c1.metric("Total NaN", P["jumlah_nan"])
    kosong_str = int(sum((dft[c].astype("string").str.strip() == "").sum()
                         for c in P["kat"]))
    c2.metric("Teks kosong ('')", kosong_str)

    st.write("**Kategori 'Unknown' per kolom**")
    unk = []
    for c in P["kat"]:
        n = int((dft[c].astype("string").str.strip().str.lower() == "unknown").sum())
        if n > 0:
            unk.append((c, n, round(n / len(dft) * 100, 2)))
    unk_df = pd.DataFrame(unk, columns=["Kolom", "Jumlah", "Persen"])
    st.dataframe(unk_df, hide_index=True)

    st.write("**Duplikasi**")
    c1, c2 = st.columns(2)
    c1.metric("Baris duplikat (tanpa ID)", P["jumlah_duplikat"])
    c2.metric("ID pelanggan duplikat", P["id_duplikat"])

    st.subheader("Tindakan")
    if P["jumlah_nan"] == 0:
        st.write("- NaN: tidak ada, tidak perlu imputasi atau penghapusan.")
    else:
        st.write(f"- NaN: {P['baris_nan']} baris yang berisi NaN dibuang.")
    if len(unk_df) > 0:
        st.write("- 'Unknown': dipertahankan sebagai kategori tersendiri, karena "
                 "'tidak diketahui' itu sendiri bisa menjadi informasi bagi model.")
    st.write(f"- Duplikat: {P['jumlah_duplikat']} baris dihapus.")

    st.subheader("Hasil")
    st.write(f"Jumlah baris setelah tahap ini: **{len(df):,}** (awal {len(raw):,}).")

# ----------------------------------------------------------------------------
# TAB 3 - KATEGORI (4e)
# ----------------------------------------------------------------------------
with tabs[2]:
    kotak("Memeriksa ejaan dan kapitalisasi nilai kategori: spasi berlebih dibuang, "
          "varian huruf besar/kecil disamakan.")
    st.subheader("Temuan, tindakan, dan hasil")
    st.metric("Jumlah sel yang diubah", P["sel_berubah"])
    if P["sel_berubah"] == 0:
        st.write("Tidak ditemukan varian ejaan atau kapitalisasi pada kolom kategorikal, "
                 "jadi nilainya tidak berubah. Tabel di bawah menunjukkan nilai tiap kategori.")
    rows = []
    for c in P["kat"]:
        sb, ss = P["kat_sebelum"][c], P["kat_sesudah"][c]
        for nilai in sorted(set(sb.index.astype(str)) | set(ss.index.astype(str))):
            b = int(sb[sb.index.astype(str) == nilai].sum())
            s = int(ss[ss.index.astype(str) == nilai].sum())
            rows.append((c, nilai, b, s))
    st.dataframe(pd.DataFrame(rows, columns=["Kolom", "Nilai", "Sebelum", "Sesudah"]),
                 hide_index=True)

# ----------------------------------------------------------------------------
# TAB 4 - OPSIONAL: OUTLIER, IMBALANCE, LEAKAGE (5a-c)
# ----------------------------------------------------------------------------
with tabs[3]:
    kotak("Tiga pengecekan tambahan: outlier (IQR), ketidakseimbangan kelas, dan indikasi kebocoran data.")

    # ---- a) Outlier IQR
    st.subheader("a) Outlier dengan metode IQR")
    ringkas = []
    for c in num:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ringkas.append((c, int(((df[c] < lo) | (df[c] > hi)).sum())))
    ringkas = pd.DataFrame(ringkas, columns=["Kolom", "Jumlah outlier"])
    ringkas["Persen"] = (ringkas["Jumlah outlier"] / len(df) * 100).round(2)

    default = num.index("Credit_Limit") if "Credit_Limit" in num else 0
    kol = st.selectbox("Pilih kolom numerik", num, index=default)
    q1, q3 = df[kol].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[kol] < lo) | (df[kol] > hi)).sum())
    m = st.columns(5)
    m[0].metric("Q1", f"{q1:,.2f}")
    m[1].metric("Q3", f"{q3:,.2f}")
    m[2].metric("Batas bawah", f"{lo:,.2f}")
    m[3].metric("Batas atas", f"{hi:,.2f}")
    m[4].metric("Outlier", f"{n_out:,} ({n_out / len(df) * 100:.1f}%)")
    g1, g2 = st.columns([2, 1])
    with g1:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.boxplot(df[kol], vert=False, patch_artist=True,
                   boxprops={"facecolor": "#C9D9F0", "edgecolor": BIRU},
                   medianprops={"color": MERAH}, flierprops={"markersize": 3, "markeredgecolor": MERAH})
        ax.set_xlabel(kol)
        ax.grid(axis="y", visible=False)
        tampil(fig)
    with g2:
        st.dataframe(ringkas, hide_index=True)
    st.write("**Tindakan:** outlier hanya dideteksi dan dilaporkan, **tidak dibuang**. Nilai tinggi "
             "pada kolom seperti limit kredit atau nilai transaksi wajar secara bisnis, dan "
             "menghapusnya akan membuang data pelanggan yang sah.")

    # ---- b) Imbalance
    st.subheader("b) Ketidakseimbangan kelas")
    vc = df["Churn"].value_counts().sort_index()
    persen = vc / vc.sum() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Tetap (0)", f"{vc.get(0, 0):,} ({persen.get(0, 0):.1f}%)")
    c2.metric("Churn (1)", f"{vc.get(1, 0):,} ({persen.get(1, 0):.1f}%)")
    c3.metric("Rasio mayoritas:minoritas", f"{vc.max() / max(vc.min(), 1):.1f} : 1")
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.bar(["Tetap", "Churn"], [vc.get(0, 0), vc.get(1, 0)], color=[BIRU, MERAH], width=0.55)
    ax.set_ylabel("Jumlah")
    ax.grid(axis="x", visible=False)
    tampil(fig)
    st.write("**Tindakan:** data dipisah dengan *stratified split* agar proporsi kelas sama di data "
             "latih dan uji. Evaluasi memakai recall dan F1 kelas churn, bukan akurasi saja, "
             "karena model yang selalu menebak 'Tetap' tetap terlihat akurat.")

    # ---- c) Leakage
    st.subheader("c) Indikasi kebocoran data (leakage)")
    nb_cols = [c for c in raw.columns if c.startswith(AWALAN_NB)]
    if nb_cols:
        st.write("Kolom hasil Naive Bayes ditemukan di data mentah dan **sudah dibuang**:")
        st.code("\n".join(nb_cols))
    else:
        st.write("Tidak ada kolom hasil model (Naive Bayes) di data mentah.")

    corr = df[num].corr()
    pasangan = [(a, b, corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:]
                if abs(corr.loc[a, b]) >= 0.8]
    st.write("**Pasangan fitur dengan |korelasi| >= 0.8** (fitur yang saling menduplikasi):")
    if pasangan:
        st.dataframe(pd.DataFrame(pasangan, columns=["Fitur A", "Fitur B", "Korelasi"])
                     .round(3), hide_index=True)
    else:
        st.write("Tidak ada.")

    cek = {"Avg_Open_To_Buy", "Credit_Limit", "Total_Revolving_Bal"}
    if cek.issubset(df.columns):
        selisih = (df["Avg_Open_To_Buy"] - (df["Credit_Limit"] - df["Total_Revolving_Bal"])).abs()
        st.write(f"Uji: `Avg_Open_To_Buy` = `Credit_Limit` - `Total_Revolving_Bal` berlaku "
                 f"(selisih <= 1) pada **{(selisih <= 1).mean() * 100:.1f}%** baris.")

    kor_target = df[num + ["Churn"]].corr()["Churn"].drop("Churn")
    top = kor_target.abs().sort_values(ascending=False).head(5)
    st.write("**5 fitur numerik paling berkorelasi dengan target** (korelasi sangat tinggi, "
             "mis. di atas 0.8, patut dicurigai bocor):")
    st.dataframe(pd.DataFrame({"Fitur": top.index, "Korelasi dengan Churn": kor_target[top.index].round(3).values}),
                 hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=7)
    ax.set_yticks(range(len(num)), num, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    tampil(fig)
    st.write("**Tindakan:** split dilakukan sebelum scaling dan scaler hanya di-fit pada data latih, "
             "sehingga tidak ada informasi data uji yang masuk ke proses latih. Fitur yang "
             "berkorelasi tinggi dilaporkan tetapi dipertahankan.")

# ----------------------------------------------------------------------------
# TAB 5 - SCALING (4d)
# ----------------------------------------------------------------------------
with tabs[4]:
    kotak("Normalisasi (MinMax, rentang 0-1) dan standarisasi (Standard, rata-rata 0 dan simpangan baku 1) "
          "pada fitur numerik. Kolom hasil one-hot tidak diskalakan.")
    pilih = st.radio("Pilih scaler", PILIHAN_SCALER, horizontal=True, index=2)

    st.subheader("Temuan: statistik sebelum scaling (data latih)")
    st.dataframe(P["Xtr"][num].describe().loc[["min", "max", "mean", "std"]].T.round(3))
    st.subheader("Hasil: statistik sesudah scaling (data latih)")
    Xtr_s, _, _ = terapkan_scaler(P, pilih)
    st.dataframe(Xtr_s[num].describe().loc[["min", "max", "mean", "std"]].T.round(3))

    kolh = "Credit_Limit" if "Credit_Limit" in num else num[0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    ax[0].hist(P["Xtr"][kolh], bins=30, color=BIRU)
    ax[0].set_title(f"{kolh} - sebelum")
    ax[1].hist(Xtr_s[kolh], bins=30, color=ORANYE)
    ax[1].set_title(f"{kolh} - sesudah ({pilih})")
    tampil(fig)
    st.caption("Bentuk sebaran tidak berubah; hanya skala sumbunya yang berubah.")

    st.subheader("Pengaruh terhadap Gaussian Naive Bayes")
    banding = []
    for nm in PILIHAN_SCALER:
        r = latih_nb(P, nm)
        banding.append((nm, r["akurasi"], r["presisi"], r["recall"], r["f1"]))
    st.dataframe(pd.DataFrame(banding, columns=["Scaler", "Akurasi", "Presisi (churn)",
                                                "Recall (churn)", "F1 (churn)"]).round(4),
                 hide_index=True)
    st.caption("GaussianNB menghitung rata-rata dan varians per kelas untuk tiap fitur secara "
               "terpisah, jadi skala fitur tidak mengubah keputusan secara berarti. Scaling "
               "tetap dilakukan sesuai tugas.")

# ----------------------------------------------------------------------------
# TAB 6 - HASIL AKHIR (6)
# ----------------------------------------------------------------------------
with tabs[5]:
    kotak("Menampilkan 5 data teratas dan 5 terbawah hasil pemrosesan, lalu melatih dan mengevaluasi Gaussian Naive Bayes.")
    st.caption(f"Scaler yang dipakai: **{pilih}** (diatur di tab 5)")
    hasil = latih_nb(P, pilih)

    st.subheader("Data bersih (sebelum encoding dan scaling)")
    st.write("5 teratas")
    st.dataframe(df.head(5))
    st.write("5 terbawah")
    st.dataframe(df.tail(5))

    st.subheader("Data akhir (one-hot + scaling, siap masuk model)")
    akhir = hasil["Xall"].copy()
    akhir["Churn"] = P["y"].values
    st.caption(f"{akhir.shape[0]:,} baris x {akhir.shape[1]} kolom (termasuk target Churn)")
    st.write("5 teratas")
    st.dataframe(akhir.head(5))
    st.write("5 terbawah")
    st.dataframe(akhir.tail(5))

    st.subheader("Gaussian Naive Bayes")
    st.write(f"Data latih: {len(P['Xtr']):,} baris | data uji: {len(P['Xte']):,} baris")
    m = st.columns(4)
    m[0].metric("Akurasi", f"{hasil['akurasi']:.3f}")
    m[1].metric("Presisi (churn)", f"{hasil['presisi']:.3f}")
    m[2].metric("Recall (churn)", f"{hasil['recall']:.3f}")
    m[3].metric("F1 (churn)", f"{hasil['f1']:.3f}")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ConfusionMatrixDisplay.from_predictions(
            P["yte"], hasil["pred"], display_labels=["Tetap", "Churn"],
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion matrix (data uji)")
        ax.grid(False)
        tampil(fig)
    with g2:
        st.code(hasil["laporan"])

    miring = P["Xtr"][num].skew().abs()
    miring = miring[miring > 1].sort_values(ascending=False)
    if len(miring):
        st.caption("Catatan: GaussianNB mengasumsikan tiap fitur berdistribusi normal per kelas. "
                   f"Ada {len(miring)} fitur numerik dengan |skewness| > 1 "
                   f"({', '.join(miring.index[:4])}{', ...' if len(miring) > 4 else ''}), "
                   "sehingga asumsi ini hanya terpenuhi secara kasar.")