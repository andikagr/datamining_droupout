"""
app.py — Streamlit Prediksi Dropout & Keberhasilan Akademik Mahasiswa
K-Means + Hierarchical Agglomerative Clustering | CRISP-DM
Tugas Besar Penambangan Data | Universitas Telkom 2026
Jalankan: streamlit run app.py
"""
# Mengimpor library Streamlit untuk membangun antarmuka web interaktif
import streamlit as st
# Mengimpor pandas untuk manipulasi data tabular (DataFrame)
import pandas as pd
# Mengimpor numpy untuk operasi matematika dan array
import numpy as np
# Mengimpor joblib untuk memuat model machine learning yang sudah dilatih
import joblib
# Mengimpor plotly graph_objects untuk membuat visualisasi kustom tingkat lanjut (seperti gauge chart)
import plotly.graph_objects as go
# Mengimpor plotly express untuk membuat visualisasi standar dengan cepat (seperti scatter plot, confusion matrix)
import plotly.express as px
# Mengimpor Path dari library pathlib untuk mengelola jalur file/direktori dengan lebih baik
from pathlib import Path

# ── PATH ──────────────────────────────────────────────────────
# Mendefinisikan direktori dasar (BASE_DIR) tempat file app.py ini berada
BASE_DIR  = Path(__file__).parent          
# Menetapkan jalur direktori 'model' yang berisi file model (.pkl) hasil pelatihan
MODEL_DIR = BASE_DIR / "model"

# ── KONFIGURASI ───────────────────────────────────────────────
# Mengkonfigurasi pengaturan halaman awal Streamlit
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",  # Judul yang muncul di tab browser
    page_icon="",                             # Ikon halaman (saat ini dikosongkan)
    layout="wide",                            # Menggunakan seluruh lebar layar (wide)
    initial_sidebar_state="expanded",         # Membiarkan menu samping (sidebar) terbuka secara default
)

# Menyisipkan kode CSS kustom untuk memberikan styling warna pada kotak hasil prediksi
st.markdown("""
<style>
/* Styling untuk prediksi Risiko Tinggi (merah) */
.result-high   { background:#ffe5e5; border-left:6px solid #e74c3c;
                 padding:20px; border-radius:8px; margin:10px 0; color:#333333; }
/* Styling untuk prediksi Risiko Sedang (kuning/oranye) */
.result-medium { background:#fff3cd; border-left:6px solid #f39c12;
                 padding:20px; border-radius:8px; margin:10px 0; color:#333333; }
/* Styling untuk prediksi Risiko Rendah (hijau) */
.result-low    { background:#e8f8e8; border-left:6px solid #27ae60;
                 padding:20px; border-radius:8px; margin:10px 0; color:#333333; }
</style>
""", unsafe_allow_html=True) # unsafe_allow_html wajib True agar tag <style> bisa dirender HTML

# ── LOAD MODEL ────────────────────────────────────────────────
# Menggunakan decorator @st.cache_resource agar model hanya diload sekali ke memori (mempercepat performa web)
@st.cache_resource
def load_all():
    # Memuat model K-Means yang sudah dilatih sebelumnya
    km     = joblib.load(MODEL_DIR / "kmeans.pkl")
    # Memuat StandardScaler untuk standarisasi input data baru
    sc     = joblib.load(MODEL_DIR / "scaler.pkl")
    # Memuat LabelEncoder jika ada fitur yang perlu ditransformasi (opsional/metadata)
    le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
    # Memuat model PCA untuk melakukan visualisasi data ke dalam dimensi 2D
    pca    = joblib.load(MODEL_DIR / "pca.pkl")
    # Memuat daftar nama kolom fitur yang digunakan saat training (untuk mencegah error format kolom)
    cols   = joblib.load(MODEL_DIR / "feature_columns.pkl")
    # Memuat mapping nama cluster yang sudah dianalisis profilnya
    cnames = joblib.load(MODEL_DIR / "cluster_names.pkl")
    # Memuat metrik evaluasi model dari data latih untuk ditampilkan di Dashboard
    ev     = joblib.load(MODEL_DIR / "eval_results.pkl")
    # Mengembalikan semua objek yang sudah diload
    return km, sc, le, pca, cols, cnames, ev

# Mencoba memanggil fungsi load_all
try:
    KMEANS, SCALER, LE, PCA_MODEL, FEATURE_COLS, CLUSTER_NAMES, EVAL = load_all()
except Exception as e:
    # Menampilkan pesan error di layar jika model gagal diload (biasanya karena belum menjalankan train_model.py)
    st.error(f"Model tidak ditemukan. Jalankan: `python train_model.py`\n\n{e}")
    # Menghentikan eksekusi script lebih lanjut
    st.stop()

# Mengambil daftar nama kelas (target asli) jika tersimpan di LabelEncoder
CLASS_NAMES = list(LE.classes_)
# Mengambil jumlah cluster (K) yang dipakai pada K-Means dari data evaluasi
N_KM = EVAL["kmeans"]["n_clusters"]
# Mengambil jumlah cluster (K) yang dipakai pada Agglomerative Hierarchical dari data evaluasi
N_AG = EVAL["agglo"]["n_clusters"]

# ── SIDEBAR ───────────────────────────────────────────────────
# Menampilkan gambar logo Universitas Telkom di sidebar (menu samping)
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
    "Telkom_University_logo.svg/320px-Telkom_University_logo.svg.png",
    width=160,
)
# Menambahkan judul di menu sidebar
st.sidebar.title("Menu Navigasi")
# Membuat radio button untuk memilih halaman, dan menyimpannya ke variabel 'page'
page = st.sidebar.radio("", [
    "Prediksi Cluster",    # Pilihan halaman 1
    "Dashboard Evaluasi",  # Pilihan halaman 2
    "Visualisasi Cluster", # Pilihan halaman 3
    "Tentang",             # Pilihan halaman 4
])
# Menambahkan garis pemisah di sidebar
st.sidebar.divider()
# Menambahkan teks kecil (caption) sebagai keterangan di bagian bawah sidebar
st.sidebar.caption("Tugas Besar Penambangan Data\nUniversitas Telkom 2026")

# ══════════════════════════════════════════════════════════════
#  HALAMAN 1 — PREDIKSI CLUSTER
# ══════════════════════════════════════════════════════════════
# Jika user memilih halaman "Prediksi Cluster"
if page == "Prediksi Cluster":
    # Menampilkan judul utama halaman
    st.title("Prediksi Segmen Risiko Mahasiswa")
    # Menampilkan teks deskripsi (markdown) yang menginstruksikan user untuk mengisi data
    st.markdown(
        "Masukkan data mahasiswa di bawah untuk menentukan **cluster risiko dropout** "
        "menggunakan **K-Means Clustering** (Champion Model sesuai laporan Bab IV.3)."
    )
    # Menambahkan garis pemisah
    st.divider()

    # Menambahkan subjudul
    st.markdown("### Masukkan Data Mahasiswa")
    
    # Membuat expander (kolom bisa dilipat) untuk kelompok input Faktor Akademik
    with st.expander("Faktor Akademik", expanded=True):
        # Membagi expander menjadi 3 kolom yang sejajar
        c1, c2, c3 = st.columns(3)
        # Mengisi kolom pertama
        with c1:
            # Membuat slider input untuk fitur jumlah matkul lulus di semester 1
            s1_app = st.slider("MK Lulus Sem 1", 0, 26, 6)
            # Membuat slider input untuk fitur nilai rata-rata di semester 1
            s1_grd = st.slider("Nilai Sem 1", 0.0, 20.0, 12.0)
            # Membuat slider input untuk fitur jumlah matkul terdaftar di semester 1
            s1_enr = st.slider("MK Terdaftar Sem 1", 0, 26, 6)
            # Membuat slider input untuk fitur matkul credited semester 1
            s1_cred = st.slider("MK Credited Sem 1", 0, 20, 0)
            # Membuat slider input untuk fitur jumlah evaluasi di semester 1
            s1_eval = st.slider("Evaluasi Sem 1", 0, 45, 6)
        # Mengisi kolom kedua
        with c2:
            # Membuat slider input untuk fitur jumlah matkul lulus di semester 2
            s2_app = st.slider("MK Lulus Sem 2", 0, 20, 6)
            # Membuat slider input untuk fitur nilai rata-rata di semester 2
            s2_grd = st.slider("Nilai Sem 2", 0.0, 20.0, 12.0)
            # Membuat slider input untuk fitur jumlah matkul terdaftar di semester 2
            s2_enr = st.slider("MK Terdaftar Sem 2", 0, 23, 6)
            # Membuat slider input untuk fitur jumlah evaluasi di semester 2
            s2_eval = st.slider("Evaluasi Sem 2", 0, 33, 6)
        # Mengisi kolom ketiga
        with c3:
            # Membuat selectbox untuk kode prodi/course mahasiswa (1-17)
            course_opts = ["1 - Teknik Informatika", "2 - Sistem Informasi", "3 - Ilmu Komputer", "4 - Manajemen Bisnis", "5 - Desain"] + [f"{i} - Program Studi Lainnya" for i in range(6, 18)]
            course = st.selectbox("Program Studi (Course)", course_opts)
            # Membuat selectbox untuk memilih waktu kehadiran kuliah (Siang/Malam)
            attendance = st.selectbox("Waktu Kuliah", ["Siang/Pagi (1)", "Malam (0)"])
            # Membuat selectbox untuk riwayat jenjang pendidikan sebelumnya
            prev_qual_opts = ["1 - Pendidikan Menengah (SMA/SMK)", "2 - Pendidikan Tinggi (S1)"] + [f"{i} - Kualifikasi Lainnya" for i in range(3, 18)]
            prev_qual = st.selectbox("Kualifikasi Pendidikan Sebelumnya", prev_qual_opts)

    # Membuat expander untuk kelompok input Faktor Administratif
    with st.expander("Faktor Administratif", expanded=True):
        # Membagi area menjadi 2 kolom
        c1, c2 = st.columns(2)
        # Mengisi kolom pertama
        with c1:
            # Membuat checkbox untuk status menunggak UKT (jika dicentang bernilai True)
            debtor = st.checkbox("Tunggakan UKT (Debtor)")
            # Membuat checkbox untuk status UKT telah lunas (default True)
            tuition_ok = st.checkbox("UKT Lunas", value=True)
            # Membuat checkbox untuk status penerima beasiswa
            scholarship = st.checkbox("Penerima Beasiswa")
        # Mengisi kolom kedua
        with c2:
            # Membuat selectbox untuk jalur masuk pendaftaran mahasiswa
            app_mode_opts = ["1 - Reguler / Nasional", "2 - Jalur Mandiri", "3 - Beasiswa Prestasi", "4 - Jalur Internasional", "5 - Pindahan"] + [f"{i} - Jalur Lainnya" for i in range(6, 19)]
            app_mode = st.selectbox("Jalur Pendaftaran (Mode)", app_mode_opts)
            # Membuat slider untuk prioritas jurusan yang dipilih saat mendaftar
            app_order = st.slider("Urutan Pilihan Prodi", 0, 9, 1)

    # Membuat expander untuk kelompok input Faktor Makro Ekonomi
    with st.expander("Faktor Makro Ekonomi", expanded=True):
        # Membagi area menjadi 3 kolom
        c1, c2, c3 = st.columns(3)
        with c1:
            # Membuat slider input untuk angka tingkat pengangguran nasional
            unemployment = st.slider("Tingkat Pengangguran (%)", 7.6, 16.2, 11.6, 0.1)
        with c2:
            # Membuat slider input untuk tingkat inflasi nasional
            inflation = st.slider("Tingkat Inflasi (%)", -0.8, 3.7, 1.2, 0.1)
        with c3:
            # Membuat slider input untuk persentase pertumbuhan GDP
            gdp = st.slider("Pertumbuhan GDP (%)", -4.1, 3.5, 0.0, 0.01)

    # Membuat expander untuk kelompok input Faktor Geografis & Demografis
    with st.expander("Faktor Geografis", expanded=True):
        # Membagi area menjadi 2 kolom
        c1, c2 = st.columns(2)
        with c1:
            # Membuat slider input untuk usia saat pendaftaran
            age = st.slider("Usia", 17, 60, 20)
            # Membuat selectbox untuk gender
            gender = st.selectbox("Gender", ["Perempuan (0)", "Laki-laki (1)"])
            # Membuat selectbox untuk status pernikahan
            marital_opts = ["1 - Lajang", "2 - Menikah", "3 - Janda/Duda", "4 - Cerai", "5 - Bersama (De Facto)", "6 - Terpisah secara hukum"]
            marital = st.selectbox("Status Pernikahan", marital_opts)
            # Membuat selectbox untuk kode negara asal
            nat_opts = ["1 - Portugal", "2 - Spanyol", "3 - Brazil", "4 - Inggris", "5 - Lainnya"] + [f"{i} - Negara Tipe {i}" for i in range(6, 22)]
            nationality = st.selectbox("Kewarganegaraan", nat_opts)
            # Membuat checkbox apakah mahasiswa adalah pindahan dari tempat lain
            displaced = st.checkbox("Pindahan (Displaced)")
            # Membuat checkbox apakah mahasiswa memiliki kebutuhan pendidikan khusus
            spec_needs = st.checkbox("Kebutuhan Khusus")
        with c2:
            edu_opts = ["1 - Pendidikan Dasar (SD/SMP)", "2 - Pendidikan Menengah (SMA/SMK)", "3 - Pendidikan Tinggi (Diploma/S1)", "4 - Pascasarjana (S2/S3)"] + [f"{i} - Kualifikasi {i}" for i in range(5, 35)]
            occ_opts = ["1 - Pegawai Swasta / Profesional", "2 - Pegawai Negeri", "3 - Wiraswasta / Pengusaha", "4 - Pekerja Lepas (Freelance)", "5 - Tidak Bekerja / Ibu Rumah Tangga"] + [f"{i} - Pekerjaan Tipe {i}" for i in range(6, 47)]
            # Membuat selectbox untuk kode level pendidikan ibu
            m_qual = st.selectbox("Pendidikan Ibu", edu_opts)
            # Membuat selectbox untuk kode level pendidikan ayah
            f_qual = st.selectbox("Pendidikan Ayah", edu_opts)
            # Membuat selectbox untuk kode pekerjaan ibu
            m_occ = st.selectbox("Pekerjaan Ibu", occ_opts)
            # Membuat selectbox untuk kode pekerjaan ayah
            f_occ = st.selectbox("Pekerjaan Ayah", occ_opts)

    st.divider() # Garis pemisah sebelum tombol prediksi
    # Membuat tombol raksasa untuk memicu proses prediksi
    predict_btn = st.button("Prediksi Cluster Sekarang", type="primary", use_container_width=True)

    # Blok logika ini dieksekusi HANYA KETIKA tombol prediksi ditekan
    if predict_btn:
        # Melakukan clipping (pembatasan batas atas) pada nilai Sem 1 mengikuti rules dataset
        s1_grd_c = float(np.clip(s1_grd, 0, 17.46))
        # Melakukan clipping pada nilai Sem 2 mengikuti rules dataset
        s2_grd_c = float(np.clip(s2_grd, 0, 16.31))
        
        # Menghitung fitur turunan (Total_Approved_Units) berdasarkan jumlah S1+S2
        total_approved = s1_app + s2_app
        # Menghitung fitur turunan (Average_Grade) gabungan rata-rata nilai semester 1 & 2
        average_grade = (s1_grd_c + s2_grd_c) / 2
        # Menghitung fitur turunan (Total_Enrolled_Units) 
        total_enrolled = s1_enr + s2_enr
        # Menghitung ratio kelulusan matkul. Jika tidak ada yang didaftarkan, rasionya 0.
        approval_ratio = total_approved / total_enrolled if total_enrolled > 0 else 0.0

        # Membuat kamus (dictionary) 'row' yang berisi seluruuh data persis format kolom dataset
        row = {
            "Marital status": int(marital.split(" - ")[0]),
            "Application mode": int(app_mode.split(" - ")[0]),
            "Application order": app_order,
            "Course": int(course.split(" - ")[0]),
            # Mengubah hasil selectbox Kehadiran menjadi bilangan bulat biner (1/0)
            "Daytime/evening attendance": 1 if "Siang" in attendance else 0,
            "Previous qualification": int(prev_qual.split(" - ")[0]),
            "Nacionality": int(nationality.split(" - ")[0]),
            "Mother's qualification": int(m_qual.split(" - ")[0]),
            "Father's qualification": int(f_qual.split(" - ")[0]),
            "Mother's occupation": int(m_occ.split(" - ")[0]),
            "Father's occupation": int(f_occ.split(" - ")[0]),
            # Mengubah hasil checkbox dari Boolean menjadi Integer (1/0)
            "Displaced": int(displaced),
            "Educational special needs": int(spec_needs),
            "Debtor": int(debtor),
            "Tuition fees up to date": int(tuition_ok),
            # Mengubah hasil selectbox Gender menjadi 1 (Laki) atau 0 (Perempuan)
            "Gender": 1 if "Laki" in gender else 0,
            "Scholarship holder": int(scholarship),
            "Age at enrollment": age,
            "Curricular units 1st sem (credited)": s1_cred,
            "Curricular units 1st sem (enrolled)": s1_enr,
            "Curricular units 1st sem (evaluations)": s1_eval,
            "Curricular units 1st sem (approved)": s1_app,
            "Curricular units 1st sem (grade)": s1_grd_c,
            "Curricular units 2nd sem (enrolled)": s2_enr,
            "Curricular units 2nd sem (evaluations)": s2_eval,
            "Curricular units 2nd sem (approved)": s2_app,
            "Curricular units 2nd sem (grade)": s2_grd_c,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
            # 4 kolom buatan hasil Feature Engineering kita
            "Total_Approved_Units": total_approved,
            "Average_Grade": average_grade,
            "Total_Enrolled_Units": total_enrolled,
            "Approval_Ratio": approval_ratio,
        }

        # Mengonversi kamus ke dalam tipe Dataframe pandas dengan 1 baris
        df_in = pd.DataFrame([row])
        # Mendefinisikan kolom mana saja yang bersifat kategorikal untuk diproses One-Hot Encoding
        categorical_cols = ['Marital status', 'Application mode', 'Course', 'Nacionality']
        # Melakukan One-Hot Encoding ke df_in agar sinkron dengan model yang dilatih
        df_ohe = pd.get_dummies(df_in, columns=categorical_cols, drop_first=True, dtype=int)
        # Menyamakan jumlah dan urutan kolom persis seperti ketika model di-training (jika ada nilai yang hilang, isi 0)
        df_ohe = df_ohe.reindex(columns=FEATURE_COLS, fill_value=0)
        
        # Menerapkan standard scaling menggunakan SCALER yang di-load
        X_sc = SCALER.transform(df_ohe)
        # Mengeksekusi model K-Means untuk memprediksi mahasiswa ini masuk ke kelompok (cluster) berapa
        cluster = int(KMEANS.predict(X_sc)[0])
        # Mengambil label / penamaan cluster dari kamus CLUSTER_NAMES berdasarkan nomor klasifikasinya
        cname   = CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")

        # Membuat sub-header untuk bagian hasil prediksi
        st.markdown("## Hasil Klasifikasi Cluster")

        # Logika if-else ini berguna untuk menentukan warna tampilan banner hasil berdasarkan risikonya
        if "Risiko Tinggi" in cname:
            # Jika mengandung kata Risiko Tinggi, gunakan CSS kelas 'result-high' (merah)
            css, icon = "result-high",   "[HIGH]"
        elif "Risiko Sedang" in cname:
            # Jika sedang, gunakan CSS kuning/oranye
            css, icon = "result-medium", "[MEDIUM]"
        else:
            # Jika rendah/aman, gunakan CSS hijau
            css, icon = "result-low",    "[LOW]"

        # Merender box warna peringatan menggunakan elemen HTML melalui st.markdown
        st.markdown(f"""
        <div class="{css}">
            <h2>{icon} {cname}</h2>
            <p style="font-size:16px;margin-top:8px">
                Mahasiswa ini dikelompokkan ke <b>Cluster {cluster}</b>
                berdasarkan kombinasi 4 faktor: Akademik, Administratif,
                Makroekonomi, dan Geografis (K-Means Champion Model).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Ringkasan 4 faktor kunci ──────────────────────────
        # Menampilkan indikator angka penting agar pengguna dapat langsung melihat summary akademiknya
        st.subheader("Ringkasan Fitur Turunan (Bab III.3)")
        c1, c2, c3, c4 = st.columns(4)
        # Metrik 1: Rasio Kelulusan
        c1.metric("Approval Ratio",   f"{approval_ratio:.2f}", "Total lulus / terdaftar")
        # Metrik 2: Rata-rata Nilai
        c2.metric("Average Grade",    f"{average_grade:.2f}",  "Rata-rata 2 semester")
        # Metrik 3: Total Matkul lulus
        c3.metric("Total MK Lulus",   f"{total_approved}",     "Sem 1 + Sem 2")
        # Metrik 4: Status Beasiswa
        c4.metric("Beasiswa", "Ya" if scholarship else "Tidak", "Status beasiswa")

        # ── Gauge probabilitas risiko ─────────────────────────
        # Menerapkan transformasi Principal Component Analysis (PCA) ke titik data input baru
        X_pca_new = PCA_MODEL.transform(X_sc)
        # Mengubah data riwayat PCA training menjadi array numpy
        X_pca_all = np.array(EVAL["X_pca_train"])
        # Menyimpan informasi hasil klasifikasi riwayat training ke numpy array
        km_lbl    = np.array(EVAL["km_labels_train"])
        
        # Menghitung rata-rata jarak titik prediksi saat ini ke semua titik dalam cluster miliknya
        dist_to_cluster = np.linalg.norm(
            X_pca_all[km_lbl == cluster] - X_pca_new, axis=1
        ).mean() if (km_lbl == cluster).sum() > 0 else 0

        # Menghitung berapa banyak mahasiswa historis di dalam cluster ini yang aslinya DROPOUT
        do_pct = EVAL["km_cluster_dist"].get(cluster, {}).get("Dropout", 0)
        # Total populasi anggota cluster tersebut
        tot    = EVAL["km_cluster_dist"].get(cluster, {}).get("n", 1)
        # Mendapatkan persentase aslinya
        risk_pct = do_pct / tot * 100

        # Membuat grafik Gauge meteran (seperti speedometer) untuk menunjukkan level risikonya
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", # Menampilkan jarum dan angka
            value=round(risk_pct, 1), # Nilai jarum menunjuk ke persentase risiko
            title={"text": "Proporsi Dropout di Cluster Ini (%)"},
            gauge={
                "axis": {"range": [0, 100]}, # Skala meteran 0-100
                "bar":  {"color": "#e74c3c"}, # Warna bar default (merah)
                "steps": [
                    {"range": [0, 30],   "color": "#d5f5e3"}, # Area aman berwarna hijau
                    {"range": [30, 60],  "color": "#fdebd0"}, # Area peringatan kekuningan
                    {"range": [60, 100], "color": "#fadbd8"}, # Area bahaya kemerahan
                ],
                # Menambahkan garis ambang batas (threshold) bahaya di angka 70% sesuai laporan
                "threshold": {"line": {"color": "red", "width": 4},
                              "thickness": 0.75, "value": 70},
            },
        ))
        # Mengatur ukuran padding layout dari plotly agar pas di web
        fig_g.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        # Merender chart tersebut ke aplikasi Streamlit
        st.plotly_chart(fig_g, use_container_width=True)
        # Keterangan pelengkap di bawah gauge
        st.caption("Garis merah = threshold kriteria bisnis 70% (Bab I.4 poin 5)")

        # ── Visualisasi Posisi Mahasiswa (Live) ───────────────
        # Subjudul untuk grafik sebaran PCA yang sifatnya real-time
        st.markdown("### Posisi Mahasiswa dalam Cluster (PCA 2D)")
        st.info("Titik bintang kuning besar di bawah ini adalah posisi mahasiswa yang baru saja Anda input, dibandingkan dengan seluruh data historis.")
        
        # Palet warna spesifik untuk 6 warna cluster historis
        palette = ["#e74c3c", "#27ae60", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
        # Inisiasi plotly graph scatter plot baru
        fig_live = go.Figure()
        
        # Plot data historis: Kita mengulang loop sebanyak N_KM (jumlah cluster)
        for c in range(N_KM):
            # Mencari titik index yang nilai k-meansnya sama dengan nomor iterasi saat ini
            mask = km_lbl == c
            # Menambahkan titik historis ini ke dalam scatter plot 
            fig_live.add_trace(go.Scatter(
                x=X_pca_all[mask, 0], y=X_pca_all[mask, 1],
                mode="markers", # Menjadikan graph sebagai titik
                marker=dict(color=palette[c % len(palette)], size=4, opacity=0.3), # opacity dibuat tipis agar jadi background
                name=f"Cluster {c} (Historis)",
                showlegend=False # Tidak perlu ditampilkan di legend karena ini background
            ))
            
        # Plot data mahasiswa baru: Kita buat 1 marker terpisah sebagai titik sorot utama
        fig_live.add_trace(go.Scatter(
            x=[X_pca_new[0, 0]], y=[X_pca_new[0, 1]], # Koordinat X dan Y data mahasiswa baru (yang tadi di-PCA_transform)
            mode="markers",
            marker=dict(color="yellow", size=18, symbol="star", line=dict(color="black", width=2)), # Diperbesar, diberi simbol bintang, warna kuning
            name="Mahasiswa Baru" # Ditampilkan di legend dengan nama "Mahasiswa Baru"
        ))
        
        # Update layout untuk Scatterplot PCA Live
        fig_live.update_layout(
            height=400,
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.2)
        )
        # Tampilkan scatter plot live ke Streamlit
        st.plotly_chart(fig_live, use_container_width=True)

        # ── Rekomendasi intervensi (Bab VI.1 & VI.2) ──────────
        # Menampilkan rekomendasi operasional untuk staf kampus berdasarkan cluster risiko
        st.markdown("### Rekomendasi Intervensi (Bab VI.2)")
        if "Risiko Tinggi" in cname:
            # Gunakan box st.error (merah) untuk memberikan aksi darurat
            st.error("""
**Intervensi Segera Diperlukan (Prioritas Tinggi):**
- Hubungi mahasiswa & wali untuk konseling akademik darurat
- Evaluasi kebutuhan beasiswa darurat / keringanan UKT
- Program mentoring intensif dan tutoring akademik
- Monitor kehadiran dan nilai setiap 2 minggu
- Identifikasi hambatan spesifik: akademik vs finansial vs personal
            """)
        elif "Risiko Sedang" in cname:
            # Gunakan box st.warning (kuning) untuk aksi pemantauan
            st.warning("""
**Pemantauan Berkala Direkomendasikan:**
- Pantau perkembangan akademik setiap akhir semester
- Konsultasi rutin dengan dosen wali akademik
- Identifikasi kendala penyelesaian tugas & mata kuliah
- Program peer mentoring dengan mahasiswa berprestasi
            """)
        else:
            # Gunakan box st.success (hijau) untuk mahasiswa aman
            st.success("""
**Pertahankan & Kembangkan Prestasi:**
- Berikan apresiasi atas pencapaian akademik
- Arahkan ke program magang / pengembangan karir
- Persiapkan administrasi wisuda
- Tawarkan beasiswa lanjutan / program S2
- Libatkan sebagai tutor/mentor mahasiswa berisiko
            """)

        # ── Detail input ──────────────────────────────────────
        # Expand header khusus jika user ingin melihat rincian array fitur yang barusan diinput
        with st.expander("Lihat Detail Data Input"):
            # Render dalam bentuk pandas tabel
            st.dataframe(
                pd.DataFrame({"Fitur": list(row.keys()), "Nilai": list(row.values())}),
                use_container_width=True, hide_index=True,
            )

# ══════════════════════════════════════════════════════════════
#  HALAMAN 2 — DASHBOARD EVALUASI
# ══════════════════════════════════════════════════════════════
# Jika user mengakses halaman kedua
elif page == "Dashboard Evaluasi":
    st.title("Dashboard Evaluasi Model Clustering")
    st.markdown(
        "Perbandingan **K-Means (Champion)** vs **Hierarchical Agglomerative Clustering** "
        "sesuai kriteria keberhasilan Bab I & Tabel IV.3, IV.4, IV.5, IV.6."
    )
    st.divider()

    # Mengambil dictionary evaluasi K-Means dan Agglomerative dari file pkl
    km_m = EVAL["kmeans"]
    ag_m = EVAL["agglo"]

    # ── Tabel IV.3: Perbandingan Metrik Clustering ─────────────
    st.subheader("Tabel IV.3 — Perbandingan Metrik Clustering")
    # Membuat tabel dataframe pandas berisi evaluasi internal seperti Silhouette dan DBI
    df_comp = pd.DataFrame({
        "Model": [f"K-Means (K={N_KM})", f"Hierarchical (K={N_AG}, {EVAL['best_ag_params']['linkage']})"],
        "Silhouette Score":    [km_m["silhouette"], ag_m["silhouette"]],
        "Davies-Bouldin Index":[km_m["dbi"],        ag_m["dbi"]],
        "Calinski-Harabasz":   [km_m["ch"],          ag_m["ch"]],
        # Membuat label ceklis keberhasilan. Silhouette targetnya >= 0.30
        "Sil ≥ 0.30": ["" if km_m["silhouette"] >= 0.30 else "Tidak",
                        "" if ag_m["silhouette"] >= 0.30 else "Tidak"],
        # Membuat label ceklis keberhasilan. DBI targetnya < 1.50
        "DBI < 1.50": ["" if km_m["dbi"] < 1.50 else "Tidak",
                        "" if ag_m["dbi"] < 1.50 else "Tidak"],
    })
    # Tampilkan tabel ini ke UI
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ── Bar chart Silhouette & DBI ─────────────────────────────
    # Membagi area jadi 2 visualisasi batang
    c1, c2 = st.columns(2)
    with c1:
        # Pembuatan Bar chart dari Plotly untuk perbandingan Silhouette model kmeans vs agglo
        fig = go.Figure(go.Bar(
            x=[f"K-Means (K={N_KM})", f"Hierarchical (K={N_AG})"],
            y=[km_m["silhouette"], ag_m["silhouette"]],
            marker_color=["#3498db", "#95a5a6"],
            text=[f"{km_m['silhouette']:.4f}", f"{ag_m['silhouette']:.4f}"],
            textposition="outside", # Text dimunculkan di atas batang
        ))
        # Menambahkan garis putus-putus merah di koordinat 0.30 sebagai standar minimal (threshold)
        fig.add_hline(y=0.30, line_dash="dash", line_color="red",
                      annotation_text="Target ≥ 0.30")
        fig.update_layout(title="Silhouette Score", height=320,
                          yaxis_range=[0, max(km_m["silhouette"], ag_m["silhouette"]) + 0.1])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Pembuatan Bar chart dari Plotly untuk perbandingan DBI model
        fig2 = go.Figure(go.Bar(
            x=[f"K-Means (K={N_KM})", f"Hierarchical (K={N_AG})"],
            y=[km_m["dbi"], ag_m["dbi"]],
            marker_color=["#3498db", "#95a5a6"],
            text=[f"{km_m['dbi']:.4f}", f"{ag_m['dbi']:.4f}"],
            textposition="outside",
        ))
        # Menambahkan batas ambang DBI maksimal yaitu 1.50
        fig2.add_hline(y=1.50, line_dash="dash", line_color="red",
                       annotation_text="Target < 1.50")
        # Semakin kecil grafik DBI, semakin baik (ditulis pada title)
        fig2.update_layout(title="Davies-Bouldin Index (↓ lebih baik)", height=320,
                           yaxis_range=[0, max(km_m["dbi"], ag_m["dbi"]) + 0.3])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tabel IV.6: Metrik Klasifikasi ────────────────────────
    # Jika kita berasumsi clustering ini sebagai masalah Supervised, ini adalah tabulasi akurasi nya
    st.subheader("Tabel IV.6 — Perbandingan Metrik Evaluasi")
    df_eval = pd.DataFrame({
        "Model":     [f"K-Means (Champion, K={N_KM})", f"Hierarchical (K={N_AG})"],
        "Accuracy":  [f"{km_m['accuracy']*100:.2f}%",  f"{ag_m['accuracy']*100:.2f}%"],
        "Precision": [f"{km_m['precision']*100:.2f}%", f"{ag_m['precision']*100:.2f}%"],
        "Recall":    [f"{km_m['recall']*100:.2f}%",    f"{ag_m['recall']*100:.2f}%"],
        "F1-Score":  [f"{km_m['f1']*100:.2f}%",        f"{ag_m['f1']*100:.2f}%"],
    })
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

    # ── Confusion Matrix ───────────────────────────────────────
    # Memvisualisasikan Heatmap Heatrix Kebingungan
    st.subheader("Confusion Matrix (Tabel IV.4 & IV.5)")
    # Membagi jadi dua tab (K-Means dan Agglomerative) agar user bisa beralih (switch)
    tab_km, tab_ag = st.tabs([f"K-Means (K={N_KM})", f"Hierarchical (K={N_AG})"])

    # Looping yang mengeksekusi logika render yang sama untuk kedua tab tersebut
    for tab, mkey, label in [(tab_km, "kmeans", f"K-Means K={N_KM}"),
                              (tab_ag, "agglo",  f"Hierarchical K={N_AG}")]:
        with tab:
            # Ambil nilai matriks dari dictionary eval
            cm   = np.array(EVAL[mkey]["confusion_matrix"])
            nc   = EVAL[mkey]["n_clusters"]
            
            # Mendefinisikan penamaan sumbu X dan Y dinamis bergantung jumlah klasternya
            xlbl = CLASS_NAMES[:nc] if nc <= 3 else [f"Pred Cls {i}" for i in range(nc)]
            ylbl = CLASS_NAMES[:nc] if nc <= 3 else [f"Act Cls {i}"  for i in range(nc)]
            
            # Membuat Heatmap via Plotly Express (px.imshow)
            fig_cm = px.imshow(
                cm, text_auto=True, # text_auto menampilkan angka matriks di dalam sel heatmap
                x=[f"Pred {c}" for c in CLASS_NAMES], # Label horizontal
                y=[f"Act {c}"  for c in CLASS_NAMES], # Label vertikal (Kenyataan sebenarnya)
                color_continuous_scale="Blues", # Memilih palet warna biru
                title=f"Confusion Matrix — {label}",
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Menampilkan kembali metrik Supervised terkait Confusion Matrix tersebut di bawahnya
            m = EVAL[mkey]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
            c2.metric("Precision", f"{m['precision']*100:.2f}%")
            c3.metric("Recall",    f"{m['recall']*100:.2f}%")
            c4.metric("F1-Score",  f"{m['f1']*100:.2f}%")

    # ── Distribusi cluster K-Means ─────────────────────────────
    # Subheader untuk porsi perwakilan status setiap kluster yang terbentuk
    st.subheader("Distribusi Status Mahasiswa per Cluster (K-Means)")
    # km_dist memegang info berapa jumlah yg DO/Lulus/Tedaftar per masing-masing kluster K-means
    km_dist = EVAL["km_cluster_dist"]
    rows = []
    # Looping untuk menghasilkan data setiap baris per cluster yang ada
    for c in range(N_KM):
        d   = km_dist[c]
        tot = d["n"] # Jumlah orang total di dalam cluster c
        rows.append({
            "Cluster": f"Cluster {c} — {CLUSTER_NAMES.get(c, '')}",
            "Total":   tot,
            # Format tampilan string agar ada jumlah asli dan (persentasenya)
            "Dropout":  f"{d['Dropout']} ({d['Dropout']/tot*100:.1f}%)",
            "Enrolled": f"{d['Enrolled']} ({d['Enrolled']/tot*100:.1f}%)",
            "Graduate": f"{d['Graduate']} ({d['Graduate']/tot*100:.1f}%)",
        })
    # Tampilkan ke layar dalam wujud Dataframe
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Stacked Bar Chart Distribusi  ─────────────────────────
    # Visualisasi barmode=stack (bertumpuk) untuk sebaran profil Dropout, Enrolled, Graduate
    fig_s = go.Figure()
    clrs  = ["#e74c3c", "#f39c12", "#27ae60"] # Warna: Merah (DO), Oranye (Terdaftar), Hijau (Lulus)
    
    # Memproses penambahan stack per tipe klasifikasi asli
    for i, cls in enumerate(["Dropout", "Enrolled", "Graduate"]):
        # Mengkalkulasi persentase rasio cls terhadap populasi total dari tiap-tiap cluster
        vals = [km_dist[c][cls] / km_dist[c]["n"] * 100 for c in range(N_KM)]
        # Menambahkan Trace (grafik per layer stack)
        fig_s.add_trace(go.Bar(
            name=cls,
            # Menamai kolom horizontal (Sumbu X) dengan label dari penamaan cluster kami
            x=[f"Cluster {c}\n{CLUSTER_NAMES.get(c,'').split('(')[-1].rstrip(')')}"
               for c in range(N_KM)],
            y=vals, marker_color=clrs[i],
            # Meletakkan text persentase agar berada di dalam bar graph tersebut
            text=[f"{v:.1f}%" for v in vals], textposition="inside",
        ))
    
    # Konfigurasi mode "stack" agar grafiknya tertumpuk ke atas membentuk 100% tinggi yang sama (atau bervariasi)
    fig_s.update_layout(barmode="stack", height=380,
                         title="Proporsi Status per Cluster (%)",
                         yaxis_title="Persentase (%)",
                         legend=dict(orientation="h", y=1.12)) # Memposisikan Legenda warna horisontal di atas graph
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Validasi Kriteria Bisnis  ───────────────────────────────
    # Mengecek apakah K-Means sukses mendeteksi minimal 1 Cluster yang probabilitas Dropout-nya > 70% (kriteria laporan)
    st.markdown("### Pencapaian Kriteria Keberhasilan (Bab I.4)")
    for c in range(N_KM):
        d      = km_dist[c]
        do_pct = d["Dropout"] / d["n"] * 100
        # Syarat bisnis terpenuhi jika persentase Dropout kluster tertentu melampaui angka 70%
        if do_pct > 70:
            st.success(
                f"**Cluster {c} ({CLUSTER_NAMES.get(c, '')})** — "
                f"Dropout {do_pct:.1f}% > 70% Kriteria Bisnis Terpenuhi (Bab I.4 poin 5)"
            )

    # Sebuah panel info (biru) sebagai resume performa Champion Model kita
    st.info(
        f"**Champion Model: K-Means (K={N_KM}, {EVAL['best_km_params']['init']})** — "
        f"Silhouette {km_m['silhouette']:.4f} | DBI {km_m['dbi']:.4f} "
    )

# ══════════════════════════════════════════════════════════════
#  HALAMAN 3 — VISUALISASI CLUSTER
# ══════════════════════════════════════════════════════════════
# Jika user membuka Tab Halaman "Visualisasi Cluster"
elif page == "Visualisasi Cluster":
    st.title("Visualisasi Cluster — Proyeksi PCA 2D")
    st.markdown(
        "Proyeksi data ke **2 dimensi** menggunakan PCA (Bab IV.3) untuk "
        "visualisasi pemisahan cluster sesuai Gambar IV.5 & IV.6 laporan."
    )
    st.divider()

    # Memuat koordinat array scatterplot yang dihasilkan dari pipeline Principal Component Analysis (PCA)
    X_pca      = np.array(EVAL["X_pca_train"])
    # Memuat label assignment (prediksi K-Means) sebagai referensi warna
    km_lbl     = np.array(EVAL["km_labels_train"])
    # Memuat label assignment Agglomerative 
    ag_lbl     = np.array(EVAL["ag_labels_train"])
    # Memuat label kelas asli (Kenyataan dataset)
    y_train    = np.array(EVAL["y_train"])
    # Persentase Variance PCA Component untuk dimunculkan sebagai label X dan Y Axis
    var_exp    = EVAL["pca_variance"]

    # Membuat 3 Navigation Tabs khusus visualisasi scatter plot ini
    tab1, tab2, tab3 = st.tabs([
        "K-Means (Gambar IV.5)",       # Tab 1 untuk K-means
        "Hierarchical (Gambar IV.6)",  # Tab 2 untuk Agglomerative
        "Status Asli",                 # Tab 3 untuk Ground Truth (Kenyataan Asli)
    ])

    # Logika untuk Tab 1 (Scatter plot K-Means)
    with tab1:
        st.subheader(f"K-Means Clustering (K={N_KM}) — PCA 2D")
        palette = ["#e74c3c", "#27ae60", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
        fig_km  = go.Figure()
        # Melakukan plot scatter secara iteratif (per warna kluster)
        for c in range(N_KM):
            mask = km_lbl == c # Menyaring elemen array khusus yang nilainya "c"
            fig_km.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1], # X adalah Komponen 1 PCA, Y adalah komponen 2
                mode="markers",                     # Titik-titik
                marker=dict(color=palette[c % len(palette)], size=4, opacity=0.6),
                name=f"Cluster {c}: {CLUSTER_NAMES.get(c, f'Cluster {c}')}", # Penamaan untuk di Legend
            ))
        # Blok opsional untuk melukiskan titik tengah (Centroid) dari perhitungan model K-means
        try:
            # Lakukan Transformasi PCA pada nilai murni centroid original dimensi-N agar bisa di-plot 2D
            cent_pca = PCA_MODEL.transform(KMEANS.cluster_centers_)
            fig_km.add_trace(go.Scatter(
                x=cent_pca[:, 0], y=cent_pca[:, 1],
                mode="markers",
                marker=dict(symbol="star", size=16, color="black", # Gunakan simbol bintang besar
                            line=dict(width=1, color="white")),
                name="Centroids ",
            ))
        except Exception:
            pass # Lewati jika gagal 

        # Mengatur konfigurasi layout grafik agar lebih bersih dan memiliki title sumbu yg tepat
        fig_km.update_layout(
            height=520,
            xaxis_title=f"PCA Component 1 ({var_exp[0]*100:.1f}%)", # Sumbu Horizontal adalah Komponen Pertama
            yaxis_title=f"PCA Component 2 ({var_exp[1]*100:.1f}%)", # Sumbu Vertikal adalah Komponen Kedua
            title=f"Visualisasi K-Means K={N_KM} (PCA 2D Projection)",
            legend=dict(orientation="h", y=-0.18), # Posisi legenda
        )
        st.plotly_chart(fig_km, use_container_width=True)

    # Logika untuk Tab 2 (Scatter plot Agglomerative)
    with tab2:
        st.subheader(f"Hierarchical Clustering (K={N_AG}, {EVAL['best_ag_params']['linkage']}) — PCA 2D")
        palette_ag = ["#9b59b6", "#f1c40f", "#34495e", "#e67e22", "#1abc9c", "#e74c3c"]
        fig_ag = go.Figure()
        # Mengulang blok pembuatan Scatter plot, namun memisahkan data berdasarkan `ag_lbl` (Label agglo)
        for c in range(N_AG):
            mask = ag_lbl == c
            fig_ag.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                marker=dict(color=palette_ag[c % len(palette_ag)], size=4, opacity=0.6),
                name=f"Agglo Cluster {c}",
            ))
        fig_ag.update_layout(
            height=520,
            xaxis_title=f"PCA Component 1 ({var_exp[0]*100:.1f}%)",
            yaxis_title=f"PCA Component 2 ({var_exp[1]*100:.1f}%)",
            title=f"Visualisasi Hierarchical K={N_AG} Ward (PCA 2D Projection)",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_ag, use_container_width=True)

    # Logika untuk Tab 3 (Distribusi Data Target Aslinya di UCI)
    with tab3:
        st.subheader("Distribusi Status Mahasiswa Asli — PCA 2D")
        colors_t = {"Dropout": "#e74c3c", "Enrolled": "#f39c12", "Graduate": "#27ae60"}
        fig_t = go.Figure()
        # Memilah titik scatter berdasarkan status y_train aslinya (Bukan hasil prediksi AI)
        for i, cls in enumerate(CLASS_NAMES):
            mask = y_train == i
            fig_t.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                marker=dict(color=colors_t.get(cls, "#888"), size=4, opacity=0.6),
                name=cls,
            ))
        fig_t.update_layout(
            height=520,
            xaxis_title=f"PCA Component 1 ({var_exp[0]*100:.1f}%)",
            yaxis_title=f"PCA Component 2 ({var_exp[1]*100:.1f}%)",
            title="Distribusi Status Mahasiswa Asli (PCA 2D)",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ── Profil rata-rata fitur per cluster (Bar Chart) ────────────────
    st.divider()
    st.subheader("Profil Rata-Rata Cluster K-Means (Standardized)")
    # Mengambil dictionary dari hasil train untuk diproses ke tabel
    km_dist   = EVAL["km_cluster_dist"]
    km_prof   = EVAL.get("km_profiles", {})
    prof_rows = []
    
    # Looping pembentukan tabel rekapan demografi jumlah mahasiswa masing-masing kluster K-Means
    for c in range(N_KM):
        d   = km_dist[c]
        tot = d["n"]
        prof_rows.append({
            "Cluster": f"Cluster {c}",
            "Nama":    CLUSTER_NAMES.get(c, ""),
            "Jumlah":  tot,
            "% Dropout":  f"{d['Dropout']/tot*100:.1f}%",
            "% Enrolled": f"{d['Enrolled']/tot*100:.1f}%",
            "% Graduate": f"{d['Graduate']/tot*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True)

    # Jika km_prof (profile centroid fitur) terekam, tampilkan sebagai Grouped Bar Chart
    if km_prof:
        # Array berisi nama-nama metrik (misal: age, grade, dsb)
        feat_lbls = list(km_prof[0].keys())
        palette_p = ["#e74c3c", "#27ae60", "#3498db", "#f39c12"]
        fig_p = go.Figure()
        
        # Ekstrak nilai standar (Z-score) untuk setiap Cluster dari dictionary
        for c in range(N_KM):
            vals = [km_prof[c].get(f, 0) for f in feat_lbls]
            fig_p.add_trace(go.Bar(
                name=f"Cluster {c}: {CLUSTER_NAMES.get(c, '')}",
                x=feat_lbls, y=vals, # x berupa nama kolom (feature), y berupa nilainya (seberapa menyimpang ia dari rata-rata total)
                marker_color=palette_p[c % len(palette_p)],
            ))
        
        # update mode barmode='group' agar batang grafik berdampingan (tidak menumpuk)
        fig_p.update_layout(
            barmode="group", height=420,
            xaxis_tickangle=-30, # Memiringkan label sumbu X agar tidak bertabrakan jika fitur panjang
            title="Profil Rata-Rata Fitur per Cluster (nilai standardized)",
            yaxis_title="Nilai Standardized",
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  HALAMAN 4 — TENTANG
# ══════════════════════════════════════════════════════════════
# Tab / Halaman statis yang berisi Markdown narasi dan info tim pembuat
elif page == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown(f"""
## Prediksi Dropout & Keberhasilan Akademik Mahasiswa

Aplikasi prototipe **deployment** Tugas Besar Mata Kuliah **Penambangan Data**  
Program Studi S1 Sistem Informasi, Fakultas Rekayasa Industri, **Universitas Telkom 2026**.

---

### Metodologi CRISP-DM (6 Tahapan)
| Bab | Tahapan | Keterangan |
|---|---|---|
| I | Business Understanding | Dropout 32,1% dari 4.424 mhs = masalah kritis |
| II | Data Understanding | EDA 4 faktor: Akademik, Administratif, Makro, Geografis |
| III | Data Preparation | Drop 4 fitur redundan, IQR Capping, 4 fitur baru, scaling |
| IV | Modeling | K-Means + Hierarchical, K=optimal via Elbow Method |
| V | Evaluation | Silhouette, DBI, Calinski-Harabasz, Confusion Matrix |
| VI | Deployment | Aplikasi Streamlit ini |

---

### Dataset
| Info | Detail |
|---|---|
| Sumber | UCI Machine Learning Repository / Kaggle |
| Asal | Polytechnic Institute of Portalegre, Portugal |
| Ukuran | 4.424 mahasiswa × 35 fitur |
| Target | Dropout 32,1% · Graduate 49,9% · Enrolled 17,9% |

---

### Rekayasa Fitur (Bab III.3, Tabel III.4)
| Fitur Baru | Deskripsi |
|---|---|
| `Total_Approved_Units` | Total MK lulus Sem 1 + Sem 2 |
| `Average_Grade` | Rata-rata nilai kedua semester |
| `Total_Enrolled_Units` | Total MK terdaftar Sem 1 + Sem 2 |
| `Approval_Ratio` | Total_Approved / Total_Enrolled |

---

### Model (Bab IV)
| Model | Paradigma | Peran | K Optimal |
|---|---|---|---|
| **K-Means** | Partisi | **Champion Model** | {N_KM} |
| **Hierarchical (Ward)** | Hierarki | Comparison Model | {N_AG} |

---

### Pencapaian Kriteria Keberhasilan (Tabel I.1)
| No | Dimensi | Indikator | Target | Hasil |
|---|---|---|---|---|
| 1 | Teknis | Silhouette Score | ≥ 0.30 | {EVAL['kmeans']['silhouette']} {'' if EVAL['kmeans']['silhouette']>=0.30 else 'Tidak'} |
| 2 | Teknis | Davies-Bouldin Index | < 1.50 | {EVAL['kmeans']['dbi']} {'' if EVAL['kmeans']['dbi']<1.50 else 'Tidak'} |
| 3 | Bisnis | Cluster Dropout | > 70% | Terpenuhi |
| 4 | Teknis | Reprodusibilitas | random_state=42 | Ditetapkan |

---

### Tim Pengembang
| NIM | Nama |
|---|---|
| 102022400122 | Michael William Setiawan Wee |
| 102022400020 | Muhammad Lukman Hakim |
| 102022400195 | Andika Gilang Ramadan |
| 102022400170 | Muhammad Rafi Fazlullah Pasya |

---

### Referensi
- Villar & Andrade (2024). *Supervised ML for predicting student dropout.* Discov Artif Intell.
- Porras et al. (2023). *ML Approaches for Predicting Dropout.* Algorithms, 16(12).
- Nagy & Molontay (2024). *Interpretable Dropout Prediction.* Int J Artif Intell Educ.
- Won et al. (2023). *University Student Dropout Prediction.* Applied Sciences, 13(12).
""")

# ── FOOTER ────────────────────────────────────────────────────
# Menutup aplikasi dengan garis pembatas dan catatan kaki untuk identitas laporan
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px'>"
    "Tugas Besar Penambangan Data · Universitas Telkom 2026 · "
    "K-Means & Hierarchical Clustering · Dataset: UCI/Kaggle Portalegre Portugal</p>",
    unsafe_allow_html=True,
)
