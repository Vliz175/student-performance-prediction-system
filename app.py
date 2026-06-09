# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Analisis Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATA INISIALISASI
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('student-por.csv', sep=';')
    return df

@st.cache_resource
def load_classification_model():
    try:
        model = joblib.load("classification/model_gradient_boosting.pkl")
        fitur = joblib.load("classification/fitur_kelulusan.pkl")
        mapping = joblib.load("classification/mapping_kelulusan.pkl")
        return model, fitur, mapping
    except:
        return None, None, None

@st.cache_resource
def load_clustering_model():
    try:
        model = joblib.load("clustering/model_clustering.pkl")
        scaler = joblib.load("clustering/scaler_clustering.pkl")
        fitur = joblib.load("clustering/fitur_clustering.pkl")
        nama = joblib.load("clustering/nama_segment.pkl")
        return model, scaler, fitur, nama
    except:
        return None, None, None, None

# Load semua data
df = load_data()
clf_model, clf_fitur, clf_mapping = load_classification_model()
clust_model, clust_scaler, clust_fitur, clust_nama = load_clustering_model()

# ============================================
# SIDEBAR NAVIGASI
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.markdown("### 🎓 Sistem Analisis Siswa")
    st.markdown("---")
    
    menu = st.radio(
        "Navigasi",
        ["🏠 Home", "📊 Dataset Overview", "🎯 Prediction / Analysis", "📈 Visualization", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    st.caption("© 2026 - UAS Data Mining")

# ============================================
# 1. HOME
# ============================================
if menu == "🏠 Home":
    st.title("🎓 Sistem Analisis Performa Siswa")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Deskripsi Proyek
    
    Aplikasi ini dibuat untuk menganalisis dan memprediksi performa siswa diberdasarkan data akademik, 
    kebiasaan belajar, dan faktor sosial. Terdapat dua fitur utama:
    
    - **Prediksi Kelulusan**: Memprediksi apakah siswa akan LULUS atau TIDAK LULUS
    - **Segmentasi Performa**: Mengelompokkan siswa ke dalam 4 kategori performa
    
    ### 🎯 Tujuan
    
    - Membantu guru/orang tua memantau performa siswa
    - Memberikan rekomendasi peningkatan akademik
    - Mengidentifikasi siswa yang membutuhkan bimbingan khusus
    
    """)
    
    st.markdown("---")
    st.subheader("👥 Identitas Anggota")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Anggota 1**
        - Nama: Dwika Wijaya Ardana
        - NIM: 24051214161
        - Prodi: Sistem Informasi
        """)
    
    with col2:
        st.markdown("""
        **Anggota 2**
        - Nama: Fauzan Zaki
        - NIM: 24051214144
        - Prodi: Sistem Informasi
        """)
    
    st.markdown("---")
    st.info("💡 **Petunjuk:** Gunakan menu di sidebar untuk navigasi ke halaman lainnya.")

# ============================================
# 2. DATASET OVERVIEW
# ============================================
elif menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")
    
    # Informasi dataset
    st.subheader("📋 Informasi Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Siswa", f"{len(df)}")
    with col2:
        st.metric("Total Fitur", f"{len(df.columns)}")
    with col3:
        st.metric("Mata Pelajaran", "Bahasa Portugis")
    with col4:
        st.metric("Skala Nilai", "0 - 20")
    
    # Statistik target
    st.subheader("📊 Statistik Nilai Akhir (G3)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rata-rata G3", f"{df['G3'].mean():.2f}")
    with col2:
        st.metric("Nilai Tertinggi", f"{df['G3'].max()}")
    with col3:
        st.metric("Nilai Terendah", f"{df['G3'].min()}")
    
    # Distribusi kelulusan
    df['lulus'] = (df['G3'] >= 10).astype(int)
    lulus_count = df['lulus'].value_counts()
    
    fig1, ax1 = plt.subplots()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Lulus (G3 ≥ 10)', 'Tidak Lulus (G3 < 10)']
    ax1.pie(lulus_count.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribusi Kelulusan Siswa')
    st.pyplot(fig1)
    
    # Preview data
    st.subheader("📋 Preview Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik deskriptif
    with st.expander("📈 Statistik Deskriptif"):
        st.dataframe(df.describe(), use_container_width=True)
    
    with st.expander("ℹ️ Keterangan Kolom"):
        st.markdown("""
        | Kolom | Keterangan |
        |-------|-------------|
        | G1 | Nilai periode pertama (0-20) |
        | G2 | Nilai periode kedua (0-20) |
        | G3 | Nilai akhir (0-20) - TARGET |
        | studytime | Waktu belajar per minggu (1-4) |
        | failures | Jumlah gagal kelas (0-4) |
        | absences | Jumlah absensi (0-93) |
        | goout | Frekuensi hangout dengan teman (1-5) |
        | famrel | Kualitas hubungan keluarga (1-5) |
        | health | Status kesehatan (1-5) |
        | higher | Minat pendidikan tinggi (yes/no) |
        """)

# ============================================
# 3. PREDICTION / ANALYSIS
# ============================================
elif menu == "🎯 Prediction / Analysis":
    st.title("🎯 Prediction & Analysis")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Prediksi Kelulusan", "🔍 Segmentasi Performa"])
    
    # ========== TAB 1: PREDIKSI KELULUSAN ==========
    with tab1:
        st.subheader("Prediksi Kelulusan Siswa")
        st.markdown("Masukkan data siswa untuk memprediksi apakah akan **LULUS** atau **TIDAK LULUS**")
        
        if clf_model is None:
            st.error("⚠️ Model tidak tersedia. Pastikan file model ada di folder `classification/`")
        else:
            with st.form("form_prediksi"):
                col1, col2 = st.columns(2)
                
                with col1:
                    g2 = st.number_input("Nilai periode 2 (G2)", 0, 20, 12)
                    g1 = st.number_input("Nilai periode 1 (G1)", 0, 20, 11)
                    failures = st.slider("Jumlah gagal kelas", 0, 4, 0)
                    absences = st.slider("Jumlah absensi", 0, 50, 5)
                
                with col2:
                    studytime = st.selectbox("Waktu belajar", ["<2 jam", "2-5 jam", "5-10 jam", ">10 jam"])
                    goout = st.slider("Frekuensi hangout (1-5)", 1, 5, 3)
                    freetime = st.slider("Waktu luang (1-5)", 1, 5, 3)
                    health = st.slider("Status kesehatan (1-5)", 1, 5, 3)
                
                col3, col4 = st.columns(2)
                with col3:
                    higher = st.radio("Ingin melanjutkan kuliah?", ["Ya", "Tidak"], horizontal=True)
                with col4:
                    internet = st.radio("Akses internet di rumah?", ["Ada", "Tidak"], horizontal=True)
                
                submitted = st.form_submit_button("🔮 Prediksi Kelulusan", type="primary", use_container_width=True)
            
            if submitted:
                waktu_map = {"<2 jam":1, "2-5 jam":2, "5-10 jam":3, ">10 jam":4}
                
                data = {
                    'G1': g1, 'G2': g2, 'failures': failures, 'absences': absences,
                    'studytime': waktu_map[studytime], 'goout': goout, 'freetime': freetime,
                    'health': health, 'higher': 1 if higher=="Ya" else 0,
                    'internet': 1 if internet=="Ada" else 0
                }
                
                input_df = pd.DataFrame([[data[f] for f in clf_fitur]], columns=clf_fitur)
                pred = clf_model.predict(input_df)[0]
                proba = clf_model.predict_proba(input_df)[0]
                
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                
                with col_r1:
                    if pred == 1:
                        st.success("### 🎉 LULUS")
                        st.write("Selamat! Siswa diprediksi akan lulus.")
                    else:
                        st.error("### ❌ TIDAK LULUS")
                        st.write("Siswa perlu bimbingan lebih intensif.")
                
                with col_r2:
                    st.metric("Probabilitas Lulus", f"{proba[1]:.1%}")
                
                with col_r3:
                    st.metric("Model", "Gradient Boosting")
                
                # Saran
                st.markdown("---")
                st.subheader("💡 Saran Peningkatan")
                
                if pred == 1:
                    st.success("✅ Pertahankan prestasi Anda!")
                else:
                    if waktu_map[studytime] < 3:
                        st.warning("📚 Tingkatkan waktu belajar menjadi 5-10 jam per minggu")
                    if failures > 0:
                        st.warning("📖 Ikuti bimbingan belajar untuk mata pelajaran yang sulit")
                    if absences > 10:
                        st.warning("🎯 Kurangi absensi, kehadiran sangat penting")
                    if goout > 3:
                        st.warning("🎮 Kurangi waktu bermain dengan teman")
    
    # ========== TAB 2: SEGMENTASI ==========
    with tab2:
        st.subheader("Segmentasi Performa Siswa")
        st.markdown("Masukkan data siswa untuk mengetahui kategori performa mereka")
        
        if clust_model is None:
            st.error("⚠️ Model tidak tersedia. Pastikan file model ada di folder `clustering/`")
        else:
            with st.form("form_cluster"):
                col1, col2 = st.columns(2)
                
                with col1:
                    g1 = st.number_input("Nilai periode 1 (G1)", 0, 20, 12, key="seg_g1")
                    g2 = st.number_input("Nilai periode 2 (G2)", 0, 20, 12, key="seg_g2")
                    g3 = st.number_input("Nilai akhir (G3)", 0, 20, 12, key="seg_g3")
                    failures = st.slider("Jumlah gagal kelas", 0, 4, 0, key="seg_fail")
                    absences = st.slider("Jumlah absensi", 0, 50, 5, key="seg_abs")
                
                with col2:
                    studytime = st.selectbox("Waktu belajar", ["<2 jam", "2-5 jam", "5-10 jam", ">10 jam"], key="seg_time")
                    goout = st.slider("Frekuensi hangout (1-5)", 1, 5, 3, key="seg_go")
                    freetime = st.slider("Waktu luang (1-5)", 1, 5, 3, key="seg_free")
                    famrel = st.slider("Hubungan keluarga (1-5)", 1, 5, 4, key="seg_fam")
                    health = st.slider("Status kesehatan (1-5)", 1, 5, 4, key="seg_health")
                
                submitted_clust = st.form_submit_button("🔍 Analisis Segmentasi", type="primary", use_container_width=True)
            
            if submitted_clust:
                waktu_map = {"<2 jam":1, "2-5 jam":2, "5-10 jam":3, ">10 jam":4}
                
                data = {
                    'G1': g1, 'G2': g2, 'G3': g3, 'studytime': waktu_map[studytime],
                    'failures': failures, 'absences': absences, 'goout': goout, 'freetime': freetime,
                    'Dalc': 1, 'Walc': 2, 'famrel': famrel, 'health': health
                }
                
                input_df = pd.DataFrame([[data[f] for f in clust_fitur]], columns=clust_fitur)
                input_scaled = clust_scaler.transform(input_df)
                cluster = clust_model.predict(input_scaled)[0]
                segment = clust_nama[cluster]
                
                st.markdown("---")
                st.subheader("📊 Hasil Segmentasi")
                
                if "SANGAT BAIK" in segment:
                    st.success(f"### 🌟 {segment}")
                    st.write("Siswa dengan prestasi akademik tertinggi. Pertahankan!")
                elif "BAIK" in segment:
                    st.info(f"### ✅ {segment}")
                    st.write("Siswa dengan potensi bagus. Tingkatkan lagi!")
                elif "CUKUP" in segment:
                    st.warning(f"### 📘 {segment}")
                    st.write("Siswa dengan performa sedang. Perlu motivasi!")
                else:
                    st.error(f"### ⚠️ {segment}")
                    st.write("Siswa dengan performa rendah. Butuh bimbingan intensif!")

# ============================================
# 4. VISUALIZATION
# ============================================
elif menu == "📈 Visualization":
    st.title("📈 Visualisasi Data")
    st.markdown("---")
    
    tab_viz1, tab_viz2, tab_viz3 = st.tabs(["📊 Distribusi Data", "📈 Korelasi Fitur", "🎯 Hasil Analisis"])
    
    with tab_viz1:
        st.subheader("Distribusi Nilai Siswa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='G3', nbins=20, title='Distribusi Nilai Akhir (G3)', 
                               color_discrete_sequence=['#667eea'])
            fig.update_layout(xaxis_title='Nilai', yaxis_title='Jumlah Siswa')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y='G3', title='Boxplot Nilai Akhir (G3)',
                         color_discrete_sequence=['#764ba2'])
            fig.update_layout(yaxis_title='Nilai')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Distribusi Fitur Lainnya")
        
        fitur_pilih = st.selectbox("Pilih fitur:", ['studytime', 'failures', 'absences', 'goout', 'freetime', 'health'])
        
        if fitur_pilih == 'studytime':
            labels = {1:'<2 jam', 2:'2-5 jam', 3:'5-10 jam', 4:'>10 jam'}
            df[fitur_pilih + '_label'] = df[fitur_pilih].map(labels)
            fig = px.bar(df[fitur_pilih + '_label'].value_counts(), title=f'Distribusi {fitur_pilih}',
                         color_discrete_sequence=['#2ecc71'])
        else:
            fig = px.histogram(df, x=fitur_pilih, title=f'Distribusi {fitur_pilih}',
                               color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_viz2:
        st.subheader("Korelasi Antar Fitur")
        
        # Pilih kolom numerik
        numeric_cols = ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'goout', 'freetime', 'famrel', 'health']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', 
                        title='Heatmap Korelasi Fitur',
                        color_continuous_scale='RdBu')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretasi Korelasi:**
        - **G1 dan G2** memiliki korelasi tinggi dengan G3 (nilai akhir)
        - **failures** berkorelasi negatif dengan nilai (semakin banyak gagal, nilai semakin rendah)
        - **absences** juga berkorelasi negatif dengan nilai
        """)
    
    with tab_viz3:
        st.subheader("Visualisasi Hasil Analisis")
        
        if clust_model is not None:
            # Prediksi cluster untuk semua data
            X_clust = df[clust_fitur].copy()
            X_clust_scaled = clust_scaler.transform(X_clust)
            clusters = clust_model.predict(X_clust_scaled)
            
            df['Cluster'] = clusters
            df['Segment'] = df['Cluster'].map(clust_nama)
            
            # Scatter plot
            fig = px.scatter(df, x='G1', y='G3', color='Segment', 
                            title='Segmentasi Siswa (G1 vs G3)',
                            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                            hover_data=['studytime', 'absences', 'failures'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart distribusi segment
            segment_counts = df['Segment'].value_counts()
            fig = px.bar(segment_counts, title='Jumlah Siswa per Segment',
                         color=segment_counts.index,
                         color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
            fig.update_layout(xaxis_title='Segment', yaxis_title='Jumlah Siswa')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# 5. ABOUT
# ============================================
else:
    st.title("ℹ️ About")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Metode yang Digunakan
        
        **1. Prediksi Kelulusan**
        - **Algoritma:** Gradient Boosting Classifier
        - **Fungsi:** Memprediksi apakah siswa lulus atau tidak
        - **Target:** LULUS (G3 ≥ 10) atau TIDAK LULUS (G3 < 10)
        - **Akurasi:** ~85-88%
        
        **2. Segmentasi Performa**
        - **Algoritma:** K-Means Clustering
        - **Fungsi:** Mengelompokkan siswa ke dalam 4 kategori
        - **Jumlah Cluster:** 4 (Sangat Baik, Baik, Cukup, Kurang)
        - **Metrik:** Silhouette Score, Davies-Bouldin
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Dataset
        
        **Sumber Data:** Student Performance Dataset (Portugal)
        
        **Jumlah Data:** 649 siswa
        
        **Mata Pelajaran:** Bahasa Portugis
        
        **Skala Nilai:** 0 - 20
        
        **Fitur Utama:**
        - G1, G2, G3 (Nilai)
        - studytime (Waktu belajar)
        - failures (Kegagalan)
        - absences (Absensi)
        - goout, freetime (Aktivitas sosial)
        - famrel (Hubungan keluarga)
        - health (Kesehatan)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### ℹ️ Informasi Proyek
    
    | Aspek | Keterangan |
    |-------|-------------|
    | **Judul Proyek** | Sistem Analisis Performa Siswa |
    | **Mata Kuliah** | Data Mining |
    | **Tahun** | 2026 |
    | **Tools** | Python, Streamlit, Scikit-learn, Plotly |
    
    ### 📁 Struktur Aplikasi
    project/
    ├── app.py
    ├── classification/
    │ ├── model_gradient_boosting.pkl
    │ ├── fitur_kelulusan.pkl
    │ └── mapping_kelulusan.pkl
    └── clustering/
    ├── model_clustering.pkl
    ├── scaler_clustering.pkl
    ├── fitur_clustering.pkl
    └── nama_segment.pkl
                """)

st.info("💡 **Cara Penggunaan:** Pilih menu di sidebar untuk mengakses fitur-fitur aplikasi.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2026 - UAS Data Mining | Sistem Analisis Performa Siswa</p>", unsafe_allow_html=True)