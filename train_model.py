"""
train_model.py — Prediksi Dropout & Keberhasilan Akademik Mahasiswa
K-Means Clustering + Hierarchical Agglomerative Clustering
Tugas Besar Penambangan Data | Universitas Telkom 2026
Pipeline: CRISP-DM (Bab II–IV Laporan)

Jalankan SEKALI: python train_model.py
"""
import pandas as pd, numpy as np, joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from scipy.optimize import linear_sum_assignment

print("="*60)
print("  PREDIKSI DROPOUT — K-Means & Hierarchical Clustering")
print("  Universitas Telkom 2026 | Penambangan Data")
print("="*60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1/7] Membaca dataset...")
df = pd.read_csv("dataset.csv")
print(f"      Shape awal: {df.shape}")
print(f"      Distribusi Target: {df['Target'].value_counts().to_dict()}")

# ── 2. HAPUS FITUR REDUNDAN (Bab III.1.3, Tabel III.3) ────────
print("[2/7] Menghapus 4 fitur redundan (Tabel III.3)...")
drop_cols = [
    "Curricular units 2nd sem (credited)",      # korelasi 0.9448 → redundan
    "International",                              # korelasi 0.9117 dengan Nacionality
    "Curricular units 1st sem (without evaluations)",   # variansi rendah
    "Curricular units 2nd sem (without evaluations)",   # variansi rendah
]
df = df.drop(columns=drop_cols)
print(f"      Shape setelah drop: {df.shape}  (35 -> 31 kolom)")

# ── 3. OUTLIER HANDLING — IQR Capping (Bab III.2.3) ──────────
print("[3/7] Outlier handling (IQR Capping)...")
for col in [
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
print("      ✓ IQR Capping pada grade sem 1 & sem 2")

# ── 4. FEATURE ENGINEERING (Bab III.3, Tabel III.4) ──────────
print("[4/7] Feature engineering (4 fitur baru)...")
df["Total_Approved_Units"] = (
    df["Curricular units 1st sem (approved)"] +
    df["Curricular units 2nd sem (approved)"]
)
df["Average_Grade"] = (
    df["Curricular units 1st sem (grade)"] +
    df["Curricular units 2nd sem (grade)"]
) / 2
df["Total_Enrolled_Units"] = (
    df["Curricular units 1st sem (enrolled)"] +
    df["Curricular units 2nd sem (enrolled)"]
)
df["Approval_Ratio"] = (
    df["Total_Approved_Units"] /
    df["Total_Enrolled_Units"].replace(0, np.nan)
).fillna(0)
print("      ✓ Total_Approved_Units, Average_Grade, Total_Enrolled_Units, Approval_Ratio")

# ── 5. ENCODING (Bab III.4.1) ─────────────────────────────────
print("[5/7] Encoding & pemilihan fitur...")
le = LabelEncoder()
df["Target_enc"] = le.fit_transform(df["Target"])
print(f"      Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding sesuai Bab III.4.1
categorical_cols = ['Marital status', 'Application mode', 'Course', 'Nacionality']
df_ohe = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Seluruh fitur setelah reduksi dan rekayasa digunakan
CLUSTER_FEATURES = [col for col in df_ohe.columns if col not in ["Target", "Target_enc"]]

X_raw = df_ohe[CLUSTER_FEATURES].copy()
y     = df_ohe["Target_enc"].values
print(f"      Total fitur clustering setelah OHE: {len(CLUSTER_FEATURES)}")

# ── 6. SCALING & DATA SPLIT (Bab III.4.2, IV.2) ──────────────
print("[6/7] Feature scaling & data splitting (80:20)...")

# Standard Scaler (digunakan untuk clustering berbasis jarak)
std_scaler = StandardScaler()
X_scaled   = std_scaler.fit_transform(X_raw)

# MinMax Scaler (juga disimpan sesuai laporan Bab III.4.2)
mm_scaler  = MinMaxScaler()
mm_scaler.fit(X_raw)

# Data splitting 80:20 (Bab IV.2)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

# 5-Fold Cross Validation setup (Bab IV.2 poin 3)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"      5-Fold CV: KFold(n_splits=5, shuffle=True, random_state=42)")

# ── 7. HYPERPARAMETER TUNING & MODEL FINAL (Bab IV.3) ────────
print("[7/7] Hyperparameter tuning + training model final...")

# === TUNING K-MEANS (Tabel IV.1) ===
print("      Grid search K-Means (Tabel IV.1)...")
param_grid_km = {
    "n_clusters": [2, 3, 4, 5, 6],
    "init":       ["k-means++", "random"],
    "max_iter":   [100, 300, 500],
    "n_init":     [10, 20],
}
best_sil_km, best_km_params = -1, None
for params in ParameterGrid(param_grid_km):
    m   = KMeans(**params, random_state=42)
    lbl = m.fit_predict(X_train)
    s   = silhouette_score(X_train, lbl)
    if s > best_sil_km:
        best_sil_km   = s
        best_km_params = params
print(f"      Best K-Means params: {best_km_params}")
print(f"      Best Silhouette   : {best_sil_km:.4f}")

# === TUNING HIERARCHICAL (Tabel IV.2) ===
print("      Grid search Hierarchical (Tabel IV.2)...")
param_grid_ag = {
    "n_clusters": [2, 3, 4, 5, 6],
    "linkage":    ["ward", "complete", "average"],
}
best_sil_ag, best_ag_params = -1, None
for params in ParameterGrid(param_grid_ag):
    m   = AgglomerativeClustering(**params)
    lbl = m.fit_predict(X_train)
    s   = silhouette_score(X_train, lbl)
    if s > best_sil_ag:
        best_sil_ag   = s
        best_ag_params = params
print(f"      Best Hierarchical params: {best_ag_params}")
print(f"      Best Silhouette        : {best_sil_ag:.4f}")

# === TRAIN MODEL FINAL ===
km_final = KMeans(**best_km_params, random_state=42)
km_tr    = km_final.fit_predict(X_train)
km_te    = km_final.predict(X_test)

ag_final = AgglomerativeClustering(**best_ag_params)
ag_tr    = ag_final.fit_predict(X_train)
ag_te    = ag_final.fit_predict(X_test)

n_km = best_km_params["n_clusters"]
n_ag = best_ag_params["n_clusters"]

# === METRIK CLUSTERING (Tabel IV.3) ===
sil_km = float(silhouette_score(X_train, km_tr))
dbi_km = float(davies_bouldin_score(X_train, km_tr))
ch_km  = float(calinski_harabasz_score(X_train, km_tr))

sil_ag = float(silhouette_score(X_train, ag_tr))
dbi_ag = float(davies_bouldin_score(X_train, ag_tr))
ch_ag  = float(calinski_harabasz_score(X_train, ag_tr))

print("\n" + "="*60)
print("  HASIL EVALUASI CLUSTERING (Tabel IV.3)")
print("="*60)
print(f"  K-Means ({n_km} cluster, {best_km_params['init']})")
print(f"    Silhouette Score : {sil_km:.4f}  (target ≥ 0.30 {'✅' if sil_km>=0.30 else '⚠️'})")
print(f"    Davies-Bouldin   : {dbi_km:.4f}  (target < 1.50 {'✅' if dbi_km<1.50 else '⚠️'})")
print(f"    Calinski-Harabasz: {ch_km:.1f}")
print(f"  Hierarchical ({n_ag} cluster, {best_ag_params['linkage']})")
print(f"    Silhouette Score : {sil_ag:.4f}  (target ≥ 0.30 {'✅' if sil_ag>=0.30 else '⚠️'})")
print(f"    Davies-Bouldin   : {dbi_ag:.4f}  (target < 1.50 {'✅' if dbi_ag<1.50 else '⚠️'})")
print(f"    Calinski-Harabasz: {ch_ag:.1f}")

# === CONFUSION MATRIX via Hungarian Algorithm (Bab IV.4) ===
def hungarian_map(cluster_lbl, true_lbl, n_c):
    n_cls = len(np.unique(true_lbl))
    sz    = max(n_c, n_cls)
    cost  = np.zeros((sz, sz))
    for i in range(n_c):
        for j in range(n_cls):
            cost[i, j] = np.sum((cluster_lbl == i) & (true_lbl == j))
    r, c = linear_sum_assignment(-cost)
    return {int(ri): int(ci) for ri, ci in zip(r, c) if int(ri) < n_c}

km_map       = hungarian_map(km_tr, y_train, n_km)
ag_map       = hungarian_map(ag_tr, y_train, n_ag)
km_te_pred   = np.array([km_map.get(int(l), 0) for l in km_te])
ag_te_pred   = np.array([ag_map.get(int(l), 0) for l in ag_te])

cm_km  = confusion_matrix(y_test, km_te_pred)
cm_ag  = confusion_matrix(y_test, ag_te_pred)
acc_km = float(accuracy_score(y_test, km_te_pred))
prec_km= float(precision_score(y_test, km_te_pred, average="weighted", zero_division=0))
rec_km = float(recall_score(y_test, km_te_pred, average="weighted", zero_division=0))
f1_km  = float(f1_score(y_test, km_te_pred, average="weighted", zero_division=0))
acc_ag = float(accuracy_score(y_test, ag_te_pred))
prec_ag= float(precision_score(y_test, ag_te_pred, average="weighted", zero_division=0))
rec_ag = float(recall_score(y_test, ag_te_pred, average="weighted", zero_division=0))
f1_ag  = float(f1_score(y_test, ag_te_pred, average="weighted", zero_division=0))

print("\n  Confusion Matrix K-Means (Tabel IV.4):")
print(f"  {cm_km}")
print(f"  Accuracy={acc_km:.4f} | Precision={prec_km:.4f} | Recall={rec_km:.4f} | F1={f1_km:.4f}")
print("\n  Confusion Matrix Hierarchical (Tabel IV.5):")
print(f"  {cm_ag}")
print(f"  Accuracy={acc_ag:.4f} | Precision={prec_ag:.4f} | Recall={rec_ag:.4f} | F1={f1_ag:.4f}")

# === PROFIL CLUSTER K-MEANS ===
print("\n  Profil Cluster K-Means (Distribusi Status):")
cluster_do_pct = {}
km_cluster_dist, ag_cluster_dist = {}, {}

for c in range(n_km):
    mask = km_tr == c
    n    = int(mask.sum())
    do   = int((y_train[mask] == 0).sum())
    en   = int((y_train[mask] == 1).sum())
    gr   = int((y_train[mask] == 2).sum())
    cluster_do_pct[c] = do / n
    km_cluster_dist[c] = {"n": n, "Dropout": do, "Enrolled": en, "Graduate": gr}
    print(f"  Cluster {c}: n={n} | DO={do/n*100:.1f}% EN={en/n*100:.1f}% GR={gr/n*100:.1f}%")

for c in range(n_ag):
    mask = ag_tr == c
    n    = int(mask.sum())
    ag_cluster_dist[c] = {
        "n": n,
        "Dropout":  int((y_train[mask] == 0).sum()),
        "Enrolled": int((y_train[mask] == 1).sum()),
        "Graduate": int((y_train[mask] == 2).sum()),
    }

# Nama cluster berdasarkan dominasi dropout
sorted_c = sorted(cluster_do_pct, key=cluster_do_pct.get, reverse=True)
CLUSTER_NAMES = {}
CLUSTER_NAMES[sorted_c[0]] = "Cluster Risiko Tinggi (Dropout)"
if n_km >= 3:
    CLUSTER_NAMES[sorted_c[1]] = "Cluster Risiko Sedang (Enrolled)"
    CLUSTER_NAMES[sorted_c[2]] = "Cluster Berhasil (Graduate)"
else:
    CLUSTER_NAMES[sorted_c[1]] = "Cluster Berhasil (Graduate)"

max_do = max(cluster_do_pct.values()) * 100
print(f"\n  Cluster dropout tertinggi: {max_do:.1f}%  "
      f"{'✅ Kriteria bisnis >70% terpenuhi' if max_do>70 else '⚠️ Belum >70%'}")

# === PCA 2D (Bab IV.3) ===
pca = PCA(n_components=2, random_state=42)
X_pca_train = pca.fit_transform(X_train)
var_exp = pca.explained_variance_ratio_
print(f"\n  PCA variance: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%")

# Profil rata-rata fitur per cluster (untuk visualisasi dashboard)
profil_fitur = [f for f in [
    "Approval_Ratio", "Average_Grade", "Total_Approved_Units",
    "Age at enrollment", "Tuition fees up to date", "Scholarship holder",
    "Debtor", "Unemployment rate", "GDP",
] if f in CLUSTER_FEATURES]
profil_idx = [CLUSTER_FEATURES.index(f) for f in profil_fitur]

km_profiles = {}
for c in range(n_km):
    mask = km_tr == c
    km_profiles[c] = {
        f: round(float(np.mean(X_train[mask, profil_idx[i]])), 4)
        for i, f in enumerate(profil_fitur)
    }

# === SIMPAN SEMUA ARTIFACTS ===
os.makedirs("model", exist_ok=True)
joblib.dump(km_final,       "model/kmeans.pkl")
joblib.dump(ag_final,       "model/agglo.pkl")
joblib.dump(std_scaler,     "model/scaler.pkl")        # primary scaler
joblib.dump(mm_scaler,      "model/scaler_mm.pkl")     # minmax scaler
joblib.dump(le,             "model/label_encoder.pkl")
joblib.dump(pca,            "model/pca.pkl")
joblib.dump(CLUSTER_FEATURES, "model/feature_columns.pkl")
joblib.dump(CLUSTER_NAMES,  "model/cluster_names.pkl")
joblib.dump({
    "best_km_params": best_km_params,
    "best_ag_params": best_ag_params,
    "km_map":         {str(k): v for k, v in km_map.items()},
    "ag_map":         {str(k): v for k, v in ag_map.items()},
    "class_names":    list(le.classes_),
    "kmeans": {
        "silhouette":       round(sil_km, 4),
        "dbi":              round(dbi_km, 4),
        "ch":               round(ch_km,  1),
        "accuracy":         round(acc_km,  4),
        "precision":        round(prec_km, 4),
        "recall":           round(rec_km,  4),
        "f1":               round(f1_km,   4),
        "confusion_matrix": cm_km.tolist(),
        "n_clusters":       n_km,
    },
    "agglo": {
        "silhouette":       round(sil_ag, 4),
        "dbi":              round(dbi_ag, 4),
        "ch":               round(ch_ag,  1),
        "accuracy":         round(acc_ag,  4),
        "precision":        round(prec_ag, 4),
        "recall":           round(rec_ag,  4),
        "f1":               round(f1_ag,   4),
        "confusion_matrix": cm_ag.tolist(),
        "n_clusters":       n_ag,
    },
    "km_cluster_dist": km_cluster_dist,
    "ag_cluster_dist": ag_cluster_dist,
    "km_profiles":     km_profiles,
    "pca_variance":    [float(v) for v in var_exp],
    "X_pca_train":     X_pca_train.tolist(),
    "km_labels_train": km_tr.tolist(),
    "ag_labels_train": ag_tr.tolist(),
    "y_train":         y_train.tolist(),
}, "model/eval_results.pkl")

print("\n" + "="*60)
print("✅ Semua file tersimpan di folder model/")
print("   Jalankan: streamlit run app.py")
print("="*60)
