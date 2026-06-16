# ============================================================
# PRAKTIKUM BIG DATA — KELOMPOK 7
# UPN VETERAN JAKARTA 2026
#
# Judul:
# Laporan Big Data Analytics Studi Kasus pada Job Descriptions Dataset
# Menggunakan TF-IDF dan Logistic Regression
#
# Fokus:
# ETL, EDA, Data Quality, Big Data 3V, Predictive Diagnosis,
# Generated Rules, dan Clustering Diagnosis
# ============================================================


# ============================================================
# CELL 1 — INSTALL & IMPORT LIBRARY
# ============================================================

import subprocess
subprocess.run(["pip", "install", "polars", "--quiet"], check=False)

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

plt.rcParams.update({
    "figure.dpi": 130,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

sns.set_palette("Set2")

OUTPUT_DIR = "BigData/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Polars : {pl.__version__}")
print(f"Pandas : {pd.__version__}")
print("✅ Semua library berhasil diimport.")


# ============================================================
# CELL 2 — AUTO FIND DATASET
# ============================================================

def find_csv_file(filename="job_descriptions.csv", root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


FILE_PATH = "job_descriptions.csv"

if not os.path.exists(FILE_PATH):
    found_path = find_csv_file("job_descriptions.csv", "/kaggle/input")
    if found_path is not None:
        FILE_PATH = found_path
    else:
        print("❌ File job_descriptions.csv tidak ditemukan. Daftar file CSV yang tersedia:")
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path) / (1024 ** 3)
                    print(f"- {full_path} ({size:.2f} GB)")
        raise FileNotFoundError("Sesuaikan FILE_PATH dengan lokasi dataset.")

size_gb = os.path.getsize(FILE_PATH) / (1024 ** 3)

print("=" * 80)
print("EXTRACT DATASET")
print("=" * 80)
print(f"File dataset : {FILE_PATH}")
print(f"Ukuran file  : {size_gb:.2f} GB")


# ============================================================
# CELL 3 — LOAD DATASET DENGAN POLARS
# ============================================================

start_time = time.time()

df_pl = pl.read_csv(
    FILE_PATH,
    infer_schema_length=10000,
    ignore_errors=True,
    null_values=["", "NA", "N/A", "null", "NULL", "None"]
)

elapsed = time.time() - start_time

print(f"Waktu baca dengan Polars : {elapsed:.2f} detik")
print(f"Jumlah baris             : {df_pl.height:,}")
print(f"Jumlah kolom             : {df_pl.width}")
print(f"Estimasi RAM Polars      : {df_pl.estimated_size('mb'):.2f} MB")

print("\nInformasi Kolom:")
print("-" * 80)
print(f"{'Kolom':<30} {'Tipe Data':<15} {'Non-Null':>15}")
print("-" * 80)

for col in df_pl.columns:
    dtype = str(df_pl[col].dtype)
    non_null = df_pl.height - df_pl[col].null_count()
    print(f"{col:<30} {dtype:<15} {non_null:>15,}")

print("-" * 80)
print("\nSample 3 baris:")
print(df_pl.head(3))


# ============================================================
# CELL 4 — HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_salary_series(series):
    extracted = series.astype(str).str.extract(r"\$(\d+)K-\$(\d+)K")

    min_k = pd.to_numeric(extracted[0], errors="coerce")
    max_k = pd.to_numeric(extracted[1], errors="coerce")
    mid_k = (min_k + max_k) / 2

    return min_k, max_k, mid_k


def parse_exp_mean_series(series):
    extracted = series.astype(str).str.extract(r"(\d+)\s+to\s+(\d+)")

    min_exp = pd.to_numeric(extracted[0], errors="coerce")
    max_exp = pd.to_numeric(extracted[1], errors="coerce")
    exp_mean = (min_exp + max_exp) / 2

    single_num = series.astype(str).str.extract(r"(\d+)")[0]
    single_num = pd.to_numeric(single_num, errors="coerce")

    exp_mean = exp_mean.fillna(single_num)

    return exp_mean


def title_overlap_ratio(job_title, job_description):
    title_words = set(clean_text(job_title).split())
    desc_words = set(clean_text(job_description).split())

    if len(title_words) == 0:
        return 0

    return len(title_words.intersection(desc_words)) / len(title_words)


print("✅ Helper functions berhasil dibuat.")


# ============================================================
# CELL 5 — DATA QUALITY SEBELUM TRANSFORMASI
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY SEBELUM TRANSFORMASI")
print("=" * 80)

total_rows = df_pl.height
total_cols = df_pl.width
total_cells = total_rows * total_cols

missing_dict = {col: df_pl[col].null_count() for col in df_pl.columns}
total_missing = sum(missing_dict.values())
completeness_before = (1 - total_missing / total_cells) * 100

if "Job Id" in df_pl.columns:
    unique_job_id = df_pl["Job Id"].n_unique()
    duplicate_rows = total_rows - unique_job_id
else:
    unique_job_id = df_pl.unique().height
    duplicate_rows = total_rows - unique_job_id

salary_valid = df_pl["Salary Range"].drop_nulls().map_elements(
    lambda x: bool(re.match(r"\$\d+K-\$\d+K", str(x))),
    return_dtype=pl.Boolean
).sum()

lat_valid = df_pl.filter(
    (pl.col("latitude") >= -90) & (pl.col("latitude") <= 90)
).height

lon_valid = df_pl.filter(
    (pl.col("longitude") >= -180) & (pl.col("longitude") <= 180)
).height

print(f"Total baris          : {total_rows:,}")
print(f"Total kolom          : {total_cols}")
print(f"Total cells          : {total_cells:,}")
print(f"Missing values       : {total_missing:,}")
print(f"Completeness         : {completeness_before:.4f}%")
print(f"Unique Job Id        : {unique_job_id:,}")
print(f"Duplicate rows       : {duplicate_rows:,}")
print(f"Salary format valid  : {salary_valid:,} ({salary_valid / total_rows * 100:.2f}%)")
print(f"Latitude valid       : {lat_valid:,} ({lat_valid / total_rows * 100:.2f}%)")
print(f"Longitude valid      : {lon_valid:,} ({lon_valid / total_rows * 100:.2f}%)")

print("\nMissing values per column:")
for col, miss in missing_dict.items():
    if miss > 0:
        print(f"{col:<30}: {miss:,} ({miss / total_rows * 100:.4f}%)")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Data Quality Sebelum Transformasi", fontsize=14, fontweight="bold")

axes[0].pie(
    [total_cells - total_missing, total_missing],
    labels=["Data Lengkap", "Missing Values"],
    autopct="%1.4f%%",
    startangle=90,
    explode=(0, 0.08)
)
axes[0].set_title(f"Completeness\n{completeness_before:.4f}%")

missing_nonzero = {k: v for k, v in missing_dict.items() if v > 0}

if missing_nonzero:
    axes[1].barh(list(missing_nonzero.keys()), list(missing_nonzero.values()))
    axes[1].set_title("Missing Values per Kolom")
    axes[1].set_xlabel("Jumlah Missing")

    for i, (col, value) in enumerate(missing_nonzero.items()):
        axes[1].text(value + 50, i, f"{value:,}", va="center", fontsize=9)
else:
    axes[1].text(0.5, 0.5, "Tidak ada missing values", ha="center", va="center")
    axes[1].axis("off")

axes[2].bar(
    ["Total Baris", "Baris Unik", "Duplikat"],
    [total_rows, unique_job_id, duplicate_rows]
)
axes[2].set_title("Uniqueness Check")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

for i, value in enumerate([total_rows, unique_job_id, duplicate_rows]):
    axes[2].text(i, value + 10000, f"{value:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_data_quality_before.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 6 — TRANSFORMASI DAN FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("ETL TRANSFORM: CLEANING DAN FEATURE ENGINEERING")
print("=" * 80)

df_pd = df_pl.to_pandas()
n_before = len(df_pd)

df_pd["Salary_Min_K"], df_pd["Salary_Max_K"], df_pd["Salary_Mid_K"] = parse_salary_series(df_pd["Salary Range"])
df_pd["exp_mean"] = parse_exp_mean_series(df_pd["Experience"])

df_pd["Job_Description_Clean"] = df_pd["Job Description"].apply(clean_text)
df_pd["skills_clean"] = df_pd["skills"].apply(clean_text)
df_pd["skill_word_count"] = df_pd["skills_clean"].apply(lambda x: len(x.split()))

company_size_rank = df_pd["Company Size"].rank(method="first")

df_pd["cs_qcode"] = pd.qcut(
    company_size_rank,
    q=5,
    labels=False,
    duplicates="drop"
)

cs_label_map = {
    0: "Very Small",
    1: "Small",
    2: "Medium",
    3: "Large",
    4: "Very Large"
}

df_pd["cs_qcat"] = df_pd["cs_qcode"].map(cs_label_map).astype("category")

df_clean = df_pd.dropna(subset=["Salary_Mid_K"]).copy()
df_clean["exp_mean"] = df_clean["exp_mean"].fillna(df_clean["exp_mean"].median())

le_work = LabelEncoder()
le_qual = LabelEncoder()
le_title = LabelEncoder()

df_clean["wt_enc"] = le_work.fit_transform(df_clean["Work Type"].astype(str))
df_clean["qual_enc"] = le_qual.fit_transform(df_clean["Qualifications"].astype(str))
df_clean["title_enc"] = le_title.fit_transform(df_clean["Job Title"].astype(str))

n_after = len(df_clean)
retention_rate = n_after / n_before * 100

engineered_cols = [
    "Salary_Min_K",
    "Salary_Max_K",
    "Salary_Mid_K",
    "exp_mean",
    "cs_qcat",
    "skill_word_count",
    "wt_enc",
    "qual_enc",
    "title_enc",
    "Job_Description_Clean"
]

missing_engineered = df_clean[engineered_cols].isna().sum()
total_engineered_cells = len(df_clean) * len(engineered_cols)
total_engineered_missing = missing_engineered.sum()
completeness_after = (1 - total_engineered_missing / total_engineered_cells) * 100

print(f"Baris sebelum transformasi : {n_before:,}")
print(f"Baris setelah transformasi : {n_after:,}")
print(f"Retention rate             : {retention_rate:.2f}%")

print("\nFitur hasil transformasi:")
for col in engineered_cols:
    print(f"{col:<25}: missing {df_clean[col].isna().sum():,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Data Quality Sebelum dan Sesudah Transformasi", fontsize=14, fontweight="bold")

axes[0].bar(["Sebelum", "Sesudah"], [completeness_before, completeness_after])
axes[0].set_title("Completeness")
axes[0].set_ylabel("Completeness (%)")
axes[0].set_ylim(95, 100.5)

for i, value in enumerate([completeness_before, completeness_after]):
    axes[0].text(i, value + 0.05, f"{value:.4f}%", ha="center", fontsize=9)

axes[1].bar(["Sebelum", "Sesudah"], [n_before, n_after])
axes[1].set_title("Data Retention")
axes[1].set_ylabel("Jumlah Baris")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

for i, value in enumerate([n_before, n_after]):
    axes[1].text(i, value + 10000, f"{value:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_data_quality_before_after.png", bbox_inches="tight", dpi=150)
plt.show()

# ============================================================
# CELL 7 — EDA WORK TYPE
# ============================================================

print("\n" + "=" * 80)
print("EDA: DISTRIBUSI WORK TYPE")
print("=" * 80)

work_type = df_clean["Work Type"].value_counts().reset_index()
work_type.columns = ["Work Type", "count"]
work_type["pct"] = work_type["count"] / work_type["count"].sum() * 100
work_type = work_type.sort_values("count", ascending=False)

print(work_type)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(work_type["Work Type"], work_type["count"])
ax.set_title("Distribusi Work Type")
ax.set_xlabel("Work Type")
ax.set_ylabel("Jumlah Lowongan")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

for i, row in work_type.reset_index(drop=True).iterrows():
    ax.text(i, row["count"] + 2000, f"{row['pct']:.2f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_work_type_distribution.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 8 — EDA TEMPORAL
# ============================================================

print("📊 TREN TEMPORAL JOB POSTING")

date_col = "Job Posting Date"  

sample_dates = df_clean[date_col].dropna().head(20).tolist()

date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
                '%Y/%m/%d', '%m-%d-%Y', '%d %b %Y', '%b %d, %Y']

detected_format = None
for fmt in date_formats:
    try:
        test = pd.to_datetime(pd.Series(sample_dates[:10]), format=fmt, errors='coerce')
        if test.notna().sum() >= 7:  # minimal 7 dari 10 berhasil dikonversi
            detected_format = fmt
            print(f'✅ Format tanggal terdeteksi: {fmt}')
            break
    except Exception:
        continue

if detected_format is None:
    print("⚠️ Format tidak terdeteksi, mencoba konversi otomatis...")
    df_clean['posting_date'] = pd.to_datetime(df_clean[date_col], errors='coerce')
else:
    df_clean['posting_date'] = pd.to_datetime(df_clean[date_col], 
                                               format=detected_format, 
                                               errors='coerce')

df_temporal = df_clean.dropna(subset=['posting_date']).copy()
print(f'✅ Baris dengan tanggal valid: {len(df_temporal):,}')

monthly = (
    df_temporal
    .groupby(df_temporal['posting_date'].dt.to_period('M'))
    .size()
    .reset_index(name='count')
)

monthly.columns = ['period', 'count']  
monthly['period'] = monthly['period'].astype(str)  
monthly = monthly.sort_values('period').reset_index(drop=True)

print(f'✅ Bulan terdeteksi: {len(monthly)}')

if len(monthly) > 0:
    avg_monthly = monthly['count'].mean()
    idx_max = monthly['count'].idxmax()
    idx_min = monthly['count'].idxmin()
    
    print(f'📊 Rata-rata per bulan : {avg_monthly:,.0f}')
    print(f'📈 Tertinggi           : {monthly.loc[idx_max, "count"]:,} ({monthly.loc[idx_max, "period"]})')
    print(f'📉 Terendah            : {monthly.loc[idx_min, "count"]:,} ({monthly.loc[idx_min, "period"]})')

if len(monthly) > 0:
    fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.fill_between(range(len(monthly)), monthly['count'], alpha=0.25, color='#3498db')
    
    ax.plot(range(len(monthly)), monthly['count'],
            marker='o', color='#2980b9', linewidth=2, markersize=5)
    
    ax.axhline(avg_monthly, color='orange', linestyle='--', linewidth=1.5,
               label=f'Rata-rata: {avg_monthly:,.0f}')
    
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['period'], rotation=45, ha='right', fontsize=8)
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    
    ax.set_title('Tren Jumlah Job Posting per Bulan (Sep 2021 – Sep 2023)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Bulan-Tahun')
    ax.set_ylabel('Jumlah Job Posting')
    ax.legend()
    
    ax.annotate(f"Puncak\n{monthly.loc[idx_max,'count']:,}",
                xy=(idx_max, monthly.loc[idx_max,'count']),
                xytext=(idx_max + 1, monthly.loc[idx_max,'count'] * 1.03),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=8, color='green')
    
    ax.annotate(f"Terendah\n{monthly.loc[idx_min,'count']:,}",
                xy=(idx_min, monthly.loc[idx_min,'count']),
                xytext=(idx_min + 1, monthly.loc[idx_min,'count'] * 0.92),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_temporal_trend.png', bbox_inches='tight', dpi=150)
    plt.show()
    print('✅ Grafik tren temporal tersimpan.')
else:
    print('❌ Tidak ada data tanggal yang valid untuk divisualisasikan.')

# ============================================================
# CELL 9 — EDA TOP JOB TITLE DAN COMPANY SIZE QUANTILE
# ============================================================

print("\n" + "=" * 80)
print("EDA: TOP JOB TITLE DAN COMPANY SIZE")
print("=" * 80)

top_titles = df_clean["Job Title"].value_counts().head(10).reset_index()
top_titles.columns = ["Job Title", "count"]
top_titles["pct"] = top_titles["count"] / len(df_clean) * 100

company_size = df_clean["cs_qcat"].value_counts().reset_index()
company_size.columns = ["Company Size Category", "count"]
company_size["pct"] = company_size["count"] / len(df_clean) * 100

cs_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]
company_size["Company Size Category"] = pd.Categorical(
    company_size["Company Size Category"],
    categories=cs_order,
    ordered=True
)
company_size = company_size.sort_values("Company Size Category")

print("\nTop 10 Job Titles:")
print(top_titles)

print("\nCompany Size Quantile:")
print(company_size)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Top Job Titles dan Company Size Quantile", fontsize=14, fontweight="bold")

axes[0].barh(top_titles["Job Title"][::-1], top_titles["count"][::-1])
axes[0].set_title("Top 10 Job Titles")
axes[0].set_xlabel("Jumlah Lowongan")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

for i, row in enumerate(top_titles[::-1].itertuples(index=False)):
    axes[0].text(row.count + 500, i, f"{row.count:,}", va="center", fontsize=8)

axes[1].bar(company_size["Company Size Category"].astype(str), company_size["count"])
axes[1].set_title("Distribusi Company Size Berbasis Quantile")
axes[1].set_ylabel("Jumlah Lowongan")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

for i, row in enumerate(company_size.itertuples(index=False)):
    axes[1].text(i, row.count + 2000, f"{row.pct:.1f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_top_titles_company_size.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 10 — EDA SALARY
# ============================================================

print("\n" + "=" * 80)
print("EDA: SALARY")
print("=" * 80)

salary_stats = df_clean["Salary_Mid_K"].describe()
print("\nStatistik Salary_Mid_K:")
print(salary_stats)

salary_by_work = (
    df_clean
    .groupby("Work Type")["Salary_Mid_K"]
    .agg(["mean", "median", "std"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

print("\nSalary per Work Type:")
print(salary_by_work)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Analisis Salary", fontsize=14, fontweight="bold")

axes[0].hist(df_clean["Salary_Mid_K"], bins=60, edgecolor="white")
axes[0].axvline(df_clean["Salary_Mid_K"].mean(), linestyle="--", label=f"Mean: {df_clean['Salary_Mid_K'].mean():.1f}K")
axes[0].axvline(df_clean["Salary_Mid_K"].median(), linestyle="-.", label=f"Median: {df_clean['Salary_Mid_K'].median():.1f}K")
axes[0].set_title("Distribusi Salary Mid")
axes[0].set_xlabel("Salary Mid ($K)")
axes[0].set_ylabel("Frekuensi")
axes[0].legend()

work_order = salary_by_work.sort_values("median")["Work Type"].tolist()
box_data = [df_clean[df_clean["Work Type"] == wt]["Salary_Mid_K"] for wt in work_order]

axes[1].boxplot(box_data, labels=work_order, patch_artist=True)
axes[1].set_title("Salary per Work Type")
axes[1].set_xlabel("Work Type")
axes[1].set_ylabel("Salary Mid ($K)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_salary_distribution.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 11 — TOP 10 COUNTRIES
# ============================================================

print("\n" + "=" * 80)
print("EDA: TOP 10 COUNTRIES")
print("=" * 80)

top_countries = df_clean["Country"].value_counts().head(10).reset_index()
top_countries.columns = ["Country", "count"]
top_countries["pct"] = top_countries["count"] / len(df_clean) * 100

print(top_countries)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_countries["Country"][::-1], top_countries["count"][::-1])
ax.set_title("Top 10 Countries Berdasarkan Jumlah Job Posting")
ax.set_xlabel("Jumlah Lowongan")
ax.set_ylabel("Country")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

for i, row in enumerate(top_countries[::-1].itertuples(index=False)):
    ax.text(row.count + 50, i, f"{row.count:,}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_top10_countries.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 12 — STRATIFIED SAMPLING UNTUK MODELING
# ============================================================

print("\n" + "=" * 80)
print("STRATIFIED SAMPLING UNTUK MODELING")
print("=" * 80)

SAMPLE_FRAC = 0.10

df_sample = (
    df_clean
    .groupby("Work Type", group_keys=False)
    .apply(lambda x: x.sample(frac=SAMPLE_FRAC, random_state=42))
    .reset_index(drop=True)
)

print(f"Full clean dataset : {len(df_clean):,}")
print(f"Sample 10%         : {len(df_sample):,}")

print("\nDistribusi Work Type dalam sample:")
print(df_sample["Work Type"].value_counts())


# ============================================================
# CELL 13 — PREDICTIVE DIAGNOSIS: JOB TITLE CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("PREDICTIVE DIAGNOSIS: JOB TITLE CLASSIFICATION")
print("=" * 80)

top10_titles = df_clean["Job Title"].value_counts().head(10).index.tolist()

df_clf = df_sample[df_sample["Job Title"].isin(top10_titles)].copy()
df_clf = df_clf.dropna(subset=["Job_Description_Clean", "Job Title"])
df_clf = df_clf.reset_index(drop=True)

print(f"Data klasifikasi Top 10 Job Title: {len(df_clf):,}")
print("\nDistribusi label:")
print(df_clf["Job Title"].value_counts())

df_clf["title_desc_overlap"] = df_clf.apply(
    lambda row: title_overlap_ratio(row["Job Title"], row["Job_Description_Clean"]),
    axis=1
)

overlap_mean = df_clf["title_desc_overlap"].mean()
overlap_nonzero = (df_clf["title_desc_overlap"] > 0).mean() * 100

print(f"\nRata-rata overlap kata Job Title dengan Job Description : {overlap_mean:.4f}")
print(f"Persentase data overlap > 0                             : {overlap_nonzero:.2f}%")

tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X = tfidf.fit_transform(df_clf["Job_Description_Clean"])
y = df_clf["Job Title"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTrain size      : {X_train.shape[0]:,}")
print(f"Test size       : {X_test.shape[0]:,}")
print(f"TF-IDF features : {X.shape[1]:,}")

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}

classification_results = {}
classification_rows = []

for model_name, model in models.items():
    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    classification_results[model_name] = {
        "model": model,
        "prediction": y_pred,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1
    }

    classification_rows.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1
    })

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

classification_table = pd.DataFrame(classification_rows)
classification_table.to_csv(f"{OUTPUT_DIR}/classification_results.csv", index=False)

print("\nRingkasan model klasifikasi:")
print(classification_table)

main_model_name = "Logistic Regression"
main_model = classification_results[main_model_name]["model"]
main_pred = classification_results[main_model_name]["prediction"]

print(f"\nModel utama untuk laporan: {main_model_name}")

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle("Predictive Diagnosis: Klasifikasi Job Title", fontsize=14, fontweight="bold")

x = np.arange(len(classification_table))
width = 0.22

axes[0].bar(x - width, classification_table["Accuracy"], width, label="Accuracy")
axes[0].bar(x, classification_table["Precision"], width, label="Precision")
axes[0].bar(x + width, classification_table["F1-Score"], width, label="F1-Score")
axes[0].set_xticks(x)
axes[0].set_xticklabels(classification_table["Model"], rotation=20, ha="right")
axes[0].set_ylim(0, 1.10)
axes[0].set_title("Perbandingan Model")
axes[0].legend()

cm_labels = top10_titles
cm = confusion_matrix(y_test, main_pred, labels=cm_labels)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    ax=axes[1],
    cbar=False
)
axes[1].set_title(f"Confusion Matrix\n{main_model_name}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_xticklabels(cm_labels, rotation=45, ha="right", fontsize=7)
axes[1].set_yticklabels(cm_labels, fontsize=7)

axes[2].hist(df_clf["title_desc_overlap"], bins=20, edgecolor="white")
axes[2].set_title("Semantic Leakage Check\nOverlap Job Title vs Job Description")
axes[2].set_xlabel("Rasio Overlap Kata")
axes[2].set_ylabel("Frekuensi")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_classification_diagnosis.png", bbox_inches="tight", dpi=150)
plt.show()


# ============================================================
# CELL 14 — GENERATED INTERPRETATIVE RULES
# ============================================================

print("\n" + "=" * 80)
print("GENERATED INTERPRETATIVE RULES")
print("=" * 80)

feature_names = np.array(tfidf.get_feature_names_out())
classes = main_model.classes_

rules_rows = []
TOP_N_TERMS = 8

for class_idx, class_name in enumerate(classes):
    coefs = main_model.coef_[class_idx]

    top_indices = np.argsort(coefs)[-TOP_N_TERMS:][::-1]
    top_terms = feature_names[top_indices]
    top_scores = coefs[top_indices]

    rule_text = (
        f"Jika Job Description mengandung istilah seperti "
        f"{', '.join(top_terms[:5])}, maka model cenderung "
        f"mengklasifikasikan data sebagai {class_name}."
    )

    rules_rows.append({
        "Job Title": class_name,
        "Top Terms": ", ".join(top_terms),
        "Generated Rule": rule_text
    })

rules_df = pd.DataFrame(rules_rows)
rules_df.to_csv(f"{OUTPUT_DIR}/generated_interpretative_rules.csv", index=False)

print(rules_df)


# ============================================================
# CELL 15 — K-MEANS CLUSTERING DIAGNOSIS
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSIS: K-MEANS CLUSTERING")
print("=" * 80)

KM_SAMPLE_N = min(30000, len(df_sample))
df_km = df_sample.sample(n=KM_SAMPLE_N, random_state=42).copy()

km_features = [
    "Salary_Mid_K",
    "exp_mean",
    "skill_word_count",
    "Work Type",
    "cs_qcat"
]

df_km = df_km[km_features].dropna().copy()

df_km_encoded = pd.get_dummies(
    df_km,
    columns=["Work Type", "cs_qcat"],
    drop_first=False
)

scaler = MinMaxScaler()
X_km = scaler.fit_transform(df_km_encoded)

k_values = range(2, 9)

wcss = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_km)

    wcss.append(kmeans.inertia_)

    sil = silhouette_score(
        X_km,
        labels,
        sample_size=min(10000, len(X_km)),
        random_state=42
    )

    silhouette_scores.append(sil)

    print(f"k={k} | WCSS={kmeans.inertia_:.2f} | Silhouette={sil:.4f}")

best_k = list(k_values)[int(np.argmax(silhouette_scores))]

print(f"\nK optimal berdasarkan Silhouette Score: {best_k}")

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df_km["cluster"] = kmeans_final.fit_predict(X_km)

cluster_profile = (
    df_km
    .groupby("cluster")
    .agg(
        jumlah_data=("cluster", "count"),
        rata_salary=("Salary_Mid_K", "mean"),
        median_salary=("Salary_Mid_K", "median"),
        rata_exp=("exp_mean", "mean"),
        rata_skill_words=("skill_word_count", "mean")
    )
    .reset_index()
    .sort_values("rata_salary")
)

cluster_profile.to_csv(f"{OUTPUT_DIR}/cluster_profile.csv", index=False)

print("\nProfil Cluster:")
print(cluster_profile)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Means Clustering Diagnosis pada Job Posting", fontsize=14, fontweight="bold")

axes[0].plot(list(k_values), wcss, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Jumlah Cluster (k)")
axes[0].set_ylabel("WCSS")

axes[1].plot(list(k_values), silhouette_scores, marker="o")
axes[1].axvline(best_k, linestyle="--", label=f"Best k = {best_k}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Jumlah Cluster (k)")
axes[1].set_ylabel("Silhouette")
axes[1].legend()

axes[2].bar(cluster_profile["cluster"].astype(str), cluster_profile["rata_salary"])
axes[2].set_title("Rata-rata Salary per Cluster")
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("Avg Salary_Mid_K")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_kmeans_clustering_diagnosis.png", bbox_inches="tight", dpi=150)
plt.show()

print("\n✅ SELESAI.")