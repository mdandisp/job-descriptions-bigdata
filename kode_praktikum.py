# ============================================================
# PRAKTIKUM BIG DATA — KELOMPOK 7
# UPN VETERAN JAKARTA 2026
# Judul : Analisis Big Data Deskripsi Pekerjaan Global:
#         Implementasi ETL, Klasifikasi, Prediksi Salary,
#         dan Klasterisasi pada Dataset Pasar Kerja 2021–2023
# Dataset: job_descriptions.csv (Kaggle - ravindrasinghrana)
# ============================================================
# Anggota:
#   Akmal Taufiqurrahman          (2310511018)
#   Wisnu Chandra Mukti Wibowo    (2310511026)
#   Muhammad Dandi Setiawan Putra (2310511035)
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1 — INSTALL & IMPORT LIBRARY                      ║
# ╚══════════════════════════════════════════════════════════╝
import subprocess
subprocess.run(['pip', 'install', 'polars', 'pyarrow', '--quiet'], check=False)

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os, re, time

os.makedirs("output", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
})
sns.set_palette('Set2')

print(f'Polars  : {pl.__version__}')
print(f'Pandas  : {pd.__version__}')
print('✅ Semua library berhasil diimport.')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — LOAD DATASET (EXTRACT — tahap ETL)            ║
# ╚══════════════════════════════════════════════════════════╝
FILE_PATH = 'job_descriptions.csv'

if not os.path.exists(FILE_PATH):
    kaggle_path = '/kaggle/input/datasets/ravindrasinghrana/job-description-dataset/job_descriptions.csv'
    if os.path.exists(kaggle_path):
        FILE_PATH = kaggle_path
    else:
        print('❌ Dataset tidak ditemukan di folder saat ini maupun di path Kaggle.')
        raise FileNotFoundError('Silakan letakkan file job_descriptions.csv di folder yang sama dengan notebook ini.')

size_gb = os.path.getsize(FILE_PATH) / (1024**3)
print(f'✅ File ditemukan: {size_gb:.2f} GB')

print('⏳ Membaca dataset dengan Polars (Big Data tool)...')
t0 = time.time()
df_pl = pl.read_csv(
    FILE_PATH,
    infer_schema_length=10000,
    ignore_errors=True,
    null_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'],
)
elapsed = time.time() - t0

print(f'✅ Selesai dalam {elapsed:.1f} detik')
print(f'   Jumlah baris : {df_pl.height:>12,}')
print(f'   Jumlah kolom : {df_pl.width:>12}')
print(f'   Estimasi RAM : {df_pl.estimated_size("mb"):>10.1f} MB')
print(f'   Ukuran file  : {size_gb:.2f} GB')

# ── Tampilkan info kolom ──────────────────────────────────────
print(f'\n{"─"*65}')
print(f'{"Nama Kolom":<30} {"Tipe Data":<15} {"Non-Null":>10}')
print(f'{"─"*65}')
for col in df_pl.columns:
    dtype  = str(df_pl[col].dtype)
    nonnull = df_pl.height - df_pl[col].null_count()
    print(f'{col:<30} {dtype:<15} {nonnull:>10,}')
print(f'{"─"*65}')

print(f'\nSample data (3 baris pertama):')
print(df_pl.head(3))

# ── Cek format tanggal ───────────────────────────────────────
date_col = 'Job Posting Date'
print(f'\nSample nilai "{date_col}" (5 pertama):')
print(df_pl[date_col].head(5).to_list())


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — HELPER FUNCTIONS (digunakan di sel berikutnya) ║
# ╚══════════════════════════════════════════════════════════╝

# ── Parse salary "$xxK-$xxK" → (min, max) ────────────────────
def parse_salary(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, None
    m = re.search(r'\$(\d+)K-\$(\d+)K', str(s))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# ── Parse pengalaman "X to Y Years" → rata-rata ──────────────
def parse_exp_mean(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r'(\d+)\s+to\s+(\d+)', str(s))
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m2 = re.search(r'(\d+)', str(s))
    if m2:
        return float(m2.group(1))
    return np.nan

# ── Kategorisasi Company Size (numerik → label) ───────────────
def categorize_company_size(val):
    if pd.isna(val):
        return 'Unknown'
    try:
        v = int(float(val))
    except (ValueError, TypeError):
        return 'Unknown'
    if v <= 10:
        return 'Very Small (1-10)'
    elif v <= 50:
        return 'Small (11-50)'
    elif v <= 200:
        return 'Medium (51-200)'
    elif v <= 500:
        return 'Large (201-500)'
    else:
        return 'Very Large (>500)'

# ── OrdinalEncoder untuk Company Size (didefinisikan sekali) ──
SIZE_CATS     = ['Very Small (1-10)', 'Small (11-50)', 'Medium (51-200)',
                 'Large (201-500)', 'Very Large (>500)']
SIZE_ORDER    = [SIZE_CATS]
# FIX: unknown_value harus di luar rentang 0-4, gunakan -1
oe_size = OrdinalEncoder(
    categories=SIZE_ORDER,
    handle_unknown='use_encoded_value',
    unknown_value=-1          # ← PERBAIKAN: -1 aman (tidak bentrok dengan 0-4)
)

print('✅ Helper functions & OrdinalEncoder berhasil didefinisikan.')
print(f'   OrdinalEncoder categories: {SIZE_CATS}')
print(f'   Encoded values: 0–{len(SIZE_CATS)-1}  |  unknown → -1')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — DATA QUALITY: COMPLETENESS, UNIQUENESS,       ║
# ║           VERACITY — SEBELUM TRANSFORMASI                ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  DATA QUALITY (SEBELUM TRANSFORMASI) — 3Vs Proof')
print('='*65)

# ── Completeness ─────────────────────────────────────────────
total_cells   = df_pl.height * df_pl.width
missing_dict  = {col: df_pl[col].null_count() for col in df_pl.columns}
total_missing = sum(missing_dict.values())
completeness  = (1 - total_missing / total_cells) * 100

print(f'\n📊 COMPLETENESS (Sebelum)')
print(f'   Total nilai data  : {total_cells:>15,}  (= {df_pl.height:,} × {df_pl.width})')
print(f'   Missing values    : {total_missing:>15,}')
print(f'   Kelengkapan data  : {completeness:>14.4f}%')
print(f'\n   Missing per kolom:')
for col, cnt in missing_dict.items():
    if cnt > 0:
        print(f'   {col:<28}: {cnt:>8,}  ({cnt/df_pl.height*100:.4f}%)')

# ── Uniqueness ───────────────────────────────────────────────
if 'Job Id' in df_pl.columns:
    n_unique = df_pl['Job Id'].n_unique()
    n_dup    = df_pl.height - n_unique
else:
    n_unique = df_pl.unique().height
    n_dup    = df_pl.height - n_unique

print(f'\n📊 UNIQUENESS')
print(f'   Total baris       : {df_pl.height:>12,}')
print(f'   Baris unik (Job Id): {n_unique:>11,}')
print(f'   Duplikat          : {n_dup:>12,}')
print(f'   Tingkat keunikan  : {n_unique/df_pl.height*100:.4f}%')

# ── Veracity ─────────────────────────────────────────────────
print(f'\n📊 VERACITY (Konsistensi Format)')
salary_valid = df_pl['Salary Range'].drop_nulls().map_elements(
    lambda x: bool(re.match(r'\$\d+K-\$\d+K', str(x))), return_dtype=pl.Boolean
).sum()
print(f'   Salary Range format valid : {salary_valid:,} / {df_pl.height:,}'
      f'  ({salary_valid/df_pl.height*100:.2f}%)')

if 'latitude' in df_pl.columns and 'longitude' in df_pl.columns:
    lat_valid = df_pl.filter(
        (pl.col('latitude') >= -90) & (pl.col('latitude') <= 90)
    ).height
    lon_valid = df_pl.filter(
        (pl.col('longitude') >= -180) & (pl.col('longitude') <= 180)
    ).height
    print(f'   Koordinat latitude valid  : {lat_valid:,} ({lat_valid/df_pl.height*100:.2f}%)')
    print(f'   Koordinat longitude valid : {lon_valid:,} ({lon_valid/df_pl.height*100:.2f}%)')

# ── Volume proof ─────────────────────────────────────────────
print(f'\n📊 VOLUME (3V Proof)')
print(f'   Ukuran file       : {size_gb:.2f} GB  (> batas Excel 1.04M baris)')
print(f'   Total baris       : {df_pl.height:,}')
print(f'   Total kolom       : {df_pl.width}')
print(f'   Total cells       : {total_cells:,}')
print(f'   Tool yang dipakai : Polars (Big Data in-memory columnar)')

# ── Visualisasi Data Quality SEBELUM ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
fig.suptitle('Data Quality — SEBELUM Transformasi\n(Completeness | Uniqueness | Veracity)',
             fontsize=13, fontweight='bold')

# Pie completeness
ax = axes[0]
sizes_pie  = [total_cells - total_missing, total_missing]
colors_pie = ['#2ecc71', '#e74c3c']
ax.pie(sizes_pie, labels=['Data Lengkap', 'Missing Values'],
       colors=colors_pie, autopct='%1.4f%%',
       startangle=90, explode=(0, 0.08), textprops={'fontsize': 9})
ax.set_title(f'Completeness\n({completeness:.4f}% lengkap)')

# Missing per kolom
ax2 = axes[1]
cols_miss = {k: v for k, v in missing_dict.items() if v > 0}
if cols_miss:
    ax2.barh(list(cols_miss.keys()), list(cols_miss.values()), color='#e74c3c')
    for i, (col, val) in enumerate(cols_miss.items()):
        ax2.text(val + 50, i, f'{val:,}', va='center', fontsize=9)
    ax2.set_xlabel('Jumlah Missing')
    ax2.set_title('Missing Values per Kolom')
else:
    ax2.text(0.5, 0.5, '✅ Tidak ada\nmissing values',
             ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    ax2.axis('off')
    ax2.set_title('Missing Values per Kolom')

# Uniqueness bar
ax3 = axes[2]
cats_u  = ['Total Baris', 'Baris Unik', 'Duplikat']
vals_u  = [df_pl.height, n_unique, max(n_dup, 0)]
colrs_u = ['#3498db', '#2ecc71', '#e74c3c']
bars3   = ax3.bar(cats_u, vals_u, color=colrs_u, width=0.5)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax3.set_title('Uniqueness Check')
for bar, val in zip(bars3, vals_u):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{val:,}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('output/quality_before.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Grafik Data Quality (Sebelum) tersimpan.')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — TRANSFORM: CLEANING & FEATURE ENGINEERING     ║
# ║           (tahap T dari ETL)                             ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  ETL — TRANSFORM: Cleaning & Feature Engineering')
print('='*65)

# Convert ke Pandas untuk transformasi
df_pd_raw = df_pl.to_pandas()
n_raw     = len(df_pd_raw)
print(f'   Baris awal: {n_raw:,}')

# 1. Parsing fitur numerik dari teks
df_pd_raw['Salary_Min_K'] = df_pd_raw['Salary Range'].apply(lambda x: parse_salary(x)[0])
df_pd_raw['Salary_Max_K'] = df_pd_raw['Salary Range'].apply(lambda x: parse_salary(x)[1])
df_pd_raw['Salary_Mid_K'] = (df_pd_raw['Salary_Min_K'] + df_pd_raw['Salary_Max_K']) / 2
df_pd_raw['exp_mean']     = df_pd_raw['Experience'].apply(parse_exp_mean)
df_pd_raw['cs_cat']       = df_pd_raw['Company Size'].apply(categorize_company_size)

# 2. Drop baris yang salary tidak terparsing (kualitas data kritis)
n_before_drop = len(df_pd_raw)
df_clean = df_pd_raw.dropna(subset=['Salary_Mid_K']).copy()
n_after_drop = len(df_clean)
print(f'   Baris setelah drop salary null: {n_after_drop:,}'
      f'  (dibuang: {n_before_drop - n_after_drop:,})')

# 3. Isi missing exp_mean dengan median
exp_median = df_clean['exp_mean'].median()
df_clean['exp_mean'].fillna(exp_median, inplace=True)

# 4. OrdinalEncode Company Size
oe_size.fit(df_clean[['cs_cat']])
df_clean['size_enc'] = oe_size.transform(df_clean[['cs_cat']])

# 5. Label encode Work Type & Qualifications
le_wt   = LabelEncoder()
le_qual = LabelEncoder()
le_title_global = LabelEncoder()
df_clean['wt_enc']    = le_wt.fit_transform(df_clean['Work Type'].fillna('Unknown').astype(str))
df_clean['qual_enc']  = le_qual.fit_transform(df_clean['Qualifications'].fillna('Unknown').astype(str))
df_clean['title_enc'] = le_title_global.fit_transform(df_clean['Job Title'].fillna('Unknown').astype(str))

print(f'\n   Feature Engineering yang dihasilkan:')
print(f'   {"Fitur Baru":<20} {"Sumber":<25} {"Metode"}')
print(f'   {"─"*65}')
print(f'   {"Salary_Min_K":<20} {"Salary Range":<25} {"Regex Parsing $xxK-$xxK"}')
print(f'   {"Salary_Max_K":<20} {"Salary Range":<25} {"Regex Parsing $xxK-$xxK"}')
print(f'   {"Salary_Mid_K":<20} {"Salary Range":<25} {"(Min+Max)/2"}')
print(f'   {"exp_mean":<20} {"Experience":<25} {"Regex avg X to Y Years"}')
print(f'   {"cs_cat":<20} {"Company Size":<25} {"Bucketing 5 kategori"}')
print(f'   {"size_enc":<20} {"cs_cat":<25} {"OrdinalEncoder (0-4)"}')
print(f'   {"wt_enc":<20} {"Work Type":<25} {"LabelEncoder"}')
print(f'   {"qual_enc":<20} {"Qualifications":<25} {"LabelEncoder"}')
print(f'   {"title_enc":<20} {"Job Title":<25} {"LabelEncoder"}')

# DATA QUALITY SETELAH TRANSFORMASI
print(f'\n📊 DATA QUALITY SETELAH TRANSFORMASI')
new_engineered_cols = ['Salary_Min_K','Salary_Max_K','Salary_Mid_K','exp_mean',
                       'cs_cat','size_enc','wt_enc','qual_enc','title_enc']
missing_after = {col: df_clean[col].isna().sum() for col in new_engineered_cols}
print(f'   {"Kolom Baru":<20} {"Missing":>10}')
for col, cnt in missing_after.items():
    print(f'   {col:<20} {cnt:>10,}')
print(f'\n   Total baris setelah cleaning: {len(df_clean):,}')
print(f'   Data retention rate         : {len(df_clean)/n_raw*100:.2f}%')

# Visualisasi Before-After Quality
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Perbandingan Data Quality: Sebelum vs Sesudah Transformasi',
             fontsize=13, fontweight='bold')

cats_ba   = ['Sebelum\nTransformasi', 'Sesudah\nTransformasi']
total_cells_after = len(df_clean) * len(df_clean.columns)
missing_after_val = df_clean[new_engineered_cols].isna().sum().sum()
completeness_after = (1 - missing_after_val / total_cells_after) * 100

bars_comp = axes[0].bar(cats_ba,
                        [completeness, min(completeness_after, 100)],
                        color=['#f39c12', '#2ecc71'], width=0.4)
axes[0].set_ylim(95, 100.5)
axes[0].set_ylabel('Completeness (%)')
axes[0].set_title('Completeness Data')
for bar, val in zip(bars_comp, [completeness, completeness_after]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05, f'{val:.4f}%', ha='center', fontsize=10, fontweight='bold')

bars_rows = axes[1].bar(cats_ba, [n_raw, len(df_clean)],
                        color=['#3498db', '#2ecc71'], width=0.4)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
axes[1].set_ylabel('Jumlah Baris')
axes[1].set_title('Jumlah Baris Data')
for bar, val in zip(bars_rows, [n_raw, len(df_clean)]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2000, f'{val:,}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('output/quality_before_after.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Grafik Before-After Quality tersimpan.')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — DESCRIPTIVE: WORK TYPE DISTRIBUTION           ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  EDA — DESCRIPTIVE ANALYSIS')
print('='*65)
print('\n📊 Distribusi Work Type')

wt = (
    df_pl.group_by('Work Type')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
    .to_pandas()
)
wt['pct'] = wt['count'] / wt['count'].sum() * 100
print(wt.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Distribusi Jenis Pekerjaan (Work Type)', fontsize=13, fontweight='bold')

palette_wt = sns.color_palette('Set2', len(wt))
axes[0].bar(wt['Work Type'], wt['count'], color=palette_wt)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
axes[0].set_xlabel('Jenis Pekerjaan')
axes[0].set_ylabel('Jumlah Lowongan')
axes[0].set_title('Jumlah per Jenis Pekerjaan')
for i, row in wt.iterrows():
    axes[0].text(i, row['count'] + 2000, f"{row['pct']:.1f}%", ha='center', fontsize=9)

axes[1].pie(wt['count'], labels=wt['Work Type'], autopct='%1.1f%%',
            colors=palette_wt, startangle=90)
axes[1].set_title('Proporsi Work Type')

plt.tight_layout()
plt.savefig('output/desc_worktype.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — DESCRIPTIVE: TREN TEMPORAL                    ║
# ╚══════════════════════════════════════════════════════════╝
print('\n📊 Tren Temporal Posting Pekerjaan')

sample_dates  = df_pl[date_col].drop_nulls().head(20).to_list()
date_formats  = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
                 '%Y/%m/%d', '%m-%d-%Y', '%d %b %Y', '%b %d, %Y']

detected_format = None
for fmt in date_formats:
    try:
        test = pl.Series(sample_dates[:10]).str.to_date(format=fmt, strict=False)
        if test.drop_nulls().len() >= 7:
            detected_format = fmt
            print(f'✅ Format tanggal: {fmt}')
            break
    except Exception:
        continue

if detected_format is None:
    df_date = df_pl.with_columns(pl.col(date_col).str.to_date(strict=False).alias('posting_date'))
else:
    df_date = df_pl.with_columns(
        pl.col(date_col).str.to_date(format=detected_format, strict=False).alias('posting_date'))

df_date = df_date.filter(pl.col('posting_date').is_not_null())

monthly = (
    df_date.with_columns([
        pl.col('posting_date').dt.year().alias('year'),
        pl.col('posting_date').dt.month().alias('month'),
    ])
    .group_by(['year', 'month'])
    .agg(pl.len().alias('count'))
    .sort(['year', 'month'])
    .with_columns(
        (pl.col('year').cast(str) + '-' +
         pl.col('month').cast(str).str.zfill(2)).alias('period')
    )
)
monthly_pd = monthly.to_pandas().sort_values('period').reset_index(drop=True)
print(f'Bulan terdeteksi: {len(monthly_pd)} | Baris tanggal valid: {df_date.height:,}')

if len(monthly_pd) > 0:
    avg_monthly = monthly_pd['count'].mean()
    idx_max = monthly_pd['count'].idxmax()
    idx_min = monthly_pd['count'].idxmin()
    print(f'Rata-rata/bulan : {avg_monthly:,.0f}  |  '
          f'Tertinggi: {monthly_pd.loc[idx_max,"count"]:,} ({monthly_pd.loc[idx_max,"period"]})  |  '
          f'Terendah: {monthly_pd.loc[idx_min,"count"]:,} ({monthly_pd.loc[idx_min,"period"]})')

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.fill_between(range(len(monthly_pd)), monthly_pd['count'], alpha=0.25, color='#3498db')
    ax.plot(range(len(monthly_pd)), monthly_pd['count'],
            marker='o', color='#2980b9', linewidth=2, markersize=5)
    ax.axhline(avg_monthly, color='orange', linestyle='--', linewidth=1.5,
               label=f'Rata-rata: {avg_monthly:,.0f}')
    ax.set_xticks(range(len(monthly_pd)))
    ax.set_xticklabels(monthly_pd['period'], rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    ax.set_title('Tren Jumlah Job Posting per Bulan (Sep 2021 – Sep 2023)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Bulan-Tahun')
    ax.set_ylabel('Jumlah Job Posting')
    ax.legend()
    ax.annotate(f"Puncak\n{monthly_pd.loc[idx_max,'count']:,}",
                xy=(idx_max, monthly_pd.loc[idx_max,'count']),
                xytext=(idx_max + 1, monthly_pd.loc[idx_max,'count'] * 1.03),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=8, color='green')
    ax.annotate(f"Terendah\n{monthly_pd.loc[idx_min,'count']:,}",
                xy=(idx_min, monthly_pd.loc[idx_min,'count']),
                xytext=(idx_min + 1, monthly_pd.loc[idx_min,'count'] * 0.92),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red')
    plt.tight_layout()
    plt.savefig('output/desc_temporal.png', bbox_inches='tight', dpi=150)
    plt.show()
    print('✅ Grafik tren temporal tersimpan.')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — DESCRIPTIVE: TOP JOB TITLES & COMPANY SIZE    ║
# ╚══════════════════════════════════════════════════════════╝
print('\n📊 Top 10 Job Titles')
top_titles = (
    df_pl.group_by('Job Title')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
    .head(10)
    .to_pandas()
)
print(top_titles.to_string(index=False))

print('\n📊 Distribusi Company Size')
cs_raw  = df_pd_raw[['Company Size']].copy()
cs_raw['category'] = cs_raw['Company Size'].apply(categorize_company_size)
cs_cat  = cs_raw['category'].value_counts().reset_index()
cs_cat.columns = ['Kategori', 'count']
order_cs = ['Very Small (1-10)', 'Small (11-50)', 'Medium (51-200)',
            'Large (201-500)', 'Very Large (>500)']
cs_cat['Kategori'] = pd.Categorical(cs_cat['Kategori'], categories=order_cs, ordered=True)
cs_cat  = cs_cat.sort_values('Kategori').reset_index(drop=True)
cs_cat['pct'] = cs_cat['count'] / cs_cat['count'].sum() * 100
print(cs_cat.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Analisis Job Title dan Ukuran Perusahaan', fontsize=13, fontweight='bold')

axes[0].barh(top_titles['Job Title'][::-1], top_titles['count'][::-1],
             color=sns.color_palette('Blues_d', 10))
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
axes[0].set_xlabel('Jumlah Lowongan')
axes[0].set_title('Top 10 Job Title')
for i, (_, row) in enumerate(top_titles[::-1].iterrows()):
    axes[0].text(row['count'] + 100, i, f"{row['count']:,}", va='center', fontsize=8)

pal_cs  = sns.color_palette('viridis', len(cs_cat))
bars_cs = axes[1].bar(cs_cat['Kategori'], cs_cat['count'], color=pal_cs)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
axes[1].set_xlabel('Kategori Ukuran Perusahaan')
axes[1].set_ylabel('Jumlah Lowongan')
axes[1].set_title('Distribusi Ukuran Perusahaan')
axes[1].set_xticklabels(cs_cat['Kategori'], rotation=20, ha='right', fontsize=8)
for bar, (_, row) in zip(bars_cs, cs_cat.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1000, f"{row['pct']:.1f}%",
                 ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('output/desc_titles_compsize.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9 — DESCRIPTIVE: SALARY & QUALIFICATIONS          ║
# ╚══════════════════════════════════════════════════════════╝
print('\n📊 Distribusi Salary')

df_sal = df_clean[['Salary_Min_K','Salary_Max_K','Salary_Mid_K',
                   'Work Type','Qualifications']].copy()
print(f'   Data salary valid : {len(df_sal):,}')
print(f'   Rata-rata salary  : ${df_sal["Salary_Mid_K"].mean():.1f}K')
print(f'   Median salary     : ${df_sal["Salary_Mid_K"].median():.1f}K')
print(f'   Std Dev           : ${df_sal["Salary_Mid_K"].std():.1f}K')
print(f'   Min               : ${df_sal["Salary_Min_K"].min():.0f}K')
print(f'   Max               : ${df_sal["Salary_Max_K"].max():.0f}K')

sal_wt = df_sal.groupby('Work Type')['Salary_Mid_K'].agg(['mean','median','std']).reset_index()
sal_wt.columns = ['Work Type', 'Mean ($K)', 'Median ($K)', 'Std ($K)']
print(f'\n   Salary per Work Type:\n{sal_wt.to_string(index=False)}')

top_qual = (
    df_pl.group_by('Qualifications')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
    .head(10)
    .to_pandas()
)
print(f'\n📊 Top 10 Kualifikasi:\n{top_qual.to_string(index=False)}')

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle('Analisis Salary dan Kualifikasi', fontsize=14, fontweight='bold')

axes[0,0].hist(df_sal['Salary_Mid_K'], bins=60, color='#3498db', edgecolor='white', alpha=0.8)
axes[0,0].axvline(df_sal['Salary_Mid_K'].mean(), color='red', linestyle='--',
                   label=f'Mean: ${df_sal["Salary_Mid_K"].mean():.1f}K')
axes[0,0].axvline(df_sal['Salary_Mid_K'].median(), color='orange', linestyle='-.',
                   label=f'Median: ${df_sal["Salary_Mid_K"].median():.1f}K')
axes[0,0].set_xlabel('Salary Mid ($K)')
axes[0,0].set_ylabel('Frekuensi')
axes[0,0].set_title('Distribusi Salary Tengah (Mid)')
axes[0,0].legend(fontsize=9)

wt_order  = df_sal.groupby('Work Type')['Salary_Mid_K'].median().sort_values().index.tolist()
data_box  = [df_sal[df_sal['Work Type'] == wt_]['Salary_Mid_K'].values for wt_ in wt_order]
bp = axes[0,1].boxplot(data_box, patch_artist=True, labels=wt_order,
                        medianprops=dict(color='black', linewidth=2))
pal_bp = sns.color_palette('Set2', len(wt_order))
for patch, col in zip(bp['boxes'], pal_bp):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[0,1].set_xlabel('Work Type')
axes[0,1].set_ylabel('Salary Mid ($K)')
axes[0,1].set_title('Distribusi Salary per Work Type')

axes[1,0].barh(top_qual['Qualifications'][::-1], top_qual['count'][::-1],
               color=sns.color_palette('Greens_d', 10))
axes[1,0].set_xlabel('Jumlah Lowongan')
axes[1,0].set_title('Top 10 Kualifikasi')
for i, (_, row) in enumerate(top_qual[::-1].iterrows()):
    axes[1,0].text(row['count'] + 20, i, f"{row['count']:,}", va='center', fontsize=8)

qual_salary = df_sal.groupby('Qualifications')['Salary_Mid_K'].mean().reset_index()
qual_salary.columns = ['Qualifications', 'Mean Salary ($K)']
qual_top10  = qual_salary[qual_salary['Qualifications'].isin(top_qual['Qualifications'])]
qual_top10  = qual_top10.sort_values('Mean Salary ($K)', ascending=True)
axes[1,1].barh(qual_top10['Qualifications'], qual_top10['Mean Salary ($K)'],
               color=sns.color_palette('Oranges_d', len(qual_top10)))
axes[1,1].set_xlabel('Rata-rata Salary ($K)')
axes[1,1].set_title('Rata-rata Salary per Kualifikasi')
for i, (_, row) in enumerate(qual_top10.iterrows()):
    axes[1,1].text(row['Mean Salary ($K)'] + 0.1, i,
                   f"${row['Mean Salary ($K)']:.1f}K", va='center', fontsize=8)

plt.tight_layout()
plt.savefig('desc_salary_qual.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 10 — DESCRIPTIVE: TOP COUNTRIES & JOB PORTAL      ║
# ╚══════════════════════════════════════════════════════════╝
print('\n📊 Top 15 Negara dengan Lowongan Terbanyak')
top_countries = (
    df_pl.group_by('Country')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
    .head(15)
    .to_pandas()
)
print(top_countries.to_string(index=False))

fig, ax = plt.subplots(figsize=(16, 9))
pal_cn = sns.color_palette('coolwarm', len(top_countries))
ax.barh(top_countries['Country'][::-1], top_countries['count'][::-1], color=pal_cn[::-1])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
ax.set_xlabel('Jumlah Lowongan')
ax.set_title('Top 15 Negara dengan Lowongan Terbanyak', fontsize=13, fontweight='bold')
for i, (_, row) in enumerate(top_countries[::-1].iterrows()):
    ax.text(row['count'] + 100, i, f"{row['count']:,}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig('output/desc_countries.png', bbox_inches='tight', dpi=150)
plt.show()

if 'Job Portal' in df_pl.columns:
    print('\n📊 Distribusi Job Portal')
    portal = (
        df_pl.group_by('Job Portal')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .head(10)
        .to_pandas()
    )
    print(portal.to_string(index=False))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(portal['Job Portal'], portal['count'],
           color=sns.color_palette('tab10', len(portal)))
    ax.set_xlabel('Job Portal')
    ax.set_ylabel('Jumlah Lowongan')
    ax.set_title('Distribusi Job Portal', fontsize=13, fontweight='bold')
    ax.set_xticklabels(portal['Job Portal'], rotation=30, ha='right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    plt.tight_layout()
    plt.savefig('output/desc_portals.png', bbox_inches='tight', dpi=150)
    plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 11 — STRATIFIED SAMPLING 10%                      ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  SAMPLING — Stratified 10% untuk Pemodelan ML')
print('='*65)

SAMPLE_FRAC = 0.10
df_sample = (
    df_clean.groupby('Work Type')
    .sample(frac=SAMPLE_FRAC, random_state=42)
    .reset_index(drop=True)
)

print(f'✅ Full dataset (clean) : {len(df_clean):>12,} baris')
print(f'   Sample 10%          : {len(df_sample):>12,} baris')
print(f'\n   Distribusi Work Type dalam sample:')
print(df_sample['Work Type'].value_counts().to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 12 — KLASIFIKASI JOB TITLE (TOP 10)               ║
# ║            Feature Selection: TF-IDF dari Job Desc      ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  KLASIFIKASI JOB TITLE — 3 Algoritma')
print('='*65)
print('\n  Feature Selection: TF-IDF (n-gram 1-2, max 8000 fitur)')
print('  Alasan pemilihan: Job Description adalah teks bebas,')
print('  TF-IDF efektif menangkap kata kunci diskriminatif antar jabatan.')

top10_list = top_titles['Job Title'].tolist()
df_clf     = df_sample[df_sample['Job Title'].isin(top10_list)].copy()
df_clf     = df_clf.dropna(subset=['Job Description', 'Job Title']).reset_index(drop=True)
print(f'\n   Data untuk klasifikasi : {len(df_clf):,} baris')
print(f'   Distribusi label:\n{df_clf["Job Title"].value_counts().to_string()}')

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                        stop_words='english', sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df_clf['Job Description'])
y_clf   = df_clf['Job Title']

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_tfidf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f'\n   Train: {X_tr_c.shape[0]:,} | Test: {X_te_c.shape[0]:,} | Fitur TF-IDF: {X_tfidf.shape[1]:,}')

clf_results = {}

print('\n  ── 1. Naive Bayes ──')
print('     Alasan: Baseline kuat untuk teks; asumsi independensi fitur.')
nb = MultinomialNB(alpha=0.1)
nb.fit(X_tr_c, y_tr_c)
y_pred_nb = nb.predict(X_te_c)
clf_results['Naive Bayes'] = {
    'acc': accuracy_score(y_te_c, y_pred_nb),
    'f1' : f1_score(y_te_c, y_pred_nb, average='weighted', zero_division=0),
    'preds': y_pred_nb
}
print(classification_report(y_te_c, y_pred_nb, zero_division=0))

print('\n  ── 2. Logistic Regression ──')
print('     Alasan: Probabilistik, regularisasi L2, cocok untuk TF-IDF sparse.')
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
lr.fit(X_tr_c, y_tr_c)
y_pred_lr = lr.predict(X_te_c)
clf_results['Logistic Regression'] = {
    'acc': accuracy_score(y_te_c, y_pred_lr),
    'f1' : f1_score(y_te_c, y_pred_lr, average='weighted', zero_division=0),
    'preds': y_pred_lr
}
print(classification_report(y_te_c, y_pred_lr, zero_division=0))

print('\n  ── 3. Random Forest ──')
print('     Alasan: Ensemble, robust terhadap overfitting, non-linear.')
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                random_state=42, n_jobs=-1)
rf_clf.fit(X_tr_c, y_tr_c)
y_pred_rf = rf_clf.predict(X_te_c)
clf_results['Random Forest'] = {
    'acc': accuracy_score(y_te_c, y_pred_rf),
    'f1' : f1_score(y_te_c, y_pred_rf, average='weighted', zero_division=0),
    'preds': y_pred_rf
}
print(classification_report(y_te_c, y_pred_rf, zero_division=0))

print('\n  📋 Ringkasan Perbandingan Model Klasifikasi:')
for name, res in clf_results.items():
    print(f'   {name:<22}: Accuracy={res["acc"]:.4f}  F1-Weighted={res["f1"]:.4f}')

best_clf_name = max(clf_results, key=lambda k: clf_results[k]['acc'])
best_clf_pred = clf_results[best_clf_name]['preds']
print(f'\n   ✅ Model terbaik: {best_clf_name} (Accuracy={clf_results[best_clf_name]["acc"]:.4f})')

# Visualisasi Klasifikasi
model_names = list(clf_results.keys())
accs = [clf_results[m]['acc'] for m in model_names]
f1s  = [clf_results[m]['f1']  for m in model_names]

fig, axes = plt.subplots(1, 3, figsize=(16, 9))
fig.suptitle('Hasil Klasifikasi Job Title — 3 Model', fontsize=14, fontweight='bold')

x = np.arange(len(model_names))
w = 0.35
b1 = axes[0].bar(x - w/2, accs, w, label='Accuracy',    color='#3498db')
b2 = axes[0].bar(x + w/2, f1s,  w, label='F1-Weighted', color='#2ecc71')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, fontsize=9)
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel('Score')
axes[0].set_title('Perbandingan Accuracy & F1-Score')
axes[0].legend()
for bar in b1: axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                              f'{bar.get_height():.4f}', ha='center', fontsize=8)
for bar in b2: axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                              f'{bar.get_height():.4f}', ha='center', fontsize=8)

cm = confusion_matrix(y_te_c, best_clf_pred, labels=top10_list)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=top10_list, yticklabels=top10_list,
            linewidths=0.5, cbar=False)
axes[1].set_xticklabels(top10_list, rotation=45, ha='right', fontsize=7)
axes[1].set_yticklabels(top10_list, fontsize=7)
axes[1].set_xlabel('Prediksi'); axes[1].set_ylabel('Aktual')
axes[1].set_title(f'Confusion Matrix\n{best_clf_name}')

report_df = pd.DataFrame(
    classification_report(y_te_c, best_clf_pred, output_dict=True, zero_division=0)
).T
report_df = report_df.loc[[t for t in top10_list if t in report_df.index]]
axes[2].barh(report_df.index, report_df['f1-score'],
             color=sns.color_palette('Set2', len(report_df)))
axes[2].set_xlabel('F1-Score')
axes[2].set_title(f'F1-Score per Job Title\n({best_clf_name})')
axes[2].set_xlim(0, 1.1)
for i, (idx, row) in enumerate(report_df.iterrows()):
    axes[2].text(row['f1-score'] + 0.01, i, f"{row['f1-score']:.3f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig('output/predictive_classification.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 13 — REGRESI PREDIKSI SALARY                      ║
# ║            Feature Selection: 5 fitur numerik/ordinal   ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  PREDIKSI SALARY — 5 Model Regresi')
print('='*65)
print('\n  Feature Selection:')
print('  wt_enc    : Work Type (LabelEncoded)     — tipe kontrak berpengaruh ke gaji')
print('  qual_enc  : Qualification (LabelEncoded) — level pendidikan proxy seniority')
print('  size_enc  : Company Size (OrdinalEncoded)— perusahaan besar = gaji lebih tinggi')
print('  exp_mean  : Experience (rata-rata tahun) — pengalaman berkorelasi dengan gaji')
print('  title_enc : Job Title (LabelEncoded)     — jabatan adalah determinan utama gaji')

features_reg = ['wt_enc', 'qual_enc', 'size_enc', 'exp_mean', 'title_enc']
df_reg  = df_sample.copy()
X_reg   = df_reg[features_reg]
y_reg   = df_reg['Salary_Mid_K']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
print(f'\n   Train: {len(X_tr_r):,}  |  Test: {len(X_te_r):,}')

reg_models = {
    'Linear Regression'       : LinearRegression(),
    'Ridge Regression'        : Ridge(alpha=1.0),
    'Lasso Regression'        : Lasso(alpha=0.1, max_iter=2000),
    'Random Forest Regressor' : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

reg_results = []
fitted_reg  = {}
for name, model in reg_models.items():
    model.fit(X_tr_r, y_tr_r)
    yp   = model.predict(X_te_r)
    r2   = r2_score(y_te_r, yp)
    mae  = mean_absolute_error(y_te_r, yp)
    rmse = np.sqrt(mean_squared_error(y_te_r, yp))
    reg_results.append({'Model': name, 'R²': r2, 'MAE ($K)': mae, 'RMSE ($K)': rmse})
    fitted_reg[name] = (model, yp)
    print(f'   {name:<28}: R²={r2:.4f}  MAE={mae:.2f}K  RMSE={rmse:.2f}K')

df_reg_results = pd.DataFrame(reg_results)
best_reg_name  = df_reg_results.loc[df_reg_results['R²'].idxmax(), 'Model']
best_reg_pred  = fitted_reg[best_reg_name][1]
print(f'\n   ✅ Model terbaik: {best_reg_name} (R²={df_reg_results["R²"].max():.4f})')

# Feature Importance jika RF
fi = None
if 'Random Forest' in best_reg_name:
    rf_reg = fitted_reg[best_reg_name][0]
    fi = pd.Series(rf_reg.feature_importances_, index=features_reg).sort_values(ascending=True)

nplots = 3 if fi is not None else 2
fig, axes = plt.subplots(1, nplots, figsize=(18 if fi is not None else 12, 6))
fig.suptitle('Prediksi Salary — Hasil Model Regresi', fontsize=14, fontweight='bold')

colors_r2 = ['#e74c3c' if r < 0.5 else '#f39c12' if r < 0.7 else '#2ecc71'
              for r in df_reg_results['R²']]
bars_r2   = axes[0].bar(df_reg_results['Model'], df_reg_results['R²'],
                        color=colors_r2, width=0.5)
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel('R² Score')
axes[0].set_title('Perbandingan R² Score')
axes[0].set_xticklabels(df_reg_results['Model'], rotation=20, ha='right', fontsize=8)
for bar, r2 in zip(bars_r2, df_reg_results['R²']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{r2:.4f}', ha='center', fontsize=8)

axes[1].scatter(y_te_r, best_reg_pred, alpha=0.2, s=8, color='#3498db')
mn, mx = y_te_r.min(), y_te_r.max()
axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal (y=x)')
axes[1].set_xlabel('Salary Aktual ($K)')
axes[1].set_ylabel('Salary Prediksi ($K)')
axes[1].set_title(f'Actual vs Predicted\n{best_reg_name}')
axes[1].legend(fontsize=9)
r2_val = df_reg_results.loc[df_reg_results['Model'] == best_reg_name, 'R²'].values[0]
axes[1].text(0.05, 0.92, f'R² = {r2_val:.4f}', transform=axes[1].transAxes,
             fontsize=10, color='red', fontweight='bold')

if fi is not None:
    axes[2].barh(fi.index, fi.values, color=sns.color_palette('Oranges_d', len(fi)))
    axes[2].set_xlabel('Feature Importance')
    axes[2].set_title(f'Feature Importance\n{best_reg_name}')
    for i, (idx, val) in enumerate(fi.items()):
        axes[2].text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('output/predictive_regression.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 14 — K-MEANS KLASTERISASI PERUSAHAAN              ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  K-MEANS KLASTERISASI SEGMENTASI PERUSAHAAN')
print('='*65)
print('\n  Fitur klasterisasi: size_enc, Salary_Mid_K, one-hot Work Type')
print('  Alasan K-Means: unsupervised, cocok menemukan segmen alami perusahaan.')

df_km = df_sample.copy()
df_km = df_km.dropna(subset=['Salary_Mid_K', 'Work Type', 'cs_cat'])

wt_dummies  = pd.get_dummies(df_km['Work Type'], prefix='wt', dtype=float)
df_km       = pd.concat([df_km.reset_index(drop=True), wt_dummies], axis=1)
# Gunakan oe_size yang sudah di-fit di Cell 5
df_km['size_enc_km'] = oe_size.transform(df_km[['cs_cat']])

wt_cols     = list(wt_dummies.columns)
feature_km  = ['size_enc_km', 'Salary_Mid_K'] + wt_cols
X_km        = df_km[feature_km].dropna()
df_km       = df_km.loc[X_km.index].reset_index(drop=True)

scaler_km   = MinMaxScaler()
X_km_scaled = scaler_km.fit_transform(X_km)

print('  ⏳ Menghitung Elbow Method & Silhouette Score (k=2..8)...')
wcss_list, sil_list = [], []
k_range = range(2, 9)
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km_tmp.fit(X_km_scaled)
    wcss_list.append(km_tmp.inertia_)
    sil = silhouette_score(X_km_scaled, km_tmp.labels_)
    sil_list.append(sil)
    print(f'   k={k}: WCSS={km_tmp.inertia_:>10.2f}  Silhouette={sil:.4f}')

best_k_sil = list(k_range)[np.argmax(sil_list)]
print(f'\n   ✅ k optimal (Silhouette): k={best_k_sil}  (Score={max(sil_list):.4f})')
best_k = 3  # domain knowledge: startup / menengah / korporasi

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
df_km['cluster'] = km_final.fit_predict(X_km_scaled)
final_sil = silhouette_score(X_km_scaled, df_km['cluster'])
print(f'  Silhouette Score final (k=3): {final_sil:.4f}')

cluster_profile = (
    df_km.groupby('cluster')
    .agg(n_records=('cluster','count'),
         avg_salary=('Salary_Mid_K','mean'),
         med_salary=('Salary_Mid_K','median'),
         avg_size_enc=('size_enc_km','mean'))
    .reset_index().sort_values('avg_salary').reset_index(drop=True)
)
print('\n  Profil Klaster:')
print(cluster_profile.to_string(index=False))

label_map = {
    cluster_profile.loc[0, 'cluster']: 'Startup / Usaha Kecil',
    cluster_profile.loc[1, 'cluster']: 'Perusahaan Menengah',
    cluster_profile.loc[2, 'cluster']: 'Korporasi Besar',
}
df_km['cluster_label'] = df_km['cluster'].map(label_map)

cluster_colors = {
    'Startup / Usaha Kecil': '#e74c3c',
    'Perusahaan Menengah'   : '#f39c12',
    'Korporasi Besar'       : '#2ecc71',
}

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle('K-Means Klasterisasi Segmentasi Perusahaan (k=3)',
             fontsize=14, fontweight='bold')

axes[0,0].plot(list(k_range), wcss_list, marker='o', color='#e74c3c', linewidth=2)
axes[0,0].axvline(best_k, color='blue', linestyle='--', alpha=0.6, label=f'k={best_k} dipilih')
axes[0,0].set_xlabel('k'); axes[0,0].set_ylabel('WCSS')
axes[0,0].set_title('Elbow Method'); axes[0,0].set_xticks(list(k_range))
axes[0,0].legend()

axes[0,1].plot(list(k_range), sil_list, marker='s', color='#2ecc71', linewidth=2)
axes[0,1].axvline(best_k, color='blue', linestyle='--', alpha=0.6, label=f'k={best_k} dipilih')
axes[0,1].set_xlabel('k'); axes[0,1].set_ylabel('Silhouette Score')
axes[0,1].set_title('Silhouette Score per k'); axes[0,1].set_xticks(list(k_range))
axes[0,1].legend()
axes[0,1].text(0.02, 0.05, f'Silhouette final (k={best_k}): {final_sil:.4f}',
               transform=axes[0,1].transAxes, fontsize=9, color='blue')

for label, grp in df_km.groupby('cluster_label'):
    axes[1,0].hist(grp['Salary_Mid_K'], bins=40, alpha=0.55,
                   label=label, color=cluster_colors.get(label, 'grey'))
axes[1,0].set_xlabel('Salary Mid ($K)'); axes[1,0].set_ylabel('Frekuensi')
axes[1,0].set_title('Distribusi Salary per Klaster'); axes[1,0].legend(fontsize=8)

ordered_labels = ['Startup / Usaha Kecil', 'Perusahaan Menengah', 'Korporasi Besar']
plot_data = [df_km[df_km['cluster_label'] == lbl]['Salary_Mid_K'].values
             for lbl in ordered_labels]
bp2 = axes[1,1].boxplot(plot_data, patch_artist=True,
                          labels=['Startup/Kecil', 'Menengah', 'Korporasi'],
                          medianprops=dict(color='black', linewidth=2),
                          flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, lbl in zip(bp2['boxes'], ordered_labels):
    patch.set_facecolor(cluster_colors[lbl]); patch.set_alpha(0.75)
axes[1,1].set_ylabel('Salary Mid ($K)')
axes[1,1].set_title(f'Boxplot Salary per Klaster\n(Silhouette = {final_sil:.4f})')

plt.tight_layout()
plt.savefig('output/clustering_result.png', bbox_inches='tight', dpi=150)
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 15 — RINGKASAN AKHIR & BIG DATA 3Vs PROOF         ║
# ╚══════════════════════════════════════════════════════════╝
print('\n' + '='*65)
print('  RINGKASAN AKHIR — BIG DATA 3Vs PROOF')
print('='*65)

best_clf_acc = clf_results[best_clf_name]['acc']
best_clf_f1  = clf_results[best_clf_name]['f1']
best_r2      = df_reg_results['R²'].max()
best_mae     = df_reg_results.loc[df_reg_results['R²'].idxmax(), 'MAE ($K)']

print(f"""
┌─────────────────┬──────────────────────────────────────────────────────┐
│  DIMENSI 3Vs    │  BUKTI DARI DATASET                                  │
├─────────────────┼──────────────────────────────────────────────────────┤
│  VOLUME         │  {df_pl.height:>10,} baris | {size_gb:.2f} GB                        │
│                 │  Melampaui batas Excel (1.048.576 baris)             │
│                 │  Total {total_cells:,} nilai data                     │
├─────────────────┼──────────────────────────────────────────────────────┤
│  VARIETY        │  {df_pl.width} kolom | 7 tipe data berbeda (num, teks, geo) │
│                 │  16 job portal | 216 negara | datetime + koordinat   │
│                 │  Teks bebas (Job Description, Skills, Responsibilities)│
├─────────────────┼──────────────────────────────────────────────────────┤
│  VERACITY       │  Kelengkapan: {completeness:.4f}%                           │
│                 │  Duplikat: {n_dup} | Salary valid: {salary_valid:,}        │
│                 │  Koordinat lat/lon valid: {lat_valid:,}               │
└─────────────────┴──────────────────────────────────────────────────────┘

┌──────────────────────┬────────────────────────────────────────────────┐
│  HASIL PEMODELAN ML  │                                                │
├──────────────────────┼────────────────────────────────────────────────┤
│  Klasifikasi         │  {best_clf_name:<26}                          │
│                      │  Accuracy = {best_clf_acc:.4f} | F1 = {best_clf_f1:.4f}           │
├──────────────────────┼────────────────────────────────────────────────┤
│  Prediksi Salary     │  {best_reg_name:<26}                          │
│                      │  R² = {best_r2:.4f} | MAE = {best_mae:.2f}K                 │
├──────────────────────┼────────────────────────────────────────────────┤
│  Klasterisasi        │  K-Means k=3 (Startup / Menengah / Korporasi) │
│                      │  Silhouette Score = {final_sil:.4f}                    │
└──────────────────────┴────────────────────────────────────────────────┘
""")

print('📁 File output grafik:')
output_files = [
    ('quality_before.png',            'Data Quality Sebelum Transformasi'),
    ('quality_before_after.png',      'Perbandingan Before-After Transformasi'),
    ('desc_worktype.png',             'Distribusi Work Type'),
    ('desc_temporal.png',             'Tren Temporal Posting per Bulan'),
    ('desc_titles_compsize.png',      'Top 10 Job Titles & Company Size'),
    ('desc_salary_qual.png',          'Distribusi Salary & Kualifikasi'),
    ('desc_countries.png',            'Top 15 Negara'),
    ('desc_portals.png',              'Distribusi Job Portal'),
    ('predictive_classification.png', 'Klasifikasi Job Title — 3 Model'),
    ('predictive_regression.png',     'Prediksi Salary — 5 Model Regresi'),
    ('clustering_result.png',         'K-Means Klasterisasi Perusahaan'),
]
for fname, desc in output_files:
    print(f'   📊 {fname:<40} — {desc}')

print('\n✅ Seluruh analisis selesai!')