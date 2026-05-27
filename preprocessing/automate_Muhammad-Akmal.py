import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_path):
    """
    Fungsi otomatisasi data preprocessing untuk Heart Disease Dataset.
    Membaca data mentah, membersihkan, melakukan encoding & scaling, 
    lakhis menyimpan hasilnya ke folder tujuan.
    """
    print(f"[*] Membaca dataset dari: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan!")
        
    df = pd.read_csv(input_path)
    df_clean = df.copy()

    # 1. Menentukan kolom target secara dinamis
    target_col = 'target' if 'target' in df_clean.columns else df_clean.columns[-1]
    print(f"[*] Kolom target diidentifikasi: '{target_col}'")

    # 2. Menghapus Data Duplikat
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"[+] Berhasil menghapus {duplicate_count} baris duplikat.")
    else:
        print("[*] Tidak ditemukan data duplikat.")

    # 3. Menangani Missing Values (jika ada)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    print("[+] Penanganan missing values selesai.")

    # 4. Encoding Data Kategorikal
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            df_clean[col] = le.fit_transform(df_clean[col])
        print(f"[+] Berhasil melakukan encoding pada kolom: {list(categorical_cols)}")

    # 5. Feature Scaling (Standardization)
    feature_cols = [col for col in df_clean.columns if col != target_col]
    scaler = StandardScaler()
    df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
    print("[+] Proses scaling fitur selesai.")

    # 6. Menyimpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[√] Sukses! Data hasil preprocessing disimpan ke: {output_path}\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(BASE_DIR, "..", "heart_raw", "heart.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "heart_preprocessing.csv")
    
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)