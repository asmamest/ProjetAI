import pandas as pd
import os

# =========================================================
# PATHS ABSOLUS
# =========================================================
SUPP_PATH = r"D:\Dataset\Bilgi\BilgiKìsmìna_Exceller\Supplementary_TRAIN1.xlsx"
AGE_PATH  = r"D:\Dataset\Bilgi\BilgiKìsmìna_Exceller\age_bins_with_decades_for suppl..xlsx"
DICOM_ROOT = r"D:\Dataset\Part_1"
OUTPUT_PATH = r"D:\Dataset\Bilgi\BilgiKìsmìna_Exceller\final_dataset_per_case.xlsx"

# =========================================================
# LOAD SUPPLEMENTARY FILE
# =========================================================
df_sup = pd.read_excel(SUPP_PATH)
df_sup.columns = df_sup.columns.str.strip()

# Extraire BI-RADS numérique
df_sup["BI_RADS"] = df_sup["BI-RADS 0/1/2/4/5"].astype(str).str.extract(r"(\d)").astype(int)

# Garder uniquement les colonnes utiles
df_sup = df_sup[
    [
        "CASENUMBER",
        "BREAST DENSITY",
        "QUADRANT INFORMATION (RIGHT)",
        "QUADRANT INFORMATION (LEFT)",
        "BI_RADS",
    ]
]

# =========================================================
# LOAD AGE BINS
# =========================================================
df_age = pd.read_excel(AGE_PATH)
df_age.columns = ["CASENUMBER", "Age_Bin"]

# =========================================================
# MERGE METADATA
# =========================================================
df_meta = df_sup.merge(df_age, on="CASENUMBER", how="left")

# =========================================================
# COUPLING WITH DICOM IMAGES (1 LINE PER CASE)
# =========================================================
views = ["RCC", "LCC", "RMLO", "LMLO"]
records = []

for _, row in df_meta.iterrows():
    case = str(row["CASENUMBER"])
    case_dir = os.path.join(DICOM_ROOT, case)

    # dictionnaire pour une seule ligne par patient
    record = {
        "CaseNumber": case,
        "BI_RADS": row["BI_RADS"],
        "Breast_Density": row["BREAST DENSITY"],
        "Age_Bin": row["Age_Bin"],
        "Quadrant_Right": row["QUADRANT INFORMATION (RIGHT)"],
        "Quadrant_Left": row["QUADRANT INFORMATION (LEFT)"],
    }

    # Ajouter les chemins des images par vue
    for view in views:
        dicom_path = os.path.join(case_dir, f"{view}.dcm")
        record[f"{view}_Path"] = dicom_path if os.path.exists(dicom_path) else None

    records.append(record)

# =========================================================
# FINAL DATAFRAME
# =========================================================
df_final = pd.DataFrame(records)

# =========================================================
# REMPLACER LES CHAMPS VIDES PAR "NaN"
# =========================================================
df_final["Quadrant_Right"] = df_final["Quadrant_Right"].replace("", "NaN")
df_final["Quadrant_Left"] = df_final["Quadrant_Left"].replace("", "NaN")

print(df_final.head())
print("Total patients:", len(df_final))

# =========================================================
# SAUVEGARDE EN EXCEL
# =========================================================
df_final.to_excel(OUTPUT_PATH, index=False)
print(f"Dataset saved to {OUTPUT_PATH}")
