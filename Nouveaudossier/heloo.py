import pandas as pd
import os

# =========================================================
# PATHS
# =========================================================
SUPP_TRAIN1 = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\Supplementary_TRAIN1.xlsx"
SUPP_TRAIN2 = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\Supplementary_TRAIN2.xlsx"
AGE_PATH    = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\age_bins_with_decades_for suppl..xlsx"

DICOM_ROOTS = [
    r"E:\Dataset\Part_1",
    r"E:\Dataset\MG_training"
]

OUTPUT_PATH = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\final_dataset_all_patients.xlsx"

# =========================================================
# UTILS
# =========================================================
def load_and_clean_supp(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df["BI_RADS"] = (
        df["BI-RADS 0/1/2/4/5"]
        .astype(str)
        .str.extract(r"(\d)")
        .astype(float)
        .astype("Int64")
    )

    df = df[
        [
            "CASENUMBER",
            "BREAST DENSITY",
            "QUADRANT INFORMATION (RIGHT)",
            "QUADRANT INFORMATION (LEFT)",
            "BI_RADS",
        ]
    ]

    df["CASENUMBER"] = df["CASENUMBER"].astype(str).str.strip()
    return df


def find_dicom_path(case_number, view, roots):
    for root in roots:
        path = os.path.join(root, case_number, f"{view}.dcm")
        if os.path.exists(path):
            return path
    return None


# =========================================================
# LOAD & MERGE LABELS (TRAIN1 + TRAIN2)
# =========================================================
df_train1 = load_and_clean_supp(SUPP_TRAIN1)
print(f"Patients dans TRAIN1 (Part_1) : {len(df_train1)}")

df_train2 = load_and_clean_supp(SUPP_TRAIN2)
print(f"Patients dans TRAIN2 (MG_training) : {len(df_train2)}")

df_labels = pd.concat([df_train1, df_train2], ignore_index=True)
df_labels = df_labels.drop_duplicates(subset="CASENUMBER")

print(f"Total patients annotés après fusion : {len(df_labels)}")

# =========================================================
# LOAD AGE
# =========================================================
df_age = pd.read_excel(AGE_PATH)
df_age.columns = ["CASENUMBER", "Age_Bin"]
df_age["CASENUMBER"] = df_age["CASENUMBER"].astype(str).str.strip()

df_labels = df_labels.merge(df_age, on="CASENUMBER", how="left")

# =========================================================
# COUPLING WITH DICOM
# =========================================================
views = ["RCC", "LCC", "RMLO", "LMLO"]
records = []

for _, row in df_labels.iterrows():
    case = row["CASENUMBER"]

    record = {
        "CaseNumber": case,
        "BI_RADS": row["BI_RADS"],
        "Breast_Density": row["BREAST DENSITY"],
        "Age_Bin": row["Age_Bin"],
        "Quadrant_Right": row["QUADRANT INFORMATION (RIGHT)"],
        "Quadrant_Left": row["QUADRANT INFORMATION (LEFT)"],
    }

    for view in views:
        record[f"{view}_Path"] = find_dicom_path(case, view, DICOM_ROOTS)

    records.append(record)

# =========================================================
# FINAL DATAFRAME
# =========================================================
df_final = pd.DataFrame(records)

df_final["Quadrant_Right"] = df_final["Quadrant_Right"].replace("", "NaN")
df_final["Quadrant_Left"] = df_final["Quadrant_Left"].replace("", "NaN")

print(df_final.head())
print("TOTAL FINAL PATIENTS :", len(df_final))

# =========================================================
# SAVE
# =========================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_final.to_excel(OUTPUT_PATH, index=False)

print("Dataset global sauvegardé :", OUTPUT_PATH)
