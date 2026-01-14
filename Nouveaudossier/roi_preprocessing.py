import time
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import os

# =========================================================
# CONFIGURATION
# =========================================================
DATAFRAME_PATH = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\final_dataset_all_patients.xlsx"
PATIENT_INDEX = 0      # index du patient à tester
IMG_SIZE = 224

IMAGE_COLUMNS = [
    "RCC_Path",
    "LCC_Path",
    "RMLO_Path",
    "LMLO_Path"
]

# =========================================================
# IMAGE UTILS
# =========================================================
def load_dicom(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array.astype(np.float32)


def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img


def segment_breast(img):
    img_u8 = (img * 255).astype(np.uint8)

    _, thresh = cv2.threshold(
        img_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)

    return mask


def crop_to_roi(img, mask):
    coords = np.column_stack(np.where(mask > 0))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img[y0:y1, x0:x1]


def preprocess_roi(dicom_path):
    img = load_dicom(dicom_path)
    img = normalize(img)
    mask = segment_breast(img)
    roi = crop_to_roi(img, mask)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    return roi


# =========================================================
# TEST SINGLE PATIENT
# =========================================================
def test_single_patient():
    df = pd.read_excel(DATAFRAME_PATH)
    row = df.iloc[PATIENT_INDEX]

    case = row["CaseNumber"]
    print(f"🧪 Test du patient : {case}")

    start = time.time()
    rois = []

    for col in IMAGE_COLUMNS:
        path = row[col]

        if pd.isna(path) or not os.path.exists(path):
            print(f"❌ Image manquante : {col}")
            return

        roi = preprocess_roi(path)
        rois.append(roi)

    elapsed = time.time() - start

    # =====================================================
    # AFFICHAGE
    # =====================================================
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    for i, roi in enumerate(rois):
        axs[i].imshow(roi, cmap="gray")
        axs[i].set_title(IMAGE_COLUMNS[i])
        axs[i].axis("off")

    plt.suptitle(f"ROI – Patient {case}", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"✔ 4 vues traitées avec succès")
    print(f"⏱ Temps total : {elapsed:.2f} secondes")
    print(f"📐 Taille ROI : {rois[0].shape}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_single_patient()
