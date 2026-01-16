import os
import time
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener 
from skimage.restoration import denoise_tv_chambolle

# =========================================================
# CONFIGURATION (CHEMINS RÉELS)
# =========================================================
DATAFRAME_PATH = r"E:\Dataset\Bilgi\BilgiKìsmìna_Exceller\final_dataset_all_patients.xlsx"
PATIENT_INDEX = 4# changer pour tester un autre patient
IMG_SIZE = 160             # 160 recommandé (plus rapide que 224)

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


# ================= SEGMENTATION SEIN =====================
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


# ================= FILTRES ===============================
def median_filter(img, ksize=5):
    return cv2.medianBlur((img * 255).astype(np.uint8), ksize).astype(np.float32) / 255.0


def wiener_filter(img, ksize=5):
    return wiener(img, (ksize, ksize))


def apply_clahe(img):
    img_u8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(img_u8)
    return cl.astype(np.float32) / 255.0


def anisotropic_diffusion(img, weight=0.1, iterations=8):
    return denoise_tv_chambolle(img, weight=weight, max_num_iter=iterations)


# ================= COLORISATION DES RÉGIONS SOMBRES ======
def apply_color_to_dark_regions(img, threshold=0.4, colormap='hot'):
    """
    Applique une colormap avec dégradation sur les régions sombres (tumeur)
    et garde le reste en niveaux de gris.
    
    Args:
        img: Image en niveaux de gris (0-1)
        threshold: Seuil pour déterminer les pixels sombres (0-1)
        colormap: Colormap à utiliser ('hot', 'jet', 'viridis', 'plasma', etc.)
    
    Returns:
        Image RGB avec couleurs sur les régions sombres
    """
    # Créer une image RGB de base en niveaux de gris
    img_rgb = np.stack([img, img, img], axis=-1)
    
    # Identifier les pixels sombres (tumeur)
    dark_mask = img < threshold
    
    if np.any(dark_mask):
        # Normaliser les valeurs sombres pour la colormap (0=plus noir, 1=moins noir)
        dark_values = img[dark_mask]
        normalized_dark = (dark_values - dark_values.min()) / (dark_values.max() - dark_values.min() + 1e-6)
        
        # Appliquer la colormap
        cmap = plt.get_cmap(colormap)
        colors = cmap(normalized_dark)[:, :3]  # RGB seulement (sans alpha)
        
        # Remplacer les pixels sombres par les couleurs
        img_rgb[dark_mask] = colors
    
    return img_rgb


# ================= PIPELINE COMPLET ======================
def preprocess_roi(dicom_path,
                   use_median=True,
                   use_wiener=False,
                   use_diffusion=False,
                   use_clahe=True):

    img = load_dicom(dicom_path)
    img = normalize(img)

    mask = segment_breast(img)
    roi = crop_to_roi(img, mask)

    if use_median:
        roi = median_filter(roi)

    if use_wiener:
        roi = wiener_filter(roi)

    if use_diffusion:
        roi = anisotropic_diffusion(roi)

    if use_clahe:
        roi = apply_clahe(roi)

    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    return roi


# =========================================================
# TEST SUR UN SEUL PATIENT
# =========================================================
def test_single_patient():
    df = pd.read_excel(DATAFRAME_PATH)
    row = df.iloc[PATIENT_INDEX]

    case = row["CaseNumber"]
    print(f"🧪 Patient testé : {case}")

    rois = []
    start = time.time()

    for col in IMAGE_COLUMNS:
        path = row[col]

        if pd.isna(path) or not os.path.exists(path):
            print(f"❌ Image manquante : {col}")
            return

        roi = preprocess_roi(
            path,
            use_median=True,
            use_wiener=False,
            use_diffusion=False,
            use_clahe=True
        )
        rois.append(roi)

    elapsed = time.time() - start

    # ================= VISUALISATION ======================
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    for i, roi in enumerate(rois):
        # Créer une image RGB pour appliquer la colormap uniquement sur les régions sombres
        roi_colored = apply_color_to_dark_regions(roi)
        
        axs[i].imshow(roi_colored)
        axs[i].set_title(IMAGE_COLUMNS[i].replace("_Path", ""))
        axs[i].axis("off")

    plt.suptitle(f"ROI finale avec visualisation couleur des tumeurs – Patient {case}", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✔ 4 vues traitées avec succès")
    print(f"⏱ Temps total : {elapsed:.2f} secondes")
    print(f"📐 Taille finale : {rois[0].shape}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_single_patient()   