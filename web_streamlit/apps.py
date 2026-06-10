from pathlib import Path
import traceback

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CLASSIFIER_PATH = MODEL_DIR / "best_rf_all_features.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_MASK_PATH = MODEL_DIR / "selected_features_ga.npy"
EXTRACTOR_PATH = MODEL_DIR / "vgg19_attention_extractor.h5"

IMG_SIZE = (224, 224)
NORMALIZATION_MODE = "vgg19"

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

CLASS_NAMES = {
    0: "Grade 0 - Normal",
    1: "Grade 1 - Doubtful",
    2: "Grade 2 - Mild",
    3: "Grade 3 - Moderate",
    4: "Grade 4 - Severe",
}

GRADE_DESCRIPTIONS = {
    0: "Tidak tampak tanda radiografis osteoarthritis lutut.",
    1: "Kemungkinan perubahan sangat awal, diagnosis masih meragukan.",
    2: "Osteofit mulai tampak, dengan perubahan ringan.",
    3: "Perubahan sedang, biasanya disertai penyempitan celah sendi yang lebih jelas.",
    4: "Perubahan berat, penyempitan celah sendi berat, dan kemungkinan deformitas tulang.",
}

# ============================================================
# Utility Streamlit
# ============================================================

def check_file_valid(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File belum ditemukan: {path}")

    if path.stat().st_size == 0:
        raise ValueError(
            f"File kosong / 0 byte: {path.name}. "
            "Export ulang file ini dari notebook training."
        )

def file_status_text(path: Path):
    if not path.exists():
        return "tidak ada"
    size = path.stat().st_size
    if size == 0:
        return "ada, tetapi kosong / 0 byte"
    return f"ada, {size / (1024 * 1024):.2f} MB"

def st_image_compatible(image, caption=None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)

def st_dataframe_compatible(df):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df)

# ============================================================
# Preprocessing
# ============================================================

def resize_with_padding(image, target_size=(224, 224)):
    """
    Me-resize gambar ke target_size dengan mempertahankan aspect ratio asli.
    Sisi yang kosong akan diisi dengan padding hitam (0).
    """
    h, w = image.shape[:2]
    
    aspect_original = w / h
    aspect_target = target_size[0] / target_size[1]
    
    if aspect_original > aspect_target:
        new_w = target_size[0]
        new_h = int(new_w / aspect_original)
    else:
        new_h = target_size[1]
        new_w = int(new_h * aspect_original)
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left
    
    padded_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_img

def crop_with_padding(image, x1, y1, x2, y2, pad_value=0):
    h, w = image.shape[:2]
    crop_w = int(x2 - x1)
    crop_h = int(y2 - y1)

    if crop_w <= 0 or crop_h <= 0:
        return image.copy()

    if image.ndim == 2:
        canvas = np.full((crop_h, crop_w), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((crop_h, crop_w, image.shape[2]), pad_value, dtype=image.dtype)

    src_x1, src_y1 = max(0, int(x1)), max(0, int(y1))
    src_x2, src_y2 = min(w, int(x2)), min(h, int(y2))

    dst_x1, dst_y1 = src_x1 - int(x1), src_y1 - int(y1)
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        if image.ndim == 2:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        else:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
    return canvas

def crop_knee_joint_area(image_rgb, center_x_ratio=0.50, center_y_ratio=0.52, crop_scale=0.85):
    image_h, image_w = image_rgb.shape[:2]
    center_x = int(center_x_ratio * image_w)
    center_y = int(center_y_ratio * image_h)

    side = int(min(image_h, image_w) * crop_scale)
    side = max(side, 224)

    x1 = int(center_x - side / 2)
    y1 = int(center_y - side / 2)
    x2 = x1 + side
    y2 = y1 + side

    crop_image = crop_with_padding(image_rgb, x1, y1, x2, y2, pad_value=0)

    bbox = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": center_x, "center_y": center_y,
        "side": side, "mode": "manual_knee_joint_square_crop"
    }
    return crop_image, bbox

def draw_crop_box(image_rgb, bbox):
    image_draw = image_rgb.copy()
    h, w = image_draw.shape[:2]
    x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
    x2, y2 = min(w - 1, int(bbox["x2"])), min(h - 1, int(bbox["y2"]))

    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (255, 0, 0), 4)

    center_y = int(np.clip(bbox["center_y"], 0, h - 1))
    center_x = int(np.clip(bbox["center_x"], 0, w - 1))

    cv2.line(image_draw, (0, center_y), (w - 1, center_y), (0, 255, 0), 2)
    cv2.line(image_draw, (center_x, 0), (center_x, h - 1), (0, 255, 0), 2)

    return image_draw

def apply_clahe_rgb(image_rgb: np.ndarray):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    image_bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    image_rgb_clahe = cv2.cvtColor(image_bgr_clahe, cv2.COLOR_BGR2RGB)

    return image_rgb_clahe

def preprocess_uploaded_image(uploaded_file, center_x_ratio=0.50, center_y_ratio=0.52, crop_scale=0.85):
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)

    crop_image, bbox = crop_knee_joint_area(
        image_rgb, center_x_ratio=center_x_ratio, center_y_ratio=center_y_ratio, crop_scale=crop_scale
    )

    image_with_crop = draw_crop_box(image_rgb, bbox)

    # PERUBAHAN: Menggunakan fungsi padding yang telah dirancang untuk menjaga aspect ratio
    resized_crop = resize_with_padding(crop_image, target_size=IMG_SIZE)

    model_image = apply_clahe_rgb(resized_crop)
    image_float = model_image.astype(np.float32)

    if NORMALIZATION_MODE == "vgg19":
        image_float = tf.keras.applications.vgg19.preprocess_input(image_float)
    elif NORMALIZATION_MODE == "rescale_01":
        image_float = image_float / 255.0
    else:
        raise ValueError(f"NORMALIZATION_MODE tidak dikenal: {NORMALIZATION_MODE}")

    batch = np.expand_dims(image_float, axis=0)

    return {
        "original_image": image_rgb,
        "image_with_crop": image_with_crop,
        "crop_image": crop_image,
        "resized_crop": resized_crop,
        "model_image": model_image,
        "batch": batch,
        "bbox": bbox
    }

# ============================================================
# Feature Pipeline
# ============================================================

def flatten_features(features: np.ndarray):
    features = np.asarray(features)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    return features

def apply_selected_features(features: np.ndarray, selected_features):
    if selected_features is None:
        return features

    selected_features = np.asarray(selected_features)
    if selected_features.dtype == object and selected_features.size == 1:
        selected_features = np.asarray(selected_features.item())

    if selected_features.dtype == bool:
        return features[:, selected_features]
    return features[:, selected_features.astype(int)]

def get_selected_feature_indices(selected_features):
    selected_features = np.asarray(selected_features)
    if selected_features.dtype == object and selected_features.size == 1:
        selected_features = np.asarray(selected_features.item())

    if selected_features.dtype == bool:
        return np.where(selected_features)[0]
    return selected_features.astype(int)

def make_selected_feature_dataframe(raw_features, scaled_features, selected_features):
    selected_indices = get_selected_feature_indices(selected_features)
    rows = []

    for order, idx in enumerate(selected_indices):
        rows.append({
            "selected_order": int(order),
            "original_feature_index": int(idx),
            "raw_vgg19_feature": float(raw_features[0, idx]),
            "scaled_feature": float(scaled_features[0, idx])
        })
    return pd.DataFrame(rows)

# ============================================================
# Model Loading
# ============================================================

def get_tf_custom_objects():
    custom_objects = {}
    try:
        from tensorflow.python.keras.layers.core import TFOpLambda, SlicingOpLambda
        custom_objects["TFOpLambda"] = TFOpLambda
        custom_objects["SlicingOpLambda"] = SlicingOpLambda
    except Exception:
        pass

    custom_objects.update({
        "Add": tf.keras.layers.Add, "Multiply": tf.keras.layers.Multiply,
        "GlobalAveragePooling2D": tf.keras.layers.GlobalAveragePooling2D,
        "GlobalMaxPooling2D": tf.keras.layers.GlobalMaxPooling2D,
        "Dense": tf.keras.layers.Dense, "Reshape": tf.keras.layers.Reshape,
        "Conv2D": tf.keras.layers.Conv2D, "Activation": tf.keras.layers.Activation,
        "Lambda": tf.keras.layers.Lambda, "Concatenate": tf.keras.layers.Concatenate,
        "Dropout": tf.keras.layers.Dropout, "BatchNormalization": tf.keras.layers.BatchNormalization,
    })
    return custom_objects

def load_extractor_h5(path: Path):
    check_file_valid(path)
    custom_objects = get_tf_custom_objects()
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            extractor = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        with tf.keras.utils.custom_object_scope(custom_objects):
            extractor = tf.keras.models.load_model(path, compile=False)
    return extractor

@st.cache_resource(show_spinner=False)
def load_assets():
    check_file_valid(CLASSIFIER_PATH)
    check_file_valid(SCALER_PATH)
    check_file_valid(FEATURE_MASK_PATH)
    check_file_valid(EXTRACTOR_PATH)

    classifier = joblib.load(CLASSIFIER_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = np.load(FEATURE_MASK_PATH, allow_pickle=True)
    extractor = load_extractor_h5(EXTRACTOR_PATH)

    return classifier, scaler, selected_features, extractor

def predict_koa(batch: np.ndarray):
    classifier, scaler, selected_features, extractor = load_assets()

    raw_features = extractor.predict(batch, verbose=0)
    raw_features = flatten_features(raw_features)
    scaled_features = scaler.transform(raw_features)
    final_features = apply_selected_features(scaled_features, selected_features)

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(final_features)[0]
        classes = classifier.classes_
        pred = classes[np.argmax(proba)]
    else:
        pred = classifier.predict(final_features)[0]
        proba = None

    return {
        "pred": int(pred), "proba": proba,
        "raw_features": raw_features, "scaled_features": scaled_features,
        "final_features": final_features, "selected_features": selected_features,
        "raw_feature_shape": raw_features.shape, "scaled_feature_shape": scaled_features.shape,
        "final_feature_shape": final_features.shape, "used_features": final_features.shape[1],
        "pipeline": (
            "Manual knee joint area crop -> resize with padding 224x224 -> CLAHE -> "
            "VGG19 preprocess_input -> extractor.h5 -> "
            "StandardScaler semua fitur -> GA selected features -> Random Forest"
        )
    }

# ============================================================
# Prediction Display
# ============================================================

def make_probability_dataframe(classifier, proba):
    classes = getattr(classifier, "classes_", np.arange(len(proba)))
    rows = []
    for cls, prob in zip(classes, proba):
        try:
            cls_int = int(cls)
        except Exception:
            cls_int = cls
        rows.append({
            "Class Asli Model": cls,
            "Label Tampilan": CLASS_NAMES.get(cls_int, f"Class {cls_int}"),
            "Probabilitas": float(prob)
        })
    return pd.DataFrame(rows)

def make_prediction_status(proba):
    if proba is None:
        return "Confidence tidak tersedia."

    sorted_idx = np.argsort(proba)[::-1]
    top1, top2 = int(sorted_idx[0]), int(sorted_idx[1])
    confidence, margin = float(proba[top1]), float(proba[top1] - proba[top2])

    if confidence >= 0.75 and margin >= 0.15:
        status = "Prediksi kuat."
    elif confidence >= 0.55:
        status = "Prediksi sedang. Perlu verifikasi."
    else:
        status = "Prediksi lemah. Wajib diverifikasi."

    if margin < 0.15:
        status += " Selisih Top-1 dan Top-2 kecil, kemungkinan kelas berdekatan."
    return status

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="KOA Severity Classification", layout="wide")
st.title("Klasifikasi Tingkat Keparahan Knee Osteoarthritis")
st.caption(
    "Aplikasi prediksi grade KOA dengan pipeline VGG19, Genetic Algorithm, "
    "dan Random Forest. Data primer dicrop ke area sendi lutut agar menyerupai "
    "format data sekunder training/testing."
)

with st.sidebar:
    st.header("Konfigurasi Model")
    st.write("Folder model:")
    st.code(str(MODEL_DIR))

    st.write("Classifier:")
    st.code(CLASSIFIER_PATH.name)
    st.caption(file_status_text(CLASSIFIER_PATH))

    st.write("Scaler:")
    st.code(SCALER_PATH.name)
    st.caption(file_status_text(SCALER_PATH))

    st.write("Mask fitur GA:")
    st.code(FEATURE_MASK_PATH.name)
    st.caption(file_status_text(FEATURE_MASK_PATH))

    st.write("Extractor:")
    st.code(EXTRACTOR_PATH.name)
    st.caption(file_status_text(EXTRACTOR_PATH))

    st.divider()
    st.header("Preprocessing Data Primer")
    st.write("Input model:")
    st.code("manual knee joint crop -> resize with padding 224x224 -> CLAHE -> VGG19 preprocess_input")

    center_x_ratio = st.slider("Posisi pusat crop horizontal", 0.30, 0.70, 0.50, 0.01)
    center_y_ratio = st.slider("Posisi pusat crop vertikal", 0.30, 0.80, 0.52, 0.01)
    crop_scale = st.slider("Ukuran crop terhadap sisi terpendek citra", 0.40, 1.00, 0.85, 0.01)

    st.caption(
        "Atur slider sampai area crop mirip data sekunder: femoral condyle di atas, "
        "joint space di tengah, dan tibial plateau di bawah."
    )

    st.write("Normalisasi:")
    st.code(NORMALIZATION_MODE)

    st.write("CLAHE:")
    st.code(f"clipLimit={CLAHE_CLIP_LIMIT}, tileGridSize={CLAHE_TILE_GRID_SIZE}")

    st.divider()
    st.info("Prototipe ini bukan pengganti diagnosis dokter spesialis radiologi.")

try:
    classifier_loaded, scaler_loaded, selected_features_loaded, extractor_loaded = load_assets()
    model_ready = True
    st.success("Model siap digunakan. Extractor memakai vgg19_attention_extractor.h5.")
except Exception as e:
    model_ready = False
    st.error("Model belum siap dimuat.")
    st.exception(e)
    with st.expander("Detail error teknis"):
        st.code(traceback.format_exc())

uploaded_file = st.file_uploader(
    "Upload satu citra radiografi lutut",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    left_col, right_col = st.columns([1, 1])

    try:
        data = preprocess_uploaded_image(
            uploaded_file, center_x_ratio=center_x_ratio,
            center_y_ratio=center_y_ratio, crop_scale=crop_scale
        )

        with left_col:
            st.subheader("Preview Preprocessing")
            st_image_compatible(
                data["image_with_crop"],
                caption="Citra asli dengan kotak crop area sendi lutut. Garis hijau menunjukkan pusat crop."
            )
            st_image_compatible(
                data["crop_image"],
                caption="Hasil crop area sendi lutut sebelum resize dengan padding."
            )
            st_image_compatible(
                data["resized_crop"],
                caption="Hasil mempertahankan proporsi asli dengan padding menjadi 224x224, sebelum CLAHE."
            )

            with st.expander("Lihat input sebenarnya yang masuk ke model"):
                st_image_compatible(
                    data["model_image"],
                    caption="Input model: crop area sendi lutut -> resize padding 224x224 -> CLAHE."
                )
                st.write("Crop bbox:", data["bbox"])

        if model_ready:
            with st.spinner("Melakukan prediksi."):
                result = predict_koa(data["batch"])

            pred_grade = result["pred"]
            proba = result["proba"]

            with right_col:
                st.subheader("Hasil Prediksi")
                st.metric("Prediksi Grade", CLASS_NAMES.get(pred_grade, f"Class {pred_grade}"))
                st.write(GRADE_DESCRIPTIONS.get(pred_grade, "Deskripsi class belum tersedia."))
                st.write(f"Jumlah fitur yang digunakan: {result['used_features']}")
                st.caption(f"Pipeline fitur: {result['pipeline']}")

                if proba is not None:
                    confidence = float(np.max(proba))
                    st.metric("Confidence", f"{confidence:.2%}")
                    st.info(make_prediction_status(proba))

                    classifier, _, _, _ = load_assets()
                    prob_df = make_probability_dataframe(classifier, proba)

                    chart_df = prob_df.copy()
                    chart_df["Grade"] = chart_df["Label Tampilan"]

                    st.subheader("Probabilitas Tiap Grade")
                    st.bar_chart(chart_df.set_index("Grade")["Probabilitas"])

                    display_df = prob_df.copy()
                    display_df["Probabilitas"] = display_df["Probabilitas"].map(lambda x: f"{x:.4f}")
                    st_dataframe_compatible(display_df)

                    sorted_idx = np.argsort(proba)[::-1]
                    top1, top2 = int(sorted_idx[0]), int(sorted_idx[1])

                    st.write("Top-1:", CLASS_NAMES.get(top1, top1), f"{float(proba[top1]):.4f}")
                    st.write("Top-2:", CLASS_NAMES.get(top2, top2), f"{float(proba[top2]):.4f}")
                else:
                    st.warning("Classifier tidak memiliki predict_proba.")

                with st.expander("Debug teknis"):
                    classifier, scaler, selected_features, extractor = load_assets()
                    st.write("Raw feature shape:", result["raw_feature_shape"])
                    st.write("Scaled feature shape:", result["scaled_feature_shape"])
                    st.write("Final feature shape:", result["final_feature_shape"])
                    st.write("Classifier classes_:", getattr(classifier, "classes_", "Tidak ada classes_"))
                    st.write("Jumlah selected features:", len(np.asarray(selected_features).ravel()))
                    st.write("Extractor file:", EXTRACTOR_PATH.name)

                    selected_feature_df = make_selected_feature_dataframe(
                        result["raw_features"], result["scaled_features"], result["selected_features"]
                    )
                    st.write("Fitur terpilih GA untuk citra ini:")
                    st_dataframe_compatible(selected_feature_df)

                    st.download_button(
                        label="Download fitur terpilih GA sebagai CSV",
                        data=selected_feature_df.to_csv(index=False).encode("utf-8"),
                        file_name="selected_features_ga_prediction.csv",
                        mime="text/csv"
                    )

        st.divider()
        st.caption("Hasil prediksi bersifat komputasional dan perlu diverifikasi oleh dokter spesialis radiologi.")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        st.exception(e)
        with st.expander("Detail error teknis"):
            st.code(traceback.format_exc())
else:
    st.info("Silakan upload satu file citra radiografi lutut berformat JPG, JPEG, atau PNG.")