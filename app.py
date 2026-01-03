import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Face Recognition - MediaPipe", layout="centered")
st.title("👤 Nhận diện khuôn mặt + Đặt tên (MediaPipe)")

# =========================
# SESSION STORAGE
# =========================
if "faces" not in st.session_state:
    st.session_state.faces = []  # {name, feature}

# =========================
# MEDIAPIPE INIT
# =========================
@st.cache_resource
def load_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

detector = load_detector()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_feature(face_img):
    face_img = cv2.resize(face_img, (100, 100))
    hist = cv2.calcHist([face_img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_features(f1, f2):
    return cv2.compareHist(f1.astype("float32"), f2.astype("float32"),
                            cv2.HISTCMP_CORREL)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["➕ Đăng ký khuôn mặt", "🔍 Nhận diện"])

# =========================
# TAB 1 – REGISTER
# =========================
with tab1:
    st.subheader("➕ Đăng ký khuôn mặt")

    name = st.text_input("Tên")
    img = st.camera_input("Chụp ảnh khuôn mặt")

    if st.button("💾 Lưu"):
        if not name or img is None:
            st.warning("⚠️ Nhập tên và chụp ảnh")
            st.stop()

        image = Image.open(img).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        results = detector.process(img_np)

        if not results.detections or len(results.detections) != 1:
            st.error("❌ Ảnh phải có đúng 1 khuôn mặt")
            st.stop()

        bbox = results.detections[0].location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        face = img_np[y1:y2, x1:x2]

        feature = extract_feature(face)
        st.session_state.faces.append({"name": name, "feature": feature})

        st.success(f"✅ Đã lưu khuôn mặt của {name}")

# =========================
# TAB 2 – RECOGNIZE
# =========================
with tab2:
    st.subheader("🔍 Nhận diện khuôn mặt")

    if len(st.session_state.faces) == 0:
        st.info("ℹ️ Chưa có dữ liệu khuôn mặt")
        st.stop()

    img = st.camera_input("Chụp ảnh để nhận diện", key="rec")

    if img:
        image = Image.open(img).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        results = detector.process(img_np)
        draw = ImageDraw.Draw(image)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                face = img_np[y1:y2, x1:x2]
                feat = extract_feature(face)

                name = "Unknown"
                best_score = 0.6

                for f in st.session_state.faces:
                    score = compare_features(feat, f["feature"])
                    if score > best_score:
                        best_score = score
                        name = f["name"]

                draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
                draw.text((x1, y1 - 10), name, fill="red")

        st.image(image, use_container_width=True)
