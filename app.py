import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

st.set_page_config(page_title="Face App", layout="centered")
st.title("👤 Nhận diện khuôn mặt (MediaPipe – Cloud Safe)")

# ======================
# STATE
# ======================
if "faces" not in st.session_state:
    st.session_state.faces = []  # {name, vector}

# ======================
# MEDIAPIPE
# ======================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# ======================
# UTILS
# ======================
def extract_vector(image_np):
    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    vector = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    vector = vector / np.linalg.norm(vector)
    return vector

def cosine_sim(a, b):
    return np.dot(a, b)

# ======================
# UI
# ======================
tab1, tab2 = st.tabs(["➕ Đăng ký", "🔍 Nhận diện"])

# ---------- REGISTER ----------
with tab1:
    name = st.text_input("Tên người dùng")
    img = st.camera_input("Chụp ảnh khuôn mặt")

    if st.button("Lưu khuôn mặt"):
        if not name or img is None:
            st.warning("Thiếu thông tin")
            st.stop()

        image = Image.open(img).convert("RGB")
        img_np = np.array(image)

        vec = extract_vector(img_np)
        if vec is None:
            st.error("Không phát hiện khuôn mặt")
            st.stop()

        st.session_state.faces.append({
            "name": name,
            "vec": vec
        })

        st.success(f"✅ Đã lưu {name}")

# ---------- RECOGNIZE ----------
with tab2:
    if not st.session_state.faces:
        st.info("Chưa có khuôn mặt nào")
        st.stop()

    img = st.camera_input("Chụp ảnh để nhận diện", key="rec")

    if img:
        image = Image.open(img).convert("RGB")
        img_np = np.array(image)
        draw = ImageDraw.Draw(image)

        vec = extract_vector(img_np)
        if vec is None:
            st.error("Không phát hiện khuôn mặt")
            st.stop()

        best_name = "Unknown"
        best_score = 0.75

        for f in st.session_state.faces:
            score = cosine_sim(vec, f["vec"])
            if score > best_score:
                best_score = score
                best_name = f["name"]

        draw.text((20, 20), f"Kết quả: {best_name}", fill="red")
        st.image(image, use_container_width=True)
