import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("👤 Nhận diện khuôn mặt + Đặt tên")

# =========================
# SESSION STORAGE
# =========================
if "known_faces" not in st.session_state:
    st.session_state.known_faces = []  # list of dicts

# =========================
# FUNCTIONS
# =========================
def encode_face(image_np):
    locations = face_recognition.face_locations(image_np)
    if len(locations) != 1:
        return None, None
    encoding = face_recognition.face_encodings(image_np, locations)[0]
    return encoding, locations[0]

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["➕ Đăng ký khuôn mặt", "🔍 Nhận diện"])

# =========================
# TAB 1 – REGISTER
# =========================
with tab1:
    st.subheader("➕ Đăng ký khuôn mặt mới")

    name = st.text_input("Tên người dùng")
    img = st.camera_input("Chụp ảnh khuôn mặt")

    if st.button("💾 Lưu khuôn mặt"):
        if not name or img is None:
            st.warning("⚠️ Vui lòng nhập tên và chụp ảnh")
            st.stop()

        image = Image.open(img).convert("RGB")
        img_np = np.array(image)

        encoding, location = encode_face(img_np)

        if encoding is None:
            st.error("❌ Ảnh phải có đúng 1 khuôn mặt")
        else:
            st.session_state.known_faces.append({
                "name": name,
                "encoding": encoding
            })
            st.success(f"✅ Đã lưu khuôn mặt của {name}")

# =========================
# TAB 2 – RECOGNITION
# =========================
with tab2:
    st.subheader("🔍 Nhận diện khuôn mặt")

    if len(st.session_state.known_faces) == 0:
        st.info("ℹ️ Chưa có khuôn mặt nào được đăng ký")
        st.stop()

    img = st.camera_input("Chụp ảnh để nhận diện", key="recognize")

    if img:
        image = Image.open(img).convert("RGB")
        img_np = np.array(image)

        with st.spinner("🧠 Đang nhận diện..."):
            face_locations = face_recognition.face_locations(img_np)
            face_encodings = face_recognition.face_encodings(img_np, face_locations)

        draw = ImageDraw.Draw(image)

        known_encodings = [f["encoding"] for f in st.session_state.known_faces]
        known_names = [f["name"] for f in st.session_state.known_faces]

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                name = known_names[matches.index(True)]

            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
            draw.text((left, top - 10), name, fill="red")

        st.image(image, caption="Kết quả nhận diện", use_container_width=True)
