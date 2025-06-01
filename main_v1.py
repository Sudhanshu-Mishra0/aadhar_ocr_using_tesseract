import streamlit as st
from streamlit_lottie import st_lottie
import requests
import cv2
import pytesseract
import numpy as np
from PIL import Image
import time

# Update this path to your Tesseract installation if needed
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

# ---------- Load Lottie Animation ----------
def load_lottie_url(url: str):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Failed to load animation: {e}")
    return None

lottie_scan = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Aadhaar OCR Reader", layout="centered")
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Aadhaar OCR Reader</h2>", unsafe_allow_html=True)
st.write("Upload front and back images of Aadhaar to extract information.")

col1, col2 = st.columns(2)
with col1:
    front_img = st.file_uploader("Upload Aadhaar Front", type=["jpg", "jpeg", "png", "webp"], key="front")
with col2:
    back_img = st.file_uploader("Upload Aadhaar Back", type=["jpg", "jpeg", "png", "webp"], key="back")

# ---------- Preprocessing & OCR ----------
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def enhanced_preprocessing(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    gaussian = cv2.GaussianBlur(filtered, (9, 9), 10.0)
    sharpened = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11
    )
    return thresh

def process_image(uploaded_img, label):
    st.markdown(f"### {label}")
    img = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if is_blurry(img_cv):
        st.warning("⚠️ Image is too blurry. Please upload a clearer scan.")
        return None

    st.image(img_np, caption=f"{label} (Original)", use_container_width=True)

    if lottie_scan:
        st_lottie(lottie_scan, height=180, key=f"{label}_anim")
    else:
        st.info("🔄 Scanning image...")

    processed = enhanced_preprocessing(img_cv)
    st.image(processed, caption=f"{label} (Enhanced for OCR)", use_container_width=True)

    text = pytesseract.image_to_string(processed)
    return text.strip().replace("\n", " ")

# ---------- Main ----------
if front_img and back_img:
    with st.spinner("🔍 Scanning Aadhaar Images..."):
        time.sleep(1.5)
        front_text = process_image(front_img, "Front Side")
        back_text = process_image(back_img, "Back Side")

        if front_text and back_text:
            result = {
                "front": front_text,
                "back": back_text
            }
            st.success("✅ OCR Completed")
            st.json(result)
