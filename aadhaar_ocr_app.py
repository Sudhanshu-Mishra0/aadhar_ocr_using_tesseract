import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_cropper import st_cropper
import requests
import cv2
import pytesseract
import numpy as np
from PIL import Image
import time
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1"

# ---------- Load Lottie ----------
def load_lottie_url(url: str):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Failed to load animation: {e}")
    return None

# ---------- OCR Utility Class ----------
class AadhaarOCR:
    def __init__(self):
        self.lottie_scan = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json")
        pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\\tesseract.exe'

    def is_blurry(self, image, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def enhanced_preprocessing(self, img_cv):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
        gaussian = cv2.GaussianBlur(filtered, (9, 9), 10.0)
        sharpened = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        thresh = cv2.adaptiveThreshold(sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
        return thresh

    def extract_text(self, img_pil):
        img_np = np.array(img_pil)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if self.is_blurry(img_cv):
            return None, "Blurry image. Please upload a clearer scan."
        processed = self.enhanced_preprocessing(img_cv)
        text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
        return text.strip().replace("\n", " "), processed

    def upload_crop_ui(self, label):
        uploaded_file = st.file_uploader(f"Upload Aadhaar {label}", type=["jpg", "jpeg", "png", "webp"], key=f"{label}_uploader")
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.write(f"✂️ Crop & Rotate the {label} image:")
            cropped_img = st_cropper(img, realtime_update=True, box_color='#4CAF50', aspect_ratio=None, return_type='image')
            st.image(cropped_img, caption=f"{label} (Cropped & Rotated)", use_container_width=True)
            return cropped_img
        return None

    def display_processing(self, label):
        if self.lottie_scan:
            st_lottie(self.lottie_scan, height=150, key=f"{label}_anim")
        else:
            st.info("🔄 Processing image...")

    def parse_text_with_llm(self, raw_text):
        prompt = f"""
        Extract and clean Aadhaar card fields like Name, DOB, Gender, Address, Aadhaar Number from the following text.
        Correct OCR mistakes if needed. Return a structured JSON with the following keys: name, dob, gender, address, aadhaar_number.
        
        Text:
        {raw_text}
        """

        try:
            response = openai.ChatCompletion.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful OCR result parser for Aadhaar cards."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error parsing with LLM: {e}"

# ---------- Streamlit App Logic ----------
def main():
    st.set_page_config(page_title="Aadhaar OCR Reader", layout="centered")
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Aadhaar OCR Reader</h2>", unsafe_allow_html=True)
    st.write("Upload front and back images of Aadhaar to extract structured information. You can crop or rotate before processing.")

    ocr = AadhaarOCR()

    col1, col2 = st.columns(2)
    with col1:
        front_img = ocr.upload_crop_ui("Front")
    with col2:
        back_img = ocr.upload_crop_ui("Back")

    if front_img and back_img:
        with st.spinner("🔍 Scanning Aadhaar Images..."):
            time.sleep(1)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Front Side")
                ocr.display_processing("Front Side")
                front_text, front_processed = ocr.extract_text(front_img)
                if front_processed is not None:
                    st.image(front_processed, caption="Front (Processed)", use_container_width=True)

            with col2:
                st.markdown("### Back Side")
                ocr.display_processing("Back Side")
                back_text, back_processed = ocr.extract_text(back_img)
                if back_processed is not None:
                    st.image(back_processed, caption="Back (Processed)", use_container_width=True)

            if front_text and back_text:
                raw_combined = f"{front_text} {back_text}"
                structured_result = ocr.parse_text_with_llm(raw_combined)
                st.success("✅ OCR Completed")
                st.markdown("### 🔍 Extracted Details")
                st.json(structured_result)
            else:
                st.warning("Something went wrong while processing the images.")

if __name__ == "__main__":
    main()
