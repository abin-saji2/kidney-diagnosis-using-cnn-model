import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("kidney_model.h5")
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Kidney AI", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Kidney AI")
page = st.sidebar.radio("Navigation", ["🏠 Prediction", "ℹ️ About"])

# ---------------- PDF FUNCTION ----------------
def create_pdf(name, age, gender, state, phone, result, confidence):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Kidney Disease Report", ln=True, align='C')

    pdf.ln(10)

    # Patient details
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"State: {state}", ln=True)
    pdf.cell(200, 10, f"Phone: {phone}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)

    pdf.ln(10)

    # Result
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Prediction Result:", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Condition: {result}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(10)

    # Disclaimer
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, "Note: This is an AI-based prediction and not a medical diagnosis.")

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path

# ================== HOME PAGE ==================
if page == "🏠 Prediction":

    st.markdown("""
    <h1 style='text-align:center;'>🩺 Kidney Disease Detection</h1>
    <p style='text-align:center;'>Upload CT Scan • AI Prediction • Download Report</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Patient details
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col2:
        state = st.text_input("State")
        phone = st.text_input("Phone Number")

    st.markdown("---")

    uploaded_file = st.file_uploader("📤 Upload Kidney Image", type=["jpg", "png", "jpeg"])
    analyze = st.button("🔍 Analyze Image")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        if analyze:
            if not name or not state or not phone:
                st.warning("Please fill all patient details")
                st.stop()

            # Preprocess
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            pred = model.predict(img_array)
            confidence = float(np.max(pred) * 100)
            result = classes[np.argmax(pred)]

            with col2:
                st.subheader("📊 Prediction Result")

                # Chart
                probabilities = pred[0] * 100
                df = pd.DataFrame({
                    "Condition": classes,
                    "Probability (%)": probabilities
                })

                st.bar_chart(df.set_index("Condition"), use_container_width=True)

                if confidence < 70:
                    st.warning("⚠️ Low confidence. Consult doctor.")
                else:
                    st.success(f"Prediction: {result}")
                    st.info(f"Confidence: {confidence:.2f}%")

                    info = {
                        "Cyst": "Fluid-filled sac in kidney.",
                        "Normal": "No abnormality detected.",
                        "Stone": "Hard mineral deposits.",
                        "Tumor": "Abnormal growth."
                    }

                    st.write(info[result])

                    # PDF download
                    pdf_file = create_pdf(name, age, gender, state, phone, result, confidence)

                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            label="📄 Download Report",
                            data=f,
                            file_name="kidney_report.pdf",
                            mime="application/pdf"
                        )

    st.markdown("---")
    st.warning("⚠️ This is an AI-based prediction and not a medical diagnosis.")

# ================== ABOUT PAGE ==================
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.write("""
    This is an AI-powered kidney disease detection system.

    🔹 Uses Deep Learning (CNN)  
    🔹 Classifies CT scan images  
    🔹 Provides prediction + confidence  
    🔹 Generates downloadable PDF reports  

    ⚠️ This is not a medical diagnosis tool.
    """)