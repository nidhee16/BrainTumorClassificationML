import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import base64

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"  # full-page layout
)

# ----------- Custom CSS Styling -----------
st.markdown("""
    <style>
    .stApp {
        background-color: #F1E1FA;
        font-family: 'Segoe UI', sans-serif;
        color: #4a306d;
    }

    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #4a306d !important;
        text-align: center;
    }

    .upload-box {
        background-color: #D9C7DD;
        padding: 2em;
        border-radius: 15px;
        margin: 2em auto;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .result-box {
        background-color: #fec5bb;
        color: #240046;
        padding: 1.5em;
        border-radius: 12px;
        margin: 1em auto;
        text-align: center;
    }

    .confidence-box {
        background-color: #d8e2dc;
        color: #240046;
        padding: 1.5em;
        border-radius: 12px;
        margin: 1em auto;
        text-align: center;
    }

    .footer {
        text-align: center;
        color: #5c5c5c;
        margin-top: 3em;
        font-size: 0.9em;
    }

    </style>
""", unsafe_allow_html=True)


# ----------- Load Model -----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model_best.h5")
    return model

model = load_model()
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# ----------- Title Section -----------
st.markdown("<h1>🧠 Brain Tumor Detection using CNN</h1>", unsafe_allow_html=True)

# ----------- Upload Section -----------
st.markdown("""
<div class="upload-box">
    <h2>📤 Upload Your MRI Scan</h2>
    <p>Upload a brain MRI image (JPG/PNG) to detect the presence of a tumor.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ----------- Prediction Section -----------
if uploaded_file is not None:
    st.markdown("---")
    
    # Open and resize the image for preview
    img = Image.open(uploaded_file)
    img.thumbnail((600, 600))  # Resize to fit within 600px

    # Save the image temporarily to display with HTML
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    encoded = base64.b64encode(byte_im).decode()

    # Center the image using HTML and custom styling
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{encoded}" alt="Uploaded MRI" width="600"/>
            <p style='color:#4a306d; font-weight: bold;'>🧠 Uploaded MRI Scan Preview</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Set text color to dark (#0e273c)
    st.markdown("<h3 style='color:#0e273c;'>🔬 Analyzing the Scan...</h3>", unsafe_allow_html=True)

    with st.spinner("🔍 Running Deep Learning Model... Please wait"):
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))  # Resize for model input
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success("✅ MRI Scan analyzed successfully!")

    # ----------- Results -----------
    st.markdown(f"""
        <div class="result-box">
            <h4>Prediction</h4>
            <h2>{predicted_class.replace('_', ' ').title()}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="confidence-box">
            <h4>Confidence Score</h4>
            <h2>{confidence:.2f}%</h2>
        </div>
    """, unsafe_allow_html=True)

# ----------- Footer -----------
st.markdown("""
    <hr>
    <div class="footer">
        Made with ❤️ for Brain Tumor Awareness
    </div>
""", unsafe_allow_html=True)
