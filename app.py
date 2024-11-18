import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configure the page layout
st.set_page_config(
    page_title="NeuraBHOS AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Updated CSS with improved colors and styling
st.markdown("""
    <style>
    /* Global Styles */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        padding: 2rem;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 6px;
        background-color: #2563eb;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    
    /* Header Styles */
    .header-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .brain-icon {
        color: #2563eb;
        font-size: 2.5rem;
    }
    
    .app-title {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Team Card Styles */
    .team-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .team-title {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .team-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }
    
    /* Improved Upload Section Styles */
    .upload-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Override Streamlit's default upload widget styling */
    .stFileUploader > div > div {
        background-color: white !important;
        border: 2px dashed #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div > div:hover {
        background-color: #f8fafc !important;
        border-color: #2563eb !important;
    }
    
    /* Hide the default streamlit upload widget background */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    
    /* Results Section Styles with improved colors */
    .results-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .result-positive {
        background-color: #dc2626;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .result-negative {
        background-color: #16a34a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #1e293b;
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Additional styling to ensure white background for upload area */
    [data-testid="stFileUploadDropzone"] {
        background-color: white !important;
        border: 2px dashed #e2e8f0 !important;
        color: #64748b !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #2563eb !important;
        background-color: #f8fafc !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cached model loading function
@st.cache_resource
def load_model():
    """Load the converted SavedModel with caching to prevent reloading"""
    try:
        if not os.path.exists('saved_model'):
            st.error("Model directory 'saved_model' not found. Please run convert_model.py first.")
            return None
        model = tf.saved_model.load('saved_model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model input"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, image):
    """Make prediction using the loaded model"""
    try:
        infer = model.signatures['serving_default']
        input_name = list(infer.structured_input_signature[1].keys())[0]
        image_tensor = tf.convert_to_tensor(image)
        prediction = infer(**{input_name: image_tensor})
        output_name = list(prediction.keys())[0]
        pred_value = prediction[output_name].numpy()
        if len(pred_value.shape) > 1:
            pred_value = pred_value[0]
        if len(pred_value.shape) > 0:
            pred_value = pred_value[0]
        return pred_value
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <div class="brain-icon">🧠</div>
            <div>
                <h1 class="app-title">NeuraBHOS AI</h1>
                <p class="app-subtitle">Advanced Brain Tumor Detection System</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Team Information
    st.markdown("""
        <div class="team-card">
            <h2 class="team-title">
                <span>👥</span>
                Project Team
            </h2>
            <div class="team-grid">
                <div>
                    <h3 style="color: #64748b; font-size: 0.875rem;">Team Members:</h3>
                    <p style="color: #1e293b;">Rahil Najafov<br>Murad Hummatli</p>
                </div>
                <div>
                    <h3 style="color: #64748b; font-size: 0.875rem;">Project Advisor:</h3>
                    <p style="color: #1e293b;">Assoc. Prof. Leyla Muradkhanli</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for upload and results
    col1, col2 = st.columns(2)

    # Load model
    model = load_model()

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1e293b; font-size: 1.25rem;">Upload MRI Scan</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is None:
            st.markdown("""
                <div class="upload-zone">
                    <p style="color: #64748b;">
                        Drag and drop your MRI scan here<br>
                        <span style="font-size: 0.875rem;">Supported formats: JPG, PNG</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            analyze_button = st.button("Analyze Scan")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1e293b; font-size: 1.25rem;">Analysis Results</h3>', unsafe_allow_html=True)
        
        if uploaded_file is not None and analyze_button:
            if model is not None:
                with st.spinner('Analyzing MRI scan...'):
                    try:
                        processed_image = preprocess_image(image)
                        if processed_image is not None:
                            pred_value = predict(model, processed_image)
                            
                            if pred_value is not None:
                                confidence = pred_value * 100 if pred_value > 0.5 else (1 - pred_value) * 100
                                
                                if pred_value > 0.5:
                                    st.markdown("""
                                        <div class="result-positive">
                                            <h4 style="margin: 0;">⚠️ Tumor Detected</h4>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                        <div class="result-negative">
                                            <h4 style="margin: 0;">✅ No Tumor Detected</h4>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style="text-align: center;">
                                        <p style="color: #64748b; margin-bottom: 0.5rem;">Confidence</p>
                                        <p class="confidence-value">{confidence:.1f}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.button("Download Full Report")
                                
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
            else:
                st.error("Model not loaded. Please ensure the model is properly converted and available.")
        else:
            st.markdown("""
                <div style="height: 200px; display: flex; align-items: center; justify-content: center;">
                    <p style="color: #64748b;">Upload and analyze an MRI scan to see results</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2024 NeuraBHOS AI - Brain Tumor Detection Project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
