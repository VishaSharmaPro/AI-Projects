import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Face Analysis AI",
    page_icon="🎭",
    layout="wide"
)

# Title and description
st.title("🎭 Complete Face Analysis - Age, Gender & Beauty Score")
st.markdown("Upload an image to predict age, gender, and facial beauty score using Deep Learning")

# Load models (cache so they load only once)
@st.cache_resource
def load_models():
    import os
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model paths
        age_gender_path = os.path.join(current_dir, 'Res_Epoch_15_Age_Gender_model.keras')
        beauty_path = os.path.join(current_dir, 'Epoch_30_Facial_Beauty_model.keras')
        
        # Check if files exist
        if not os.path.exists(age_gender_path):
            st.error(f"Age/Gender model not found at: {age_gender_path}")
            return None, None
        if not os.path.exists(beauty_path):
            st.error(f"Beauty model not found at: {beauty_path}")
            return None, None
        
        # Load models
        age_gender_model = load_model(age_gender_path)
        beauty_model = load_model(beauty_path)
        return age_gender_model, beauty_model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

age_gender_model, beauty_model = load_models()

# Gender dictionary
genderdict = {0: 'Male', 1: 'Female'}

# Function to preprocess image
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to numpy array
    img_array = np.array(img)
    # Reshape for model input
    img_array = img_array.reshape(1, 128, 128, 1)
    # Normalize
    img_array = img_array / 255.0
    return img_array

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear face image for complete analysis"
)

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        # Display original image
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔮 Analysis Results")
        
        if age_gender_model is not None and beauty_model is not None:
            # Preprocess image
            processed_img = preprocess_image(image)
            
            # Make predictions
            with st.spinner('Analyzing...'):
                # Age and Gender prediction
                age_gender_pred = age_gender_model.predict(processed_img, verbose=0)
                gender_pred = age_gender_pred[0][0][0]
                age_pred = age_gender_pred[1][0][0]
                
                # Beauty score prediction
                beauty_pred = beauty_model.predict(processed_img, verbose=0)
                beauty_score = beauty_pred[0][0]
                
                # Convert to readable format
                predicted_gender = genderdict[round(gender_pred)]
                predicted_age = round(age_pred)
                predicted_beauty = round(beauty_score, 2)
                
                # Display results with styling
                st.markdown("---")
                
                # Create 3 columns for metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    # Gender prediction
                    gender_confidence = gender_pred if predicted_gender == 'Female' else 1 - gender_pred
                    st.metric(
                        label="👤 Gender",
                        value=predicted_gender,
                        delta=f"{gender_confidence*100:.1f}% confidence"
                    )
                
                with metric_col2:
                    # Age prediction
                    st.metric(
                        label="🎂 Age",
                        value=f"{predicted_age} years"
                    )
                
                with metric_col3:
                    # Beauty score prediction
                    # Normalize beauty score to 0-10 scale for better display
                    beauty_normalized = min(10, max(0, predicted_beauty * 2))
                    st.metric(
                        label="✨ Beauty Score",
                        value=f"{predicted_beauty:.2f}/5.0"
                    )
                
                st.markdown("---")
                
                # Beauty score bar
                st.subheader("Beauty Score Visualization")
                progress_value = float(min(1.0, predicted_beauty / 5.0))  # Fixed: Convert to Python float
                st.progress(progress_value)
                
                # Beauty rating
                if predicted_beauty >= 4.0:
                    rating = "⭐⭐⭐⭐⭐ Excellent"
                    color = "green"
                elif predicted_beauty >= 3.5:
                    rating = "⭐⭐⭐⭐ Very Good"
                    color = "blue"
                elif predicted_beauty >= 3.0:
                    rating = "⭐⭐⭐ Good"
                    color = "orange"
                elif predicted_beauty >= 2.5:
                    rating = "⭐⭐ Average"
                    color = "orange"
                else:
                    rating = "⭐ Below Average"
                    color = "red"
                
                st.markdown(f"**Exact Score:** :{color}[{predicted_beauty:.4f}]")
                st.markdown(f"**Rating:** :{color}[{rating}]")
                
                st.markdown("---")
                
                # Additional info
                st.info("💡 Note: All predictions are based on facial features analysis and may vary. Beauty is subjective!")
        else:
            st.error("❌ Models not loaded. Please check your model files.")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses Deep Learning models to analyze:
    - **Gender**: Male or Female classification
    - **Age**: Estimated age in years
    - **Beauty Score**: Facial attractiveness rating (0-5 scale)
    
    ### How to use:
    1. Upload a clear face image (JPG/PNG)
    2. Wait for the AI analysis
    3. View comprehensive results
    
    ### Model Details:
    - **Architecture**: CNN (Convolutional Neural Network)
    - **Input**: 128x128 grayscale images
    - **Age/Gender Model**: 15 epochs trained
    - **Beauty Model**: 30 epochs trained
    
    ### About Beauty Score:
    - Scale: 0.0 to 5.0
    - Based on facial symmetry, features, and patterns
    - Trained on SCUT-FBP5500 dataset
    """)
    
    st.markdown("---")
    st.write("Made with ❤️ using Streamlit & TensorFlow")
    
    # Model status
    st.markdown("---")
    st.subheader("📊 Model Status")
    if age_gender_model is not None:
        st.success("✅ Age/Gender Model Loaded")
    else:
        st.error("❌ Age/Gender Model Failed")
    
    if beauty_model is not None:
        st.success("✅ Beauty Model Loaded")
    else:
        st.error("❌ Beauty Model Failed")

# Footer
st.markdown("---")
st.caption("Complete Face Analysis Web App | Powered by TensorFlow & Streamlit | AI-Based Prediction System")
