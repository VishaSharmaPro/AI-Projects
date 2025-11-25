import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Anime Face Generator",
    page_icon="🎨",
    layout="wide"
)

# Color mapping
COLOR_LABELS = {
    'Red': 0,
    'Blue': 1,
    'Green': 2,
    'Yellow': 3,
    'Black': 4
}

NUM_CLASSES = len(COLOR_LABELS)
LATENT_DIM = 100

# Cache the model loading
@st.cache_resource
def load_generator_model():
    try:
        model = keras.models.load_model("E:\VS CODE Coding files\AI_Projects\generator_model.keras")
        return model, None
    except Exception as e:
        return None, str(e)

# Generate images function
def generate_images(generator, color_label, num_images=1, seed=None):
    """Generate anime face images for a specific color"""
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    # Generate random noise
    noise = tf.random.normal((num_images, LATENT_DIM))
    
    # Create one-hot encoded labels
    labels = keras.utils.to_categorical([color_label] * num_images, NUM_CLASSES)
    
    # Generate images
    generated_images = generator([noise, labels], training=False)
    
    # Denormalize images from [-1, 1] to [0, 255]
    generated_images = (generated_images * 127.5) + 127.5
    generated_images = generated_images.numpy().astype('uint8')
    
    return generated_images

# Main app
def main():
    st.title("🎨 Conditional DCGAN Anime Face Generator")
    st.markdown("Generate anime faces with specific hair colors using AI")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Load model
    with st.spinner("Loading model..."):
        generator, error = load_generator_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please make sure 'generator_model.keras' is in the same directory as this script.")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # User inputs
    st.sidebar.subheader("Generation Parameters")
    
    selected_color = st.sidebar.selectbox(
        "Select Hair Color:",
        options=list(COLOR_LABELS.keys()),
        index=0
    )
    
    num_images = st.sidebar.slider(
        "Number of Images:",
        min_value=1,
        max_value=16,
        value=4,
        step=1
    )
    
    use_seed = st.sidebar.checkbox("Use Random Seed (for reproducibility)", value=False)
    seed_value = None
    
    if use_seed:
        seed_value = st.sidebar.number_input(
            "Seed Value:",
            min_value=0,
            max_value=10000,
            value=42,
            step=1
        )
    
    # Generate button
    if st.sidebar.button("🎨 Generate Images", type="primary"):
        with st.spinner(f"Generating {num_images} anime face(s) with {selected_color.lower()} hair..."):
            # Generate images
            color_label = COLOR_LABELS[selected_color]
            generated_imgs = generate_images(
                generator, 
                color_label, 
                num_images, 
                seed_value if use_seed else None
            )
            
            # Display images
            st.subheader(f"Generated {selected_color} Hair Anime Faces")
            
            # Calculate grid dimensions
            cols_per_row = min(4, num_images)
            rows = (num_images + cols_per_row - 1) // cols_per_row
            
            # Display in grid
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    img_idx = row * cols_per_row + col_idx
                    if img_idx < num_images:
                        with cols[col_idx]:
                            img = Image.fromarray(generated_imgs[img_idx])
                            st.image(img, caption=f"{selected_color} #{img_idx + 1}", use_container_width=True)
                            
                            # Download button for each image
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            st.download_button(
                                label="📥 Download",
                                data=buf.getvalue(),
                                file_name=f"anime_{selected_color.lower()}_{img_idx + 1}.png",
                                mime="image/png",
                                key=f"download_{img_idx}"
                            )
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This app uses a Conditional Deep Convolutional GAN (DCGAN) 
        to generate anime face images with specific hair colors.
        
        **Available Colors:**
        - Red
        - Blue
        - Green
        - Yellow
        - Black
        
        Each generation creates unique faces based on random noise.
        """
    )
    
    # Main page info
    if generator is not None:
        with st.expander("📊 Model Information"):
            st.write(f"**Latent Dimension:** {LATENT_DIM}")
            st.write(f"**Number of Classes:** {NUM_CLASSES}")
            st.write(f"**Output Image Size:** 64x64 pixels")
            st.write(f"**Color Classes:** {', '.join(COLOR_LABELS.keys())}")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Select the desired hair color from the sidebar
        2. Choose how many images you want to generate (1-16)
        3. (Optional) Enable random seed for reproducible results
        4. Click the **Generate Images** button
        5. Download your favorite generated images
        
        **Tips:**
        - Try different colors to see variety in styles
        - Use the same seed value to regenerate identical images
        - Generate multiple images at once to get more options
        """)

if __name__ == "__main__":
    main()