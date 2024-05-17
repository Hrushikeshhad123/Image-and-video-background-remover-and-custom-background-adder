import streamlit as st
from PIL import Image
import io
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

def remove_bg(image: Image.Image) -> Image.Image:
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        image_np = np.array(image)
        results = selfie_segmentation.process(image_np)
        mask = results.segmentation_mask

        # Create a binary mask where segmentation mask is above a threshold
        binary_mask = mask > 0.1

        # Convert the binary mask to 3 channels
        binary_mask_3 = np.stack((binary_mask,) * 3, axis=-1)

        # Apply the mask to the input image
        fg_image = np.where(binary_mask_3, image_np, 0)

        # Convert back to PIL Image
        fg_image_pil = Image.fromarray(fg_image.astype(np.uint8))
        return fg_image_pil

def main():
    st.title("Background Removal for Images with Streamlit")

    st.subheader("Background Removal for Images")

    # Function to select an image
    def select_image():
        file_path = st.file_uploader("Upload Image", type=["jpg", "png"])
        if file_path is not None:
            img = Image.open(file_path)
            return img
        return None

    uploaded_img = select_image()
    if uploaded_img is not None:
        bg_removed_img = remove_bg(uploaded_img)
        st.image(bg_removed_img, caption='Background Removed Image', use_column_width=True)

        # Ask user to upload background image
        background_image = st.file_uploader("Upload Background Image", type=["jpg", "png"])
        if background_image is not None:
            background_img = Image.open(background_image)

            # Resize background image to match processed image dimensions
            background_img = background_img.resize(bg_removed_img.size)

            # Overlay the processed image onto the background image
            final_image = Image.alpha_composite(background_img.convert("RGBA"), bg_removed_img.convert("RGBA"))

            # Display the final image with the input image background
            st.image(final_image, caption='Processed Image with Input Image Background', use_column_width=True)

            # Add a download button for the final image with the input image background
            img_stream_with_input_bg = io.BytesIO()
            final_image.save(img_stream_with_input_bg, format="PNG")
            img_bytes_with_input_bg = img_stream_with_input_bg.getvalue()
            st.download_button(label="Download Processed Image with Input Image Background", data=img_bytes_with_input_bg, file_name="processed_image_with_input_bg.png", mime="image/png")

if __name__ == "__main__":
    main()
