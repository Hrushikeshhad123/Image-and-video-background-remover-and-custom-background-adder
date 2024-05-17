import streamlit as st
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import tempfile
import os
from PIL import Image
import io
import numpy as np
from backgroundremover import remove

# Function to remove background of an image using backgroundremover
def remove_bg(input_image):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    # Save the input image to the temporary input file
    input_image.save(temp_input.name)

    # Use backgroundremover to remove the background
    remove(
        src_img_path=temp_input.name,
        out_img_path=temp_output.name,
        model_name="u2net",
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_structure_size=10,
        alpha_matting_base_size=1000
    )

    # Open the output image
    output_image = Image.open(temp_output.name)

    # Clean up temporary files
    temp_input.close()
    temp_output.close()
    os.remove(temp_input.name)
    os.remove(temp_output.name)

    return output_image

def main():
    st.title("Background Removal for Images and Videos with Streamlit")

    option = st.sidebar.selectbox("Select Option", ["Image", "Video"])

    if option == "Video":
        st.subheader("Video Background Removal")
        st.write("Video background removal is not supported in this version.")

    elif option == "Image":
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
