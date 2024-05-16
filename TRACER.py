import streamlit as st
from PIL import Image
import io
from rembg import remove

def remove_bg(input_image):
    # Convert PIL Image to bytes
    with io.BytesIO() as buf:
        input_image.save(buf, format='PNG')
        input_bytes = buf.getvalue()

    # Use rembg to remove the background
    output_bytes = remove(input_bytes)

    # Convert output bytes to PIL Image
    output_image = Image.open(io.BytesIO(output_bytes))

    return output_image

def main():
    st.title("Background Removal for Images and Videos with Streamlit")

    option = st.sidebar.selectbox("Select Option", ["Image", "Video"])

    if option == "Video":
        st.subheader("Video Background Removal")
        st.write("This feature is not yet implemented. Check back later!")

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
