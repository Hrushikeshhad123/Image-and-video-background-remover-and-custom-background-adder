import streamlit as st
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import tempfile
import os
from PIL import Image
import io
from backgroundremover.bg import remove

# Function to remove background of an image using backgroundremover.bg
def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    
    with open(src_img_path, "rb") as f:
        data = f.read()
    
    img = remove(data, model_name=model_choices[0],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)

    with open(out_img_path, "wb") as f:
        f.write(img)

def main():
    st.title("Background Removal for Images and Videos with Streamlit")

    option = st.sidebar.selectbox("Select Option", ["Image", "Video"])

    if option == "Video":
        st.subheader("Video Background Removal")

        # Specify the path to the input video file
        video_file = st.file_uploader("Upload a video file", type=["mp4"])

        if video_file is not None:
            # Save the uploaded file to a temporary directory
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            # Initialize the SelfiSegmentation class. It will be used for background removal.
            # model is 0 or 1 - 0 is general 1 is landscape(faster)
            segmentor = SelfiSegmentation(model=0)

            # Initialize the video capture object
            cap = cv2.VideoCapture(video_path)

            # Set the width and height of the output video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Open a placeholder for video display
            placeholder = st.empty()

            # Infinite loop to process each frame and display the processed video
            while True:
                # Capture a single frame
                success, frame = cap.read()

                if not success:
                    break

                # Use the SelfiSegmentation class to remove the background
                processed_frame = segmentor.removeBG(frame)

                # Display the processed frame
                placeholder.image(processed_frame, channels="BGR")

                # Check for 'q' key press to break the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture object
            cap.release()

            # Close all OpenCV windows
            cv2.destroyAllWindows()

    elif option == "Image":
        st.subheader("Background Removal for Images")

        # Function to select an image
        def select_image():
            file_path = st.file_uploader("Upload Image", type=["jpg", "png"])
            if file_path is not None:
                return file_path
            return None

        uploaded_img_path = select_image()
        if uploaded_img_path is not None:
            with open(uploaded_img_path, "rb") as f:
                img_data = f.read()

            temp_output_path = "output.png"  # Temporary path for output image
            remove_bg(uploaded_img_path, temp_output_path)

            with open(temp_output_path, "rb") as f:
                bg_removed_img = Image.open(f)

            st.image(bg_removed_img, caption='Background Removed Image', use_column_width=True)

if __name__ == "__main__":
    main()
