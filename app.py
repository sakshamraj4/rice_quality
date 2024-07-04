import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Function to resize the image if it's too large
def resize_image(image, max_width=1024, max_height=1024):
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        scale_factor = min(max_width / w, max_height / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale_factor

def process_image(resized_image, og_image, scale_factor, min_area):
    image_info = []  # List to store image information (size in pixels and file size)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding (using mean method)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, blockSize=15, C=2)

    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    def process_contour(contour):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        x_og = int(x / scale_factor)
        y_og = int(y / scale_factor)
        w_new = int(w / scale_factor)
        h_new = int(h / scale_factor)
        
        if w_new > 150 and h_new > 150:        
            # Crop the region
            cropped_image = og_image[y_og:y_og + h_new, x_og:x_og + w_new]
            
            # Encode image to bytes
            _, buffer = cv2.imencode('.png', cropped_image)
            image_size_kb = len(buffer) / 1024  # Size in KB
            image_size_mb = image_size_kb / 1024  # Size in MB
            
            return (w_new, h_new, image_size_kb, image_size_mb, cropped_image)
        else:
            return None

    # Process contours in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour) for contour in filtered_contours]
        results = [future.result() for future in futures]

    # Combine results
    for i, result in enumerate(results):
        if result is None:
            continue
        w_new, h_new, size_kb, size_mb, cropped_image = result
        image_info.append((i, w_new, h_new, size_kb, size_mb, cropped_image))
    
    return image_info

# Function to save selected images to a zip file with higher quality
def save_to_zip(selections):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "selected_grains.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, (name, image) in enumerate(selections):
                image_filename = f"{name}.png"
                image_path = os.path.join(tmpdirname, image_filename)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Save the image with higher quality
                cv2.imwrite(image_path, rgb_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # Compression quality set to maximum (0-9)
                zipf.write(image_path, image_filename)
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
    return zip_data

# Streamlit app
st.set_page_config(layout="wide")  # Utilize full screen

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://i.imgur.com/8a7Ujv8.jpg");
        background-size: cover;
        color: #0000FF;
    }
    .sidebar .sidebar-content {
        background: #393e46;
        color: #00adb5;
    }
    .css-18e3th9 {
        padding: 20px;
    }
    .css-1d391kg p {
        color: #0000FF;
    }
    .stButton button {
        background-color: #00adb5;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #00adb5;
    }
    .css-1offfwp {
        background-color: #00adb5;
        color: white;
    }
    .css-1l3cr7v img {
        border: 2px solid #00adb5;
        border-radius: 4px;
        padding: 5px;
        background: #393e46;
    }
    .css-1offfwp .css-1l3cr7v p {
        color: #0000FF;
    }
    .selectbox, .checkbox, .file_uploader {
        color: #00adb5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌾Grain Extraction")

# Instructions
with st.expander("Instructions", expanded=True):
    st.markdown("<p style='color:#0000FF;'>1. Upload an image containing grains.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0000FF;'>2. Select the grains you want to extract.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0000FF;'>3. Click the 'Extract' button to download the selected grains as a ZIP file.</p>", unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # User input for minimum area
    min_area = st.number_input("Enter the minimum area for grains to be extracted:", value=400, min_value=0)

    # Define size filter options
    size_filter = st.selectbox("Select size range to display images:",
                               ["All", "0-0.03 MB", "0.031-0.06 MB", "0.061-0.09 MB", ">0.09 MB"])

    # Select all checkbox
    select_all = st.checkbox("Select All", key="select_all", value=False)

    # Button to extract selected images
    extract_button = st.button("Extract")

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    # Resize image for display
    resized_image,scalefactor = resize_image(original_image)

    # Display the resized image
    st.image(original_image, caption='Original Image', use_column_width=True)

    # Process the resized image and get selected images
    image_info = process_image(resized_image, original_image, scalefactor, min_area)
    
    # Filter images based on selected size range
    if size_filter == "0-0.03 MB":
        filtered_images = [info for info in image_info if info[4] <= 0.03]
    elif size_filter == "0.031-0.06 MB":
        filtered_images = [info for info in image_info if 0.031 <= info[4] <= 0.06]
    elif size_filter == "0.061-0.09 MB":
        filtered_images = [info for info in image_info if 0.061 <= info[4] <= 0.09]
    elif size_filter == ">0.09 MB":
        filtered_images = [info for info in image_info if info[4] > 0.09]
    else:
        filtered_images = image_info

    # Display filtered images
    max_images_per_row = 4
    checkboxes = []

    # Calculate and display the total number of grains extracted
    total_grains_extracted = len(image_info)
    st.write(f"<p style='color:#0000FF;'>Total number of grains extracted: {total_grains_extracted}</p>", unsafe_allow_html=True)

    for i, (idx, w, h, size_kb, size_mb, cropped_image) in enumerate(filtered_images):
        if i % max_images_per_row == 0:
            col = st.columns(max_images_per_row)  # Create a new row
        with col[i % max_images_per_row]:
            st.image(cropped_image, caption=f"Grain {idx + 1}")
            st.write(f"<p style='color:#0000FF;'>Size: {w} x {h} pixels</p>", unsafe_allow_html=True)
            st.write(f"<p style='color:#0000FF;'>File Size: {size_kb:.2f} KB ({size_mb:.2f} MB)</p>", unsafe_allow_html=True)
            checkbox = st.checkbox(f"Select Grain {idx + 1}", key=f"select_{idx}", value=select_all)
            checkboxes.append((checkbox, (f"Grain {idx + 1}", cropped_image)))

    # Update selections based on checkboxes
    selections = [img_info for selected, img_info in checkboxes if selected]

    # Display the total number of selected images
    total_selected_images = len(selections)
    st.write(f"<p style='color:#0000FF;'>Total number of selected images: {total_selected_images}</p>", unsafe_allow_html=True)

    # Handle the extract button click event
    if extract_button:
        if selections:
            zip_data = save_to_zip(selections)
            st.sidebar.download_button(label="Download ZIP", data=zip_data, file_name="selected_grains.zip", mime="application/zip")
        else:
            st.warning("No grains selected. Please select at least one grain.")
