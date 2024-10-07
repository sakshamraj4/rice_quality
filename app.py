import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def crop_and_remove_black(image, threshold):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def find_top_left_corner(img_array):
        rows, cols = img_array.shape
        for row in range(rows):
            if np.mean(img_array[row, :]) > threshold:
                for col in range(cols):
                    if np.mean(img_array[:, col]) > threshold:
                        return row + 30, col + 30
        return 0, 0

    def find_bottom_right_corner(img_array):
        rows, cols = img_array.shape
        for row in range(rows - 1, -1, -1):
            if np.mean(img_array[row, :]) > threshold:
                for col in range(cols - 1, -1, -1):
                    if np.mean(img_array[:, col]) > threshold:
                        return row - 30, col - 30
        return rows - 1, cols - 1

    top_left = find_top_left_corner(gray_img)
    bottom_right = find_bottom_right_corner(gray_img)
    
    return image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], top_left, bottom_right

def threshold_image(gray):
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=2)
    return thresh_image

def edge_detection(thresh):
    edges = cv2.Canny(thresh, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    return edges_closed

def remove_noise(edges):
    kernel = np.ones((3, 3), np.uint8)
    edges_eroded = cv2.erode(edges, kernel, iterations=1)
    return edges_eroded

def contours_overlap(contour_a, contour_b):
    rect_a = cv2.boundingRect(contour_a)
    rect_b = cv2.boundingRect(contour_b)
    return not (rect_a[0] + rect_a[2] < rect_b[0] or rect_a[0] > rect_b[0] + rect_b[2] or
                rect_a[1] + rect_a[3] < rect_b[1] or rect_a[1] > rect_b[1] + rect_b[3])

def unique_contours(contours):
    return [contour for i, contour in enumerate(contours) 
            if all(not np.array_equal(contour, other_cont) for j, other_cont in enumerate(contours) if i != j)]

def find_contours(masked_image, min_area):
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_image = masked_image.shape[0] * masked_image.shape[1]
    
    filtered_contours = [contour for contour in contours
                         if len(contour) >= 5 and 0.5 * area_image >= cv2.contourArea(contour) > min_area]
    
    non_overlapping_contours = []
    for i, contour_a in enumerate(filtered_contours):
        overlap = False
        for j, contour_b in enumerate(filtered_contours):
            if i != j and contours_overlap(contour_a, contour_b):
                overlap = True
                if cv2.arcLength(contour_a, True) > cv2.arcLength(contour_b, True):
                    non_overlapping_contours.append(contour_a)
                break
        if not overlap:
            non_overlapping_contours.append(contour_a)
    
    return unique_contours(non_overlapping_contours)

def process_contour(contour, original_image, pad):
    x, y, w, h = cv2.boundingRect(contour)
    image_height, image_width = original_image.shape[:2]
    padded_x = max(x - pad, 0)
    padded_y = max(y - pad, 0)
    padded_w = min(w + 2 * pad, image_width - padded_x)
    padded_h = min(h + 2 * pad, image_height - padded_y)
    cropped_img = original_image[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]
    
    # Calculate size in KB and MB
    _, buffer = cv2.imencode('.png', cropped_img)
    image_size_kb = len(buffer) / 1024
    image_size_mb = image_size_kb / 1024
    
    return padded_w, padded_h, image_size_kb, image_size_mb, cropped_img

def process_image(original_image, min_area):
    image_info = []
    
    # Crop and remove black borders
    cropped_image, _, _ = crop_and_remove_black(original_image, threshold=150)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    thresh_image = threshold_image(gray_image)
    
    # Apply edge detection
    edges = edge_detection(thresh_image)
    
    # Remove noise
    cleaned_edges = remove_noise(edges)
    
    # Find contours
    contours = find_contours(cleaned_edges, min_area)
    
    # Process contours in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_contour, contour, cropped_image, 20) for contour in contours]
        results = [future.result() for future in futures]
    
    # Combine results
    for i, result in enumerate(results):
        w_new, h_new, size_kb, size_mb, cropped_img = result
        image_info.append((i, w_new, h_new, size_kb, size_mb, cropped_img))
    
    return image_info

# Function to save selected images to a zip file with higher quality
def save_to_zip(selections):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "selected_grains.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, (name, image) in enumerate(selections):
                image_filename = f"{name}.png"
                image_path = os.path.join(tmpdirname, image_filename)
                cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # Compression quality set to maximum (0-9)
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
    
    # Display the original image
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

    # Process the image and get selected images
    image_info = process_image(original_image, min_area)
    
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
            st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption=f"Grain {idx + 1}")
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
