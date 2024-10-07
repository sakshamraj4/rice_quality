import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Function to find the top-left corner of the white box
def find_top_left_corner(img_array, threshold):
    rows, cols = img_array.shape

    top_row, top_col = 0, 0
    for row in range(rows):
        row_values = img_array[row, :]
        if np.mean(row_values) > threshold:
            top_row = row
            break

    for col in range(cols):
        col_values = img_array[:, col]
        if np.mean(col_values) > threshold:
            top_col = col
            break

    return top_row + 30, top_col + 30

# Function to find the bottom-right corner of the white box
def find_bottom_right_corner(img_array, threshold):
    rows, cols = img_array.shape

    bottom_row, bottom_col = rows - 1, cols - 1

    # Start from the bottom row and move upwards to find the first row with white pixels
    for row in range(rows - 1, -1, -1):
        row_values = img_array[row, :]
        if np.mean(row_values) > threshold:
            bottom_row = row
            break

    # Start from the rightmost column and move left to find the first column with white pixels
    for col in range(cols - 1, -1, -1):
        col_values = img_array[:, col]
        if np.mean(col_values) > threshold:
            bottom_col = col
            break

    return bottom_row - 30, bottom_col - 30


def crop_and_remove_black(image, threshold):
    # Convert the image to grayscale and numpy array again
    gray_img = image.convert("L")
    gray_img_array = np.array(gray_img)

    # Find the top-left and bottom-right corners of this cropped part
    top_left = find_top_left_corner(gray_img_array, threshold)
    bottom_right = find_bottom_right_corner(gray_img_array, threshold)

    return image.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0])), top_left, bottom_right

# Now, crop the image into four parts and remove internal black boundaries
def crop_blackbox(img,grid_flag):
    if grid_flag:
        # Save each quadrant
        quadrants = []
        x_off = []
        y_off = []
        image = img
        black_threshold = 150

        width, height = image.size
        crop_width = width // 2
        crop_height = height // 2

        part1, part1_top, part1_bottom = crop_and_remove_black(image.crop((0, 0, crop_width, crop_height)), black_threshold)
        part1_x1, part1_y1, part1_x2, part1_y2 = part1_top[0], part1_top[1], part1_bottom[0], part1_bottom[1]
        x_off.append(part1_top[0])
        y_off.append(part1_top[1])
        quadrants.append(part1)

        part2, part2_top, part2_bottom = crop_and_remove_black(image.crop((crop_width, 0, width, crop_height)), black_threshold)
        part2_x1, part2_y1, part2_x2, part2_y2 = part2_top[0], crop_width+part2_top[1], part2_bottom[0], crop_width+part2_bottom[1]
        x_off.append(part2_top[0])
        y_off.append(crop_width+part2_top[1])
        quadrants.append(part2)

        part3, part3_top, part3_bottom = crop_and_remove_black(image.crop((0, crop_height, crop_width, height)), black_threshold)
        part3_x1, part3_y1, part3_x2, part3_y2 = crop_height+part3_top[0], part3_top[1], crop_height+part3_bottom[0], part3_bottom[1]
        x_off.append(crop_height+part3_top[0])
        y_off.append(part3_top[1])
        quadrants.append(part3)

        part4, part4_top, part4_bottom = crop_and_remove_black(image.crop((crop_width, crop_height, width, height)), black_threshold)
        part4_x1, part4_y1, part4_x2, part4_y2 = crop_height+part4_top[0], crop_width+part4_top[1], crop_height+part4_bottom[0], crop_width+part4_bottom[1]
        x_off.append(crop_height+part4_top[0])
        y_off.append(crop_width+part4_top[1])
        quadrants.append(part4)

    else:
      quadrants=[img]
      x_off=[0]
      y_off=[0]

    return quadrants, x_off, y_off

def contours_overlap(contour_a, contour_b):
    # Implement logic to determine if contours overlap
    # For simplicity, you can compare the bounding rectangles
    rect_a = cv2.boundingRect(contour_a)
    rect_b = cv2.boundingRect(contour_b)
    overlap = not (rect_a[0] + rect_a[2] < rect_b[0] or rect_a[0] > rect_b[0] + rect_b[2] or
                   rect_a[1] + rect_a[3] < rect_b[1] or rect_a[1] > rect_b[1] + rect_b[3])
    return overlap


def unique_contours(contours):
    unique = []
    for contour in contours:
        if all(not np.array_equal(contour, unique_cont) for unique_cont in unique):
            unique.append(contour)
    return unique

def translate_contours_to_original(contours, x_offset,y_offset):

    translated_contours = []
    for contour in contours:
      translated_contour = [(point[0][0] + y_offset, point[0][1] + x_offset) for point in contour]
      translated_contours.append(translated_contour)

    return translated_contours

def resize_image(image, max_width=1024, max_height=1024):
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        scale_factor = min(max_width / w, max_height / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale_factor

# Apply adaptive thresholding to segment grains and coin
def threshold_image(gray):
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding (using mean method)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=2)

    # _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh_image

# Apply Canny Edge Detection to find edges
def edge_detection(thresh):
    edges = cv2.Canny(thresh, 50, 150)

    # Apply dilation to connect broken edges
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for dilation
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Apply closing to fill small gaps and solidify contours
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

    return edges_closed

# Erosion to remove small noise
def remove_noise(edges):
    # Apply erosion to reduce noise inside contours
    kernel = np.ones((3, 3), np.uint8)
    edges_eroded = cv2.erode(edges, kernel, iterations=1)

    return edges_eroded

# Find contours in the masked image
def find_contours(masked_image, scale_factor, area_image, min_area):
    contours1_2 = []

    # Find contours
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [contour for contour in contours
        if len(contour) >= 5
        and 0.5 * area_image >= cv2.contourArea(contour) > min_area]

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

    # Deduplicate the list
    non_overlapping_contours = unique_contours(non_overlapping_contours)

    # return filtered_contours

    def scale_contours(contour):
        # Scale each point in the contour
        scaled_contour = np.array(contour, dtype=np.float32) / scale_factor
        scaled_contour = np.int32(scaled_contour)  # Convert back to integer coordinates

        return scaled_contour

    # Process contours in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(scale_contours, contour) for contour in non_overlapping_contours]
        results = [future.result() for future in futures]

    # Combine results
    for i, result in enumerate(results):
        if result is None:
            continue
        contours1_2.append(result)

    return contours1_2

def process_contour(contour, bbox_mask2, pad):
    bbox_mask2 = cv2.cvtColor(bbox_mask2, cv2.COLOR_BGR2RGB)
    x, y, w, h = cv2.boundingRect(contour)
    image_height, image_width = bbox_mask2.shape[:2]

    padded_x = max(x - pad, 0)
    padded_y = max(y - pad, 0)
    padded_w = min(w + 2 * pad, image_width - padded_x)
    padded_h = min(h + 2 * pad, image_height - padded_y)

    cropped_img = bbox_mask2[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]
    return cropped_img  # Return the cropped image directly

def process_image(original_image_color, grid_flag):

    height, width, channels = original_image_color.shape

    if grid_flag:
      clip_margin = int((8 / 100) * width)

    else:
      clip_margin=0

    
    cropped_box, x_off, y_off = crop_blackbox(original_image_color, grid_flag)
    contours1_2=[]

    for n, x_offset,y_offset in zip(cropped_box, x_off,y_off):

        # Step 1: Load the original image
        original_image = n
        
        image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        
        image, scalefactor = resize_image(image)
        
        original_image = ''
        # resized_image, scalefactor = original_image, 1
        # Calculate the area
        height, width,_ = image.shape
        area_image = width * height
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding (using mean method)
        image = threshold_image(image)
        
        image = edge_detection(image)
        
        # Step 5: Remove noise by applying erosion
        image = remove_noise(image)
        
        # Step 6: Find contours in the black background image
        contours = find_contours(image, scalefactor, area_image, 100)
        
        contours = translate_contours_to_original(contours, x_offset,y_offset)
        contours = [np.array(contour, dtype=np.int32).reshape((-1, 1, 2)) for contour in contours]
        
        contours1_2.extend(contours)
        contours = ''

    # Step 7: Draw those contours on the original image
    image = draw_contours_on_original(original_image_color, contours1_2)
    
    print(f"Number of contours found: {len(contours1_2)}")
    
    croped_image_list =list()
    for contour in contours1_2:
        croped_image_list.append(process_contour(contour, cv2.cvtColor(original_image_color,cv2.COLOR_BGR2RGB), 20))
    
    return croped_image_list, contours1_2
    
# Draw contours on the original image
def draw_contours_on_original(original_image, contours):
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 15)
    return contour_image

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
    grid_flag = st.checkbox("Grid Image", key="True_False", value=False)

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
    
    # Display the resized image
    st.image(original_image, caption='Original Image', use_column_width=True)

    # Process the resized image and get selected images
    croped_images, contours = process_image(original_image, grid_flag)

    # Display the resized image
    st.image(draw_contours_on_original(original_image, contours), caption='Original Image', use_column_width=True)
    
    # Display filtered images
    max_images_per_row = 4
    checkboxes = []

    # Calculate and display the total number of grains extracted
    total_grains_extracted = len(croped_images)
    st.write(f"<p style='color:#0000FF;'>Total number of grains extracted: {total_grains_extracted}</p>", unsafe_allow_html=True)

    i = 0
    for cropped_image in croped_images:
        if i % max_images_per_row == 0:
            col = st.columns(max_images_per_row)  # Create a new row
        with col[i % max_images_per_row]:
            st.image(cropped_image, caption=f"Grain {i + 1}")
            checkbox = st.checkbox(f"Select Grain {i + 1}", key=f"select_{i}", value=select_all)
            checkboxes.append((checkbox, (f"Grain {i + 1}", cropped_image)))

        i = i + 1

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
