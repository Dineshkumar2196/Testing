import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io

# RGB color for light gray background
light_gray_color = (211, 211, 211)

# Extract PDF to Img
def pdf_page_to_image(pdf_file, page_number):
    """Convert a PDF page to a PIL Image."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image

# Create Bounding Box & Highlighting Img - Start
def draw_bounding_box(image, x_min, y_min, x_max, y_max, color):
    """Draw a bounding box on the image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
    return image

def highlight_pink_green_blue_and_orange_pixels(image, x_min, y_min, x_max, y_max, light_gray_color):
    """Highlight pixels within the pink, green, blue, and orange color ranges inside the bounding box."""
    np_image = np.array(image)
    highlighted_image = np_image.copy()
    
    cropped_image = np_image[y_min:y_max, x_min:x_max]
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    
    # Define HSV ranges for pink
    lower_pink1 = np.array([140, 50, 50])
    upper_pink1 = np.array([180, 255, 255])
    
    lower_pink2 = np.array([0, 20, 97])
    upper_pink2 = np.array([10, 255, 255])
    
    lower_pink3 = np.array([160, 100, 200])
    upper_pink3 = np.array([180, 255, 255])
    
    lower_pink4 = np.array([0, 60, 240])
    upper_pink4 = np.array([10, 255, 255])
    
    # Define HSV ranges for green
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    lower_pure_green = np.array([60, 255, 255])
    upper_pure_green = np.array([60, 255, 255])
    
    # Define HSV ranges for blue
    lower_blue = np.array([97, 200, 200])
    upper_blue = np.array([105, 255, 255])
    
    # Define HSV ranges for orange and light orange
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    
    lower_light_orange = np.array([10, 50, 50])
    upper_light_orange = np.array([30, 255, 255])

    # Create masks for pink
    mask_pink1 = cv2.inRange(hsv_image, lower_pink1, upper_pink1)
    mask_pink2 = cv2.inRange(hsv_image, lower_pink2, upper_pink2)
    mask_pink3 = cv2.inRange(hsv_image, lower_pink3, upper_pink3)
    mask_pink4 = cv2.inRange(hsv_image, lower_pink4, upper_pink4)
    mask_pink = cv2.bitwise_or(mask_pink1, mask_pink2)
    mask_pink = cv2.bitwise_or(mask_pink, mask_pink3)
    mask_pink = cv2.bitwise_or(mask_pink, mask_pink4)

    # Create masks for green
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_pure_green = cv2.inRange(hsv_image, lower_pure_green, upper_pure_green)
    
    # Create masks for blue
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Create masks for orange and light orange
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    mask_light_orange = cv2.inRange(hsv_image, lower_light_orange, upper_light_orange)
    mask_orange_combined = cv2.bitwise_or(mask_orange, mask_light_orange)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_pink, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_pure_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_orange_combined)

    # Highlight pixels in the bounding box that match any of the masks
    highlighted_image[y_min:y_max, x_min:x_max][combined_mask == 0] = light_gray_color
    
    return Image.fromarray(highlighted_image)

# Create Bounding Box & Highlighting Img - End

# Pressure Point Detection - Start
def contains_green_blue_and_orange_colors(image, x_min, y_min, x_max, y_max):
    """Check if there are green, blue, and orange colors within the bounding box."""
    np_image = np.array(image)
    cropped_image = np_image[y_min:y_max, x_min:x_max]
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    
    # Define HSV ranges for green, blue, and orange
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    lower_blue = np.array([97, 200, 200])
    upper_blue = np.array([105, 255, 255])
    
    lower_orange = np.array([20, 50, 50])
    upper_orange = np.array([25, 255, 255])
    
    # Create masks for green, blue, and orange
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    
    # Check if each color is present in the bounding box
    contains_green = np.any(mask_green)
    contains_blue = np.any(mask_blue)
    contains_red = detect_red_solid_line_in_bounding_box(image, x_min, y_min, x_max, y_max)
    contains_orange = np.any(mask_orange)

    conditions = (contains_green, contains_blue, contains_red, contains_orange)
    
    if conditions == (True, True, True, True):
        return "Intermediate, Medium, Low and High Pressure Point detected"        
    elif conditions == (True, True, True, False):
        return "Intermediate, Medium and Low Pressure Point detected"
    elif conditions == (True, True, False, True):
        return "Intermediate, Medium and High Pressure Point detected"
    elif conditions == (True, False, True, True):
        return "Intermediate, Low and High Pressure Point detected"
    elif conditions == (False, True, True, True):
        return "Medium, Low and High Pressure Point detected"        
    elif conditions == (True, True, False, False):
        return "Intermediate and Medium Pressure Point detected"
    elif conditions == (True, False, True, False):
        return "Intermediate and Low Pressure Point detected"
    elif conditions == (True, False, False, True):
        return "Intermediate and High Pressure Point detected"        
    elif conditions == (False, True, True, False):
        return "Medium and Low Pressure Point detected"        
    elif conditions == (False, True, False, True):
        return "Medium and High Pressure Point detected"        
    elif conditions == (False, False, True, True):
        return "Low and High Pressure Point detected"        
    elif conditions == (True, False, False, False):
        return "Intermediate Pressure Point detected"
    elif conditions == (False, True, False, False):
        return "Medium Pressure Point detected"
    elif conditions == (False, False, True, False):
        return "Low Pressure Point detected"
    elif conditions == (False, False, False, True):
        return "High Pressure Point detected"        
    else:
        return "No Pressure Point detected"
        
def detect_red_solid_line_in_bounding_box(image, x_min, y_min, x_max, y_max):
    """Detect if there is a red solid line within the bounding box."""
    np_image = np.array(image)
    cropped_image = np_image[y_min:y_max, x_min:x_max]
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    # Define HSV range for red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Threshold to avoid detecting small noise
            return True
    return False
# Pressure Point Detection - End

# Streamlit app
st.title("PDF and Image Processing App")

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    # Set default values
    page_number = st.number_input("Select the page number", min_value=0, max_value=fitz.open(stream=pdf_file.read(), filetype="pdf").page_count - 1, step=1, value=0)
    x_min = st.number_input("X Min", min_value=0, step=1, value=8)
    y_min = st.number_input("Y Min", min_value=0, step=1, value=10)
    x_max = st.number_input("X Max", min_value=0, step=1, value=585)
    y_max = st.number_input("Y Max", min_value=0, step=1, value=585)
    
    # Reset file pointer to the start
    pdf_file.seek(0)
    
    # Convert PDF to image
    image = pdf_page_to_image(pdf_file, page_number)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Highlight Colors"):
        highlighted_image = highlight_pink_green_blue_and_orange_pixels(image, x_min, y_min, x_max, y_max, light_gray_color)
        st.image(highlighted_image, caption="Highlighted Image", use_column_width=True)

    if st.button("Detect Pressure Points"):
        result = contains_green_blue_and_orange_colors(image, x_min, y_min, x_max, y_max)
        st.write(f"Pressure Point Detection Result: {result}")
