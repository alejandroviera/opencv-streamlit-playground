import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64


def app():
    # Create application title and file uploader widget.
    st.title("Applying image filters")
    img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
    select_box = st.selectbox("Filter", ("None", "B&W", "Sepia", "Vignette", "Canny Edges", "Embossed Edges", "Sketch", "Sketch BW", "Stylization"))
    conf_threshold1 = st.slider("Threshold1", min_value=0.0, max_value=255.0, step=1.0, value=100.0)
    conf_threshold2 = st.slider("Threshold2", min_value=0.0, max_value=255.0, step=1.0, value=200.0)

    if img_file_buffer is not None:
        # Read the file and convert it to opencv Image.
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        # Loads image in a BGR channel order.
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        # Or use PIL Image (which uses an RGB channel order)
        # image = np.array(Image.open(img_file_buffer))

        # Create placeholders to display input and output images.
        placeholders = st.columns(2)
        # Display Input image in the first placeholder.
        placeholders[0].image(image, channels='BGR')
        placeholders[0].text("Input Image")

        filtered_image = apply_filter(select_box, image, conf_threshold1, conf_threshold2)
        placeholders[1].image(filtered_image)
        placeholders[1].text("Filtered Image")
        placeholders[1].markdown(get_image_download_link(filtered_image, img_file_buffer.name, 'Download Image'),
                    unsafe_allow_html=True)


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    imagearray = Image.fromarray(img)
    imagearray.save(buffered, format="PNG")
    buffer = buffered.getvalue()

    img_str = base64.b64encode(buffer).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}_filtered.png">{text}</a>'
    return href


def apply_filter(select_box, image, conf_threshold1, conf_threshold2):
    if select_box == "None":
        return image[:,:,::-1]
    elif select_box == "B&W":
        return apply_bw(image)
    elif select_box == "Sepia":
        return apply_sepia(image)
    elif select_box == "Vignette":
        return apply_vignette(image, 3)
    elif select_box == "Canny Edges":
        return apply_canny(image, conf_threshold1, conf_threshold2)
    elif select_box == "Embossed Edges":
        return apply_embossed_edges(image)
    elif select_box == "Sketch":
        return apply_sketch(image, conf_threshold1, conf_threshold2)
    elif select_box == "Sketch BW":
        return apply_sketch_bw(image, conf_threshold1, conf_threshold2)
    elif select_box == "Stylization":
        return apply_stylization(image, conf_threshold1, conf_threshold2)


def apply_bw(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sepia(image):
    img_sepia = image.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB) 
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype = np.uint8)
    # img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia

def apply_vignette(img, level = 2):
    
    height, width = img.shape[:2]  
    
    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)
        
    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T 
    mask = kernel / kernel.max()
    
    img_vignette = np.copy(img)
        
    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask
    
    return img_vignette[:,:,::-1]

def apply_canny(img, conf_threshold1, conf_threshold2):
    img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
    img_edges = cv2.Canny(img_blur, conf_threshold1, conf_threshold2)
    return cv2.bitwise_not(img_edges)


def apply_embossed_edges(img):
    
    kernel = np.array([[0, -3, -3], 
                       [3,  0, -3], 
                       [3,  3,  0]])
    
    img_emboss = cv2.filter2D(img, -1, kernel=kernel)
    return img_emboss[:,:,::-1]


def apply_sketch(img, value1, value2):
    img_blur = img
    _, img_sketch = cv2.pencilSketch(img_blur, sigma_s=value1*200.0/255.0, sigma_r=value2/255.0)
    return img_sketch[:,:,::-1]


def apply_sketch_bw(img, value1, value2):
    img_blur = img
    img_sketch_bw, _ = cv2.pencilSketch(img_blur, sigma_s=value1*200.0/255.0, sigma_r=value2/255.0)
    return img_sketch_bw


def apply_stylization(img, value1, value2):
    # img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
    img_blur = img
    img_stylized = cv2.stylization(img_blur, sigma_s=value1*200.0/255.0, sigma_r=value2/255.0)
    return img_stylized[:,:,::-1]