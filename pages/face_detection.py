import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64


def app():
    # Create application title and file uploader widget.
    st.title("Face Detection from a photo")
    st.markdown("<small><i>Implemented with OpenCV using a Caffe Model (Deep Learning)</i></small>", unsafe_allow_html=True)
    img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if img_file_buffer is not None:
        net = load_model()
        blur_enabled = st.checkbox("Blur faces")
        
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

        # Create a Slider and get the threshold from the slider.
        conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

        # Call the face detection model to detect faces in the image.
        detections = detectFaceOpenCVDnn(net, image)

        # Process the detections based on the current confidence threshold.
        out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold, blur_enabled=blur_enabled)

        # Display Detected faces.
        placeholders[1].image(out_image, channels='BGR')
        placeholders[1].text("Output Image")

        # Convert opencv image to PIL.
        out_image = Image.fromarray(out_image[:, :, ::-1])
        # Create a link for downloading the output file.
        st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
                    unsafe_allow_html=True)



# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections


# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5, blur_enabled=False):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    bb_line_thickness = max(1, int(round(frame_h / 200)))
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])

            if not blur_enabled:
                # Draw bounding boxes around detected faces.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
            else:
                frame = blur_face(frame, x1, y1, x2, y2)
    return frame, bboxes


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/face_detection/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href



def blur(face, factor = 3):
    h, w = face.shape[:2]
    if factor < 1: factor = 1
    if factor > 5: factor = 5

    kernel_w = int(w/factor)
    kernel_h = int(h/factor)

    #ensure kernel size is odd
    if kernel_h %2 == 0: kernel_h += 1
    if kernel_w %2 == 0: kernel_w += 1

    return cv2.GaussianBlur(face, (kernel_w, kernel_h), 0, 0)


def pixelate(roi, pixels=16):
    roi_h, roi_w = roi.shape[:2]
    if roi_h > pixels and roi_w > pixels:
        roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
        roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_pixelated = roi

    return roi_pixelated


def blur_face(image, x1: int, y1: int, x2: int, y2: int, factor=3, pixels=10):
    img_out = image.copy()
    img_temp = image.copy()
    
    correct_y1 = (y1 if y1 > 0 else 0)
    correct_x1 = (x1 if x1 > 0 else 0)
    face = image[correct_y1:y2, correct_x1:x2, :]
    face = blur(face, factor=factor)
    face = pixelate(face, pixels=pixels)
    img_temp[correct_y1:y2, correct_x1:x2, :] = face

    elliptical_mask = np.zeros(image.shape, dtype=image.dtype)
    ellipsis_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
    ellipisis_size = (x2-x1, y2-y1)
    ellipsis_angle = 0.0
    cv2.ellipse(elliptical_mask, (ellipsis_center, ellipisis_size, ellipsis_angle), (255, 255, 255), -1, cv2.LINE_AA)
    
    np.putmask(img_out, elliptical_mask, img_temp)
    return img_out