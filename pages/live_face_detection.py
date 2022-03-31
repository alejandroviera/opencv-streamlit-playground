import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
import av
import cv2
import numpy as np

class VideoFaceTransformer:
    def __init__(self):
        self.net = load_model()
        self.blur_enabled = False
        self.flip_enabled = True


    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")
        if img is not None:
            if self.flip_enabled:
                img = cv2.flip(img, 1)
                
            detections = detectFaceOpenCVDnn(self.net, img)
            img, _ = process_detections(img, detections, blur_enabled=self.blur_enabled)
        output = []
        output.append(av.VideoFrame.from_ndarray(img, format="bgr24"))
        return output


def app():
    # Create application title and file uploader widget.
    st.title("Face Detection from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using a Caffe Model (Deep Learning)</i></small>", unsafe_allow_html=True)
    blur_enabled = st.checkbox("Blur faces")
    flip_enabled = st.checkbox("Flip image", value=True)
    
    #configuration for localhost
    rtc_context = webrtc_streamer(key="face_video", video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})
    
    # configuration for Cloud environment with a TURN/STUN server
    #RTC_CONFIGURATION = RTCConfiguration(
    #{
    #  "iceServers": [{
    #    "urls": ["turn:turn.alejandroviera.com:5349"],
    #    "username": "aviera",
    #    "credential": "rtcpassword",
    #  }]
    #})
    #rtc_context = webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})

    if rtc_context.video_processor:
        rtc_context.video_processor.blur_enabled = blur_enabled
        rtc_context.video_processor.flip_enabled = flip_enabled



# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    mean = [104, 117, 123]
    shape = (300, 300)
    scale = 1.0
    blob = cv2.dnn.blobFromImage(frame, scale, shape, mean, False, False)
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


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/face_detection/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net