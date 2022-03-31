import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
import av
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

class VideoFaceTransformer:
    def __init__(self):
        self.net = load_model()


    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")
        if img is not None:
            img = cv2.flip(img, 1)
            detections = detectFaceOpenCVDnn(self.net, img)
            process_detections(img, detections)
        output = []
        output.append(av.VideoFrame.from_ndarray(img, format="bgr24"))
        return output


def app():
    # Create application title and file uploader widget.
    st.title("Face Detection from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using a Caffe Model (Deep Learning)</i></small>", unsafe_allow_html=True)
    
    #configuration for localhost
    webrtc_streamer(key="face_video", video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})
    
    # configuration for Cloud environment with a TURN/STUN server
    #RTC_CONFIGURATION = RTCConfiguration(
    #{
    #  "iceServers": [{
    #    "urls": ["turn:turn.alejandroviera.com:5349"],
    #    "username": "aviera",
    #    "credential": "rtcpassword",
    #  }]
    #})
    #webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})



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
def process_detections(frame, detections, conf_threshold=0.5):
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
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/face_detection/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net