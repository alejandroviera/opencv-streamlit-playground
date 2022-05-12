import cv2
import av
import streamlit as st
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
from multipage import MultiPage

class ForegroundSegmantationTransformer:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.background_image = None
        self.change_background = False

    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")
        
        if img is not None:    
            if self.flip_enabled:
                img = cv2.flip(img, 1)
                
            img = self.process_image(img)

        output = []
        output.append(av.VideoFrame.from_ndarray(img, format="bgr24"))
        return output
   
    def process_image(self, img):
        # Convert the image to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with self.mp_selfie_segmentation.SelfieSegmentation() as segment:
            # Segment the image.
            results = segment.process(img)

            # Apply the threshold to create a binary map.
            binary_mask = results.segmentation_mask > self.threshold

            # Convert the mask to a 3-channel image.
            mask = np.dstack((binary_mask, binary_mask, binary_mask))

            # Convert the image back to BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if self.change_background == True and self.background_image is not None:
                if (self.background_image.shape[1] != img.shape[1] or self.background_image.shape[0] != img.shape[0]):
                    self.background_image = cv2.resize(self.background_image, (img.shape[1], img.shape[0]))
                background = self.background_image
            else:
                background = img

            if self.blur:
                background = cv2.GaussianBlur(background, (self.kernel_size, self.kernel_size), 0)

            if self.gray_background:
                background = cv2.cvtColor(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            output_image = np.where(mask, img, background)

            return output_image

def app():
    # Create application title and file uploader widget.
    st.title("Foreground Segmentation from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using Google MediaPipe</i></small>", unsafe_allow_html=True)
    flip_enabled = st.sidebar.checkbox("Flip image", value=True)
    blur = st.sidebar.checkbox("Blur", value=False)
    gray_background = st.sidebar.checkbox("Grayscale Background", value=False)
    change_background = st.sidebar.checkbox("Change Background", value=False)

    if blur:
        kernel_size = st.sidebar.slider("Blur level", min_value=3, max_value=45, step=2, value=15)
    
    threshold = st.sidebar.slider("Threshold", min_value=0.1, max_value=1.0, step=0.01, value=0.5)

    if change_background:
        background_image = st.sidebar.file_uploader("Choose background image", type=['jpg', 'jpeg', 'png'])
    
    if MultiPage.localhost:
        #configuration for localhost
        rtc_context = webrtc_streamer(key="face_video", video_processor_factory=ForegroundSegmantationTransformer, media_stream_constraints={"video": True, "audio": False})
    else:
        # configuration for Cloud environment with a TURN/STUN server
        RTC_CONFIGURATION = RTCConfiguration(
        {
        "iceServers": [{
            "urls": ["turn:turn.alejandroviera.com:5349"],
            "username": "aviera",
            "credential": "rtcpassword",
        }]
        })
        rtc_context = webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=ForegroundSegmantationTransformer, media_stream_constraints={"video": True, "audio": False})

    if rtc_context.video_processor:
        rtc_context.video_processor.flip_enabled = flip_enabled
        rtc_context.video_processor.blur = blur
        rtc_context.video_processor.threshold = threshold
        rtc_context.video_processor.gray_background = gray_background
        rtc_context.video_processor.change_background = change_background

        if blur:
            rtc_context.video_processor.kernel_size = kernel_size
        
        if change_background and background_image is not None:
            # Read the file and convert it to opencv Image.
            background_raw_bytes = np.asarray(bytearray(background_image.read()), dtype=np.uint8)
            # Loads image in a BGR channel order.
            rtc_context.video_processor.background_image = cv2.imdecode(background_raw_bytes, cv2.IMREAD_COLOR)