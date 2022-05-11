import cv2
import av
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
from multipage import MultiPage

class PoseDetectionTransformer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")

        if img is not None:
            if self.flip_enabled:
                img = cv2.flip(img, 1)
        
            # Make a copy of the original image.
            annotated_img = img.copy()

            img_width = img.shape[1]
            img_height = img.shape[0]

            with self.mp_pose.Pose(static_image_mode=True) as pose:

                # Process image.
                results = pose.process(img)

                # Draw landmarks.
                circle_radius = int(.007 * img_height)  # Scale landmark circles as percentage of image height.

                # Specify landmark drawing style.
                point_spec = self.mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)

                # Specify landmark connections drawing style.
                line_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

                # Draw both landmark points and connections.
                self.mp_drawing.draw_landmarks(annotated_img,
                                        landmark_list=results.pose_landmarks,
                                        connections=self.mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=point_spec,
                                        connection_drawing_spec=line_spec)

        output = []
        output.append(av.VideoFrame.from_ndarray(annotated_img, format="bgr24"))
        return output

def app():
    # Create application title and file uploader widget.
    st.title("Pose Detection from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using Google MediaPipe</i></small>", unsafe_allow_html=True)
    flip_enabled = st.checkbox("Flip image", value=True)
    
    if MultiPage.localhost:
        #configuration for localhost
        rtc_context = webrtc_streamer(key="face_video", video_processor_factory=PoseDetectionTransformer, media_stream_constraints={"video": True, "audio": False})
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
        rtc_context = webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=PoseDetectionTransformer, media_stream_constraints={"video": True, "audio": False})

    if rtc_context.video_processor:
        rtc_context.video_processor.flip_enabled = flip_enabled