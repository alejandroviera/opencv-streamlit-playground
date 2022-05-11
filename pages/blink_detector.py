from contextlib import nullcontext
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
import av
import cv2
import numpy as np
from multipage import MultiPage

class VideoFaceTransformer:
    def __init__(self):
        self.net = load_model()
        self.landmark_detector = load_landmark_detector()
        self.flip_enabled = True
        self.frame_count = 0
        self.frame_calib = 30  # Number of frames to use for threshold calibration.
        self.sum_ear = 0
        self.state_curr = 'open'
        self.state_prev = 'open'
        self.blinks = 0


    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")
        if img is not None:
            if self.flip_enabled:
                img = cv2.flip(img, 1)
                
            detections = detectFaceOpenCVDnn(self.net, img)
            _, faces = process_detections(img, detections, 0.8)
            if len(faces) > 0:
                primary_face = get_primary_face(faces, img.shape[0], img.shape[1])
                if primary_face is not None:
                    #cv2.rectangle(img, primary_face, (0, 255, 0), 3)
                    retval, landmarksList = self.landmark_detector.fit(img, np.expand_dims(primary_face, 0))
                    if retval:
                        landmarks = landmarksList[0][0]
                        visualize_eyes(img, landmarks)
                        ear = get_eye_aspect_ratio(landmarks)
                        if self.frame_count < self.frame_calib:
                            self.sum_ear += ear
                            self.frame_count += 1
                        elif self.frame_count == self.frame_calib:
                            self.frame_count += 1
                            self.avg_ear = self.sum_ear / self.frame_count
                            self.higher_threshold = 0.9 * self.avg_ear
                            self.lower_threshold = 0.7 * self.avg_ear
                        else:
                            if ear < self.lower_threshold:
                                self.state_curr = 'closed'
                            elif ear > self.higher_threshold:
                                self.state_curr = 'open'
                            if self.state_prev == 'closed' and self.state_curr == 'open':
                                self.blinks += 1

                        self.state_prev = self.state_curr
                        cv2.putText(img, "Blink counter: {}".format(self.blinks), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        output = []
        output.append(av.VideoFrame.from_ndarray(img, format="bgr24"))
        return output


def app():
    # Create application title and file uploader widget.
    st.title("Blink Detection from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using a Caffe Model (Deep Learning)</i></small>", unsafe_allow_html=True)
    flip_enabled = st.checkbox("Flip image", value=True)

    if MultiPage.localhost:
        #configuration for localhost
        rtc_context = webrtc_streamer(key="face_video", video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})
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
        rtc_context = webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoFaceTransformer, media_stream_constraints={"video": True, "audio": False})

    if rtc_context.video_processor:
        rtc_context.video_processor.flip_enabled = flip_enabled


def visualize_eyes(image, landmarks):
    for i in range(36, 48):
        cv2.circle(image, tuple(landmarks[i].astype('int')), 2, (0,255,0), -1)

def calculate_distance(A, B):
    return ((A[0] - B[0])**2 + (A[1] - B[1])**2)**0.5


def get_eye_aspect_ratio(landmarks):
    vertical_distance_right1 = calculate_distance(landmarks[37], landmarks[41])
    vertical_distance_right2 = calculate_distance(landmarks[38], landmarks[40])
    vertical_distance_left1 = calculate_distance(landmarks[43], landmarks[47])
    vertical_distance_left2 = calculate_distance(landmarks[44], landmarks[46])

    horizontal_distance_left = calculate_distance(landmarks[36], landmarks[39])
    horizontal_distance_right = calculate_distance(landmarks[42], landmarks[45])

    EAR_LEFT = (vertical_distance_left1 + vertical_distance_left2) / (2 * horizontal_distance_left)
    EAR_RIGHT = (vertical_distance_right1 + vertical_distance_right2) / (2 * horizontal_distance_right)

    return (EAR_LEFT + EAR_RIGHT) / 2


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
    
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2 - x1, y2 - y1])
    return frame, bboxes


def get_primary_face(faces, frame_h, frame_w):
    primary_face_index = None
    face_height_max = 0
    for idx in range(len(faces)):
        face = faces[idx]
        # Confirm bounding box of primary face does not exceed frame size.
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]
        if x1 > frame_w or y1 > frame_h or x2 > frame_w or y2 > frame_h:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        if face[3] > face_height_max:
            primary_face_index = idx

    if primary_face_index is not None:
        primary_face = faces[primary_face_index]
    else:
        primary_face = None

    return primary_face


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/face_detection/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def load_landmark_detector():
    landmarkDetector = cv2.face.createFacemarkLBF()
    landmarkDetector.loadModel('models/face_detection/lbfmodel.yaml')
    return landmarkDetector