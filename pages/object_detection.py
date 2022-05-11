from charset_normalizer import detect
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
from streamlit_webrtc import WebRtcMode
import av
import cv2
import numpy as np
from multipage import MultiPage

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

class ObjectDetectionTransformer:
    def __init__(self):
        self.net = load_model()
        self.flip_enabled = True
        self.objectnessThreshold = 0.4 # Objectness threshold, high values filter out low objectness
        self.confThreshold = 0.2       # Confidence threshold, high values filter out low confidence detections
        self.nmsThreshold = 0.5        # Non-maximum suppression threshold, higher values result in duplicate boxes per object 
        self.inpWidth = 416            # Width of network's input image, larger is slower but more accurate
        self.inpHeight = 416           # Height of network's input image, larger is slower but more accurate
        
        classesFile = "models/object_detection/coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')


    def getOutputsNames(self):
        """Get the names of all output layers in the network."""
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    async def recv_queued(self, frames):
        img = frames[-1].to_ndarray(format="bgr24")
        if img is not None:
            if self.flip_enabled:
                img = cv2.flip(img, 1)

            detections = self.detect_objects(img)
            self.display_objects(img, detections)

        output = []
        output.append(av.VideoFrame.from_ndarray(img, format="bgr24"))
        return output


    def detect_objects(self, img):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(img, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        return self.net.forward(self.getOutputsNames())


    def display_objects(self, frame, outs):
        """Remove the bounding boxes with low confidence using non-maxima suppression."""
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        
        # Loop through all outputs.
        for out in outs:
            for detection in out:
                if detection[4] > self.objectnessThreshold:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > self.confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
            label = "{}:{:.2f}".format(self.classes[classIds[i]], confidences[i])
            self.display_text(frame, label, left, top)


    def display_text(self, im, text, x, y):
        """Draw text onto image at location."""
        # Get text size 
        textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
        dim = textSize[0]
        baseline = textSize[1]
                
        # Use text size to create a black rectangle. 
        cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(im, text, (x, y + dim[1]), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    #modelFile = "models/object_detection/yolov4-tiny.cfg"
    #modelWeights = "models/object_detection/yolov4-tiny.weights"
    modelFile = "models/object_detection/yolov4.cfg"
    modelWeights = "models/object_detection/yolov4.weights"
    net = cv2.dnn.readNetFromDarknet(modelFile, modelWeights)
    return net

def app():
    # Create application title and file uploader widget.
    st.title("Object Detection from live video stream")
    st.markdown("<small><i>Implemented with OpenCV using a YOLOv4</i></small>", unsafe_allow_html=True)
    flip_enabled = st.checkbox("Flip image", value=True)

    if MultiPage.localhost:
        #configuration for localhost
        rtc_context = webrtc_streamer(key="face_video", video_processor_factory=ObjectDetectionTransformer, media_stream_constraints={"video": True, "audio": False})
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
        rtc_context = webrtc_streamer(key="face_video", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=ObjectDetectionTransformer, media_stream_constraints={"video": True, "audio": False})

    if rtc_context.video_processor:
        rtc_context.video_processor.flip_enabled = flip_enabled