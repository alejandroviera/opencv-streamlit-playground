from distutils.command.upload import upload
import cv2
import streamlit as st
import numpy as np

def app():
    # Create application title and file uploader widget.
    st.title("Text Recognition")
    st.warning("For natural scene text, like license plates and road signs. Not for documents.")
    uploaded_file = st.sidebar.file_uploader("Choose a text image", type="jpg")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        textDetector = getTextDetector()
        textRecognizer = getTextRecognizer()
        boxes, _ = textDetector.detect(image)
        cv2.polylines(image, boxes, True, (255, 0, 255), 3)
        st.image(image[:,:,::-1])

        st.write("Recognized Text:")
        for box in boxes:
            croppedRoi = fourPointsTransform(image, box)
            recognizedText = textRecognizer.recognize(croppedRoi)
            st.write(recognizedText)
            


def getTextDetector():
    textDetector = cv2.dnn_TextDetectionModel_DB("models/text_recognition/DB_TD500_resnet18.onnx")
    inputSize = (640, 640)
    binaryThreshold = 0.3
    polygonThreshold = 0.5
    mean = (122.67891434, 116.66876762, 104.00698793)
    textDetector.setBinaryThreshold(binaryThreshold).setPolygonThreshold(polygonThreshold)
    textDetector.setInputParams(1.0/255, inputSize, mean, True)
    return textDetector


def getTextRecognizer():
    textRecognizer = cv2.dnn_TextRecognitionModel("models/text_recognition/crnn_cs.onnx")
    textRecognizer.setDecodeType("CTC-greedy")
    vocabulary = getVocabulary()
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)
    return textRecognizer

def getVocabulary():
    vocabulary = []
    with open("models/text_recognition/alphabet_94.txt") as file:
        for line in file:
            vocabulary.append(line.strip())
    return vocabulary
    
def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]
    ], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

