import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from requests.models import MissingSchema
from io import BytesIO



def app():
    # Create application title and file uploader widget.
    st.title("OpenCV Deep Learning based Object Classification")
    img_file_buffer = st.file_uploader("Choose a file or Camera", type=['jpg', 'jpeg', 'png'])
    st.text("OR")
    url = st.text_input("Enter URL")
    image = None

    if img_file_buffer is not None:
        # Read the file and convert it to opencv Image.
        image = np.array(Image.open(img_file_buffer))
    elif url != "":
        try:
            response = requests.get(url)
            image = np.array(Image.open(BytesIO(response.content)))
        except MissingSchema as err:
            st.header("Invalid URL, try again!")
            print(err)
        except UnidentifiedImageError as err:
            st.header("URL has no valid image, try again!")
            print(err)

    if image is not None:
        st.image(image)
        net = load_model()
        class_names = load_class_names()

        # Call the face detection model to detect faces in the image.
        detections = detectObjectsOpenCVDnn(net, image).reshape(len(class_names), 1)
        out_text = describe_detection(detections, class_names)
        st.header(out_text)


# Function for detecting facses in an image.
def detectObjectsOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123), swapRB=False, crop=False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections[0]


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "models/object_classification/DenseNet_121.caffemodel"
    configFile = "models/object_classification/DenseNet_121.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def load_class_names():
    with open('models/object_classification/classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

    # Save the names of all possible classifications, removing empty final line.
    return image_net_names[:-1]

def describe_detection(detections, class_names):
    #Get the class label
    label_id = np.argmax(detections)
    class_name = class_names[label_id]

    #Convert scores to probabilities
    probs = np.exp(detections) / np.sum(np.exp(detections))
    final_prob = np.max(probs)*100

    return f"Class: {class_name}, Confidence: {final_prob:.3f}"