import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import pkcs1_15

FOOTER_HEIGHT = 50

def app():
    # Create application title and file uploader widget.
    st.title("Sign Photo")
    img_file_buffer = st.file_uploader("Choose a file to sign", type=['jpg', 'jpeg', 'png'])
    
    # Create placeholders to display input and output images.
    placeholders = st.columns(2)

    if img_file_buffer is not None:
        image = get_opencv_image(img_file_buffer)

        # Display Input image in the first placeholder.
        placeholders[0].image(image, channels='BGR')
        placeholders[0].text("Input Image") 

        # Call the face detection model to detect faces in the image.
        signed_image = sign_image(image)

        # Display Detected faces.
        placeholders[1].image(signed_image, channels='BGR')
        placeholders[1].text("Output Image")

        # Convert opencv image to PIL.
        #signed_image = Image.fromarray(signed_image[:, :, ::-1])
        # Create a link for downloading the output file.
        placeholders[1].markdown(get_image_download_link(signed_image, "signed_"+img_file_buffer.name, 'Download Signed Image'),
                    unsafe_allow_html=True)


def get_opencv_image(uploaded_file):
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, 1)
    return image


def sign_image(image):
    _, width, _ = image.shape
    footer = np.zeros((FOOTER_HEIGHT, width, 3), np.uint8)
    signature = get_hash_from_image(image)
    footer = cv2.putText(footer, signature.hex(), (10,FOOTER_HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    signature_bytes = np.asarray(bytearray(signature))

    for i in range(len(signature_bytes)):
        footer[0,i,0] = signature_bytes[i]
    return np.vstack([image, footer])


def get_hash_from_image(image):
    digest = SHA256.new(bytearray(image))
    with open("photo_sign.key", "r") as myfile:
        private_key = RSA.import_key(myfile.read(), "opencvtests")
        signer =  pkcs1_15.new(private_key)
        result = signer.sign(digest)

    return result


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    imagearray = Image.fromarray(img[:, :, ::-1])
    imagearray.save(buffered, format="PNG")
    buffer = buffered.getvalue()

    img_str = base64.b64encode(buffer).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}.png">{text}</a>'
    return href