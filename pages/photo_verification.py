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
    st.title("Verify Photo")
    img_verification_file_buffer = st.file_uploader("Choose a file to verify", type=['jpg', 'jpeg', 'png', 'webp'])
    verification_placeholders = st.columns(2)
    
    if img_verification_file_buffer is not None:
        image = get_opencv_image(img_verification_file_buffer)
        
        # Display Input image in the first placeholder.
        verification_placeholders[0].image(image, channels='BGR')
        verification_placeholders[0].text("Image to Verify")

        is_valid, unsigned_image = get_unsigned_image(image)
        verification_placeholders[1].image(unsigned_image, channels='BGR')
        verification_placeholders[1].text("Image without signature")
        if is_valid:
            verification_placeholders[1].success("Image has a valid signature")
        else:
            verification_placeholders[1].error("Image has an invalid signature")

        # Convert opencv image to PIL.
        #signed_image = Image.fromarray(signed_image[:, :, ::-1])
        # Create a link for downloading the output file.
        verification_placeholders[1].markdown(get_image_download_link(unsigned_image, "unsigned_"+img_verification_file_buffer.name, 'Download Original Image'),
                    unsafe_allow_html=True)

def get_opencv_image(uploaded_file):
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, 1)
    return image


def get_unsigned_image(image):
    original_height = image.shape[0] - FOOTER_HEIGHT
    original_image = image[0:original_height,:]
    footer = image[original_height:,:]
    signature_in_footer = np.ascontiguousarray(footer[0,0:256,0])

    with open('photo_sign.pub', 'r') as myfile:
        public_key = RSA.import_key(myfile.read())
        digest = SHA256.new(bytearray(original_image))
        try:
            verifier =  pkcs1_15.new(public_key)
            verifier.verify(digest, signature_in_footer)
            is_valid = True
        except (ValueError, TypeError):
            is_valid = False

    return is_valid, original_image


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    imagearray = Image.fromarray(img[:, :, ::-1])
    imagearray.save(buffered, format="PNG")
    buffer = buffered.getvalue()

    img_str = base64.b64encode(buffer).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}.png">{text}</a>'
    return href