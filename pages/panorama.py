"""Application to demo inpainting using streamlit.

Run using: streamlit run 10_03_image_inpaint_streamlit.py
"""

import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image

def app():
    STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

    # Set title.
    st.title('Create Panorama')

    images = []
    img1_file = st.sidebar.file_uploader("Upload the first image:", type=["png", "jpg"])
    img2_file = st.sidebar.file_uploader("Upload the second image:", type=["png", "jpg"])

    if img1_file is not None:
        img1_bytes = np.asarray(bytearray(img1_file.read()), dtype=np.uint8)
        img1 = cv2.imdecode(img1_bytes, 1)[:,:,::-1]
        images.append(img1)
        st.image(img1)
    if img2_file is not None:
        img2_bytes = np.asarray(bytearray(img2_file.read()), dtype=np.uint8)
        img2 = cv2.imdecode(img2_bytes, 1)[:,:,::-1]
        images.append(img2)
        st.image(img2)

    if img1_file is not None and img2_file is not None:
        #images.append(cv2.cvtColor(cv2.imread('C:\\Users\\alevi\\Downloads\\IMG_20201022_084401.jpg'), cv2.COLOR_BGR2RGB))
        #images.append(cv2.cvtColor(cv2.imread('C:\\Users\\alevi\\Downloads\\IMG_20201022_084406.jpg'), cv2.COLOR_BGR2RGB))
    
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)
        if status == 0:
            img_panorama = Image.fromarray(panorama)
            st.image(img_panorama)
            st.markdown(get_image_download_link(img_panorama, 'panorama.png', 'Download panorama'), unsafe_allow_html=True)
        else:
            img1 = images[0]
            img2 = images[1]
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(500)
            keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
            keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = list(matcher.match(descriptors1, descriptors2, None))
            matches.sort(key = lambda x: x.distance, reverse=False)
            matches = matches[:int(len(matches)*0.15)]

            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                points1[i:] = keypoints1[match.queryIdx].pt
                points2[i:] = keypoints2[match.trainIdx].pt
            
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
            img1_h, img1_w = img1_gray.shape
            img2_h, img2_w = img2_gray.shape
            img2_aligned = cv2.warpPerspective(img2, h, (img2_w + img1_w, img2_h))
            img_output = Image.fromarray(img2_aligned)
            st.image(img_output)
            st.markdown(get_image_download_link(img_output, 'panorama.png', 'Download panorama'), unsafe_allow_html=True)



def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href
