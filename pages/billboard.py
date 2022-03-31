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

STREAMLIT_STATIC_PATH = ''
DOWNLOADS_PATH = ''

def app():
    STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

    # We create a downloads directory within the streamlit static asset directory
    # and we write output files to it.
    DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
    if not DOWNLOADS_PATH.is_dir():
        DOWNLOADS_PATH.mkdir()

    # Set title.
    st.title('Replace Billboards')

    # Specify canvas parameters in application
    base_image_file = st.sidebar.file_uploader("Upload image with billboards:", type=["png", "jpg"])
    billboard_image_file = st.sidebar.file_uploader("Upload your image for the billboard:", type=["png", "jpg"])
    base_image = None

    if billboard_image_file is not None:

        if base_image_file is None:
            base_image_file = 'base_billboard.jpg'
            base_image = cv2.imread(base_image_file)
        else:
            # Convert the file to an opencv image.
            base_file_bytes = np.asarray(bytearray(base_image_file.read()), dtype=np.uint8)
            base_image = cv2.imdecode(base_file_bytes, 1)

        billboard_file_bytes = np.asarray(bytearray(billboard_image_file.read()), dtype=np.uint8)
        billboard_image = cv2.imdecode(billboard_file_bytes, 1)

        stroke_width = 5
        h, w = base_image.shape[:2]
        if w > 800:
            h_, w_ = int(h * 800 / w), 800
        else:
            h_, w_ = h, w
        factor = float(h) / h_
        st.text("Draw four points for the four corners of where you want your billboard. Start with the top left corner, and continue clockwise.")

        # Create a canvas component.
        canvas_result = st_canvas(
            fill_color='white',
            stroke_width=stroke_width,
            stroke_color='yellow',
            background_image=Image.open(base_image_file).resize((h_, w_)),
            update_streamlit=True,
            height=h_,
            width=w_,
            drawing_mode='point',
            key="canvas",
        )
        stroke = canvas_result.image_data

        if stroke is not None and canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) >= 4:
            size = billboard_image.shape
            src_pts = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)

            roi_points = []
            for object in canvas_result.json_data["objects"]:
                roi_points.append([(object["left"] + 3) * factor, (object["top"] + 3) * factor])

            roi_dst = np.vstack(roi_points).astype(float)

            # Compute the homography.
            homography, _ = cv2.findHomography(src_pts, roi_dst[:4])
            warped_img = cv2.warpPerspective(billboard_image, homography, (base_image.shape[1], base_image.shape[0]))
            cv2.fillConvexPoly(base_image, roi_dst.astype(int), 0, 16) # Black out polygonal area first
            img_dst = cv2.add(base_image, warped_img)
            output_image = Image.fromarray(img_dst[:,:,::-1])
            st.image(output_image)
            st.sidebar.markdown(get_image_download_link(output_image, 'billboard.png', 'Download'), unsafe_allow_html=True)
            



def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href
