import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import face_detection, live_face_detection, photo_signing, photo_verification, \
        image_filters, image_restoration, billboard, panorama, object_classification, \
        blink_detector, object_detection, pose_detection, hands_detection

st.set_page_config(page_title="Computer Vision",page_icon=":eyeglasses:",layout="wide")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Create an instance of the app 
app = MultiPage()

# Add all your application here
app.add_page("Photo Face Detection", face_detection.app)
app.add_page("Video Face Detection", live_face_detection.app)
app.add_page("Blink Detection", blink_detector.app)
app.add_page("Object Detection", object_detection.app)
app.add_page("Human Pose Detection", pose_detection.app)
app.add_page("Hands Detection", hands_detection.app)
app.add_page("Sign Photo", photo_signing.app)
app.add_page("Verify Photo Signature", photo_verification.app)
app.add_page("Image Filters", image_filters.app)
app.add_page("Image Restoration", image_restoration.app)
app.add_page("Virtual Billboards", billboard.app)
app.add_page("Create Panoramas", panorama.app)
app.add_page("Object Classification", object_classification.app)

# The main app
app.run()