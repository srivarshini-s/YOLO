import streamlit as st
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from PIL import Image
import io

# Initialize Firebase app
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(r"C:\Users\S SRIVARSHINI\Downloads\igniters-street-auto-firebase-adminsdk-uuiih-533b0bb74d.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://your-database-url.firebaseio.com/'  # Update this URL
        })
    return firestore.client()

# Streamlit app
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Header and subheader
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔍 Person and Vehicle Detection with YOLO</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #ff7f0e;'>📊 Detecting and Sending Data to Firebase</h2>", unsafe_allow_html=True)

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_path = 'uploaded_image.jpg'
    image.save(image_path)

    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Load a pretrained model

    # Perform inference on the uploaded image
    results = model(image_path, save=True)

    # Define class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

    # Initialize counters for detected classes
    person_count = 0
    bicycle_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    total_vehicles = 0

    # Iterate through each detection result
    for result in results:
        boxes = result.boxes
        cls = boxes.cls.tolist()

        # Count occurrences of each class
        person_count += cls.count(class_names.index('person'))
        bicycle_count += cls.count(class_names.index('bicycle'))
        car_count += cls.count(class_names.index('car'))
        motorcycle_count += cls.count(class_names.index('motorcycle'))
        bus_count += cls.count(class_names.index('bus'))

    # Adjust person count to exclude bicycle and motorcycle detections
    person_count -= (bicycle_count + motorcycle_count)
    total_vehicles = bicycle_count + car_count + bus_count + motorcycle_count

    # Define a single color and style for all text
    text_color = "#4a4a4a"  # Dark gray for professional look
    text_style = "font-family: Arial, sans-serif; font-size: 18px; font-weight: bold;"

    # Display results with consistent style
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of people detected: {person_count}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of bicycles detected: {bicycle_count}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of cars detected: {car_count}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of motorcycles detected: {motorcycle_count}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of buses detected: {bus_count}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Total vehicles detected: {total_vehicles}</h3>", unsafe_allow_html=True)

    # Push data to Firebase
    db = initialize_firebase()
    ref = db.collection('AI').document('chromepet')

    data = {
        "person_count": person_count,
        "bicycle_count": bicycle_count,
        "car_count": car_count,
        "motorcycle_count": motorcycle_count,
        "bus_count": bus_count
    }

    ref.set(data)
    st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Data pushed successfully!</h3>", unsafe_allow_html=True)
